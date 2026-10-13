from unittest.mock import patch

import pytest

from app.chat.openai_client import ModelServerUnavailableError
from app.config import OPENAI_TRANSLATE_MODEL
from app.translate.llm_translate import (
    TranslationUnavailableError,
    coach_reply_stream,
    gloss_message,
    interpret_user_input,
    tag_draft,
    translate_text,
)


def test_translate_text_empty_input_short_circuits():
    with patch("app.translate.llm_translate.model_chat") as mock_chat:
        assert translate_text("   ", "es", "en") == ""
    mock_chat.assert_not_called()


def test_translate_text_calls_llm_with_language_names():
    with patch("app.translate.llm_translate.model_chat", return_value="Hello, how are you?") as mock_chat:
        result = translate_text("Hola, ¿cómo estás?", "es", "en")

    assert result == "Hello, how are you?"
    (messages,), kwargs = mock_chat.call_args
    assert messages[-1] == {"role": "user", "content": "Hola, ¿cómo estás?"}
    assert "Spanish" in messages[0]["content"]
    assert "English" in messages[0]["content"]
    assert kwargs["model"] == OPENAI_TRANSLATE_MODEL


def test_translate_text_strips_whitespace():
    with patch("app.translate.llm_translate.model_chat", return_value="  Hello  \n"):
        assert translate_text("Hola", "es", "en") == "Hello"


def test_translate_text_wraps_model_server_error():
    with patch("app.translate.llm_translate.model_chat", side_effect=ModelServerUnavailableError("down")):
        with pytest.raises(TranslationUnavailableError, match="down"):
            translate_text("Hola", "es", "en")


def test_interpret_user_input_empty_short_circuits():
    with patch("app.translate.llm_translate.model_chat") as mock_chat:
        assert interpret_user_input("   ", "en", "es") == ("", "")
    mock_chat.assert_not_called()


def test_interpret_user_input_parses_labeled_lines():
    reply = "English: Do you speak Spanish well?\nSpanish: ¿Hablas bien español?"
    with patch("app.translate.llm_translate.model_chat", return_value=reply) as mock_chat:
        native, target = interpret_user_input("Do you hablo the espanol good?", "en", "es")

    assert native == "Do you speak Spanish well?"
    assert target == "¿Hablas bien español?"
    (messages,), kwargs = mock_chat.call_args
    assert messages[-1] == {"role": "user", "content": "Do you hablo the espanol good?"}


def test_interpret_user_input_falls_back_when_format_not_followed():
    # Model ignores the "Label: text" instruction and just answers in prose
    # (but not so much longer than the input that the hallucination backstop
    # below also kicks in - that's covered by its own test).
    with patch(
        "app.translate.llm_translate.model_chat",
        side_effect=["just some text", "un poco de texto"],
    ) as mock_chat:
        native, target = interpret_user_input("some input", "en", "es")

    assert native == "just some text"
    assert target == "un poco de texto"
    assert mock_chat.call_count == 2  # interpret call, then translate_text fallback


def test_interpret_user_input_wraps_model_server_error():
    with patch("app.translate.llm_translate.model_chat", side_effect=ModelServerUnavailableError("down")):
        with pytest.raises(TranslationUnavailableError, match="down"):
            interpret_user_input("hola", "en", "es")


def test_interpret_user_input_swaps_mislabeled_content():
    # Observed live: the model labeled its two lines correctly but put the
    # wrong language's content under each one - the "English" row came back
    # showing Spanish text (and vice versa).
    reply = "English: ¿Hablas bien español?\nSpanish: Do you speak Spanish well?"
    with patch("app.translate.llm_translate.model_chat", return_value=reply):
        native, target = interpret_user_input("Do you hablo the espanol good?", "en", "es")

    assert native == "Do you speak Spanish well?"
    assert target == "¿Hablas bien español?"


def test_interpret_user_input_forces_translation_when_target_line_never_translated():
    # Observed live: "sup bro. there once was a red monkey" came back with
    # the same English text under BOTH the "English" and "Spanish" labels -
    # the model just never translated it. This is different from the
    # mislabeled-swap case above: here target_text isn't Spanish, and
    # neither is native_text, so swapping them would just leave English
    # under both labels either way. Must force a real translation instead.
    reply = (
        "English: sup bro. there once was a red monkey\n"
        "Spanish: sup bro. there once was a red monkey"
    )
    with patch(
        "app.translate.llm_translate.model_chat",
        side_effect=[reply, "qué tal, bro. había una vez un mono rojo"],
    ) as mock_chat:
        native, target = interpret_user_input("sup bro. there once was a red monkey", "en", "es")

    assert native == "sup bro. there once was a red monkey"
    assert target == "qué tal, bro. había una vez un mono rojo"
    assert mock_chat.call_count == 2  # interpret call, then the forced translate_text call


def test_interpret_user_input_backfills_missing_native_line():
    # The model produced only the target-language line.
    with patch(
        "app.translate.llm_translate.model_chat",
        side_effect=["Spanish: ¿Hablas bien español?", "Do you speak Spanish well?"],
    ) as mock_chat:
        native, target = interpret_user_input("Do you hablo the espanol good?", "en", "es")

    assert native == "Do you speak Spanish well?"
    assert target == "¿Hablas bien español?"
    assert mock_chat.call_count == 2


def test_translate_text_retries_once_on_garbled_reply():
    # Qwen3 is heavily trained on Chinese data and can occasionally emit CJK
    # tokens instead of the requested language - observed live: a click
    # inserted Chinese characters into the input box.
    with patch(
        "app.translate.llm_translate.model_chat",
        side_effect=["你好世界", "Hello"],
    ) as mock_chat:
        assert translate_text("Hola", "es", "en") == "Hello"
    assert mock_chat.call_count == 2


def test_translate_text_raises_if_still_garbled_after_retry():
    with patch("app.translate.llm_translate.model_chat", return_value="你好世界"):
        with pytest.raises(TranslationUnavailableError, match="garbled"):
            translate_text("Hola", "es", "en")


def test_interpret_user_input_falls_back_on_hallucinated_reply():
    # Observed live: "hello sir" -> "Hello, sir. I am here to assist you."
    # The model answered the greeting instead of just correcting it, despite
    # the prompt's explicit instruction not to - falls back to a plain
    # translate_text pass, which has no conversational framing to overreach
    # with.
    reply = "English: Hello, sir. I am here to assist you.\nSpanish: Hola, señor. Estoy aquí para ayudarle."
    with patch(
        "app.translate.llm_translate.model_chat",
        side_effect=[reply, "Hola, señor."],
    ) as mock_chat:
        native, target = interpret_user_input("hello sir", "en", "es")

    assert native == "hello sir"
    assert target == "Hola, señor."
    assert mock_chat.call_count == 2  # interpret call, then the translate_text fallback


def test_interpret_user_input_prompt_warns_against_answering():
    with patch("app.translate.llm_translate.model_chat", return_value="English: hi\nSpanish: hola") as mock_chat:
        interpret_user_input("hi", "en", "es")

    (messages,), _ = mock_chat.call_args
    prompt = messages[0]["content"]
    assert "hola bro" in prompt.lower()
    assert "not having a" in prompt.lower() or "not have a" in prompt.lower()


def test_gloss_message_empty_input_short_circuits():
    with patch("app.translate.llm_translate.model_chat") as mock_chat:
        assert gloss_message("   ", "en", "es") == ("", [], [])
    mock_chat.assert_not_called()


def test_gloss_message_parses_words_grammar_and_phrases():
    reply = (
        '{"translation": "How wonderful, I ate.", "words": ['
        '{"surface": "Qué", "gloss": "what", "grammar": ""}, '
        '{"surface": "bueno", "gloss": "good", "grammar": ""}, '
        '{"surface": "comí", "gloss": "ate", "grammar": "comer (to eat): I, preterite"}], '
        '"phrases": [{"surface": "Qué bueno", "gloss": "how wonderful", "note": ""}]}'
    )
    with patch("app.translate.llm_translate.model_chat", return_value=reply) as mock_chat:
        translation, words, phrases = gloss_message("Qué bueno, comí.", "en", "es")

    assert translation == "How wonderful, I ate."
    assert words == [
        {"surface": "Qué", "gloss": "what", "grammar": ""},
        {"surface": "bueno", "gloss": "good", "grammar": ""},
        {"surface": "comí", "gloss": "ate", "grammar": "comer (to eat): I, preterite"},
    ]
    assert phrases == [{"surface": "Qué bueno", "gloss": "how wonderful", "note": ""}]
    (messages,), kwargs = mock_chat.call_args
    assert kwargs["max_tokens"] > 300
    assert messages[-1] == {"role": "user", "content": "Qué bueno, comí."}
    assert '"translation"' in messages[0]["content"]


def test_gloss_message_drops_single_word_phrases_and_missing_phrases_key():
    reply = '{"words": [{"surface": "hola", "gloss": "hello"}], "phrases": [{"surface": "hola", "gloss": "hi"}]}'
    with patch("app.translate.llm_translate.model_chat", return_value=reply):
        assert gloss_message("Hola", "en", "es") == ("", [{"surface": "hola", "gloss": "hello", "grammar": ""}], [])
    with patch("app.translate.llm_translate.model_chat", return_value='{"words": []}'):
        assert gloss_message("Hola", "en", "es") == ("", [], [])


def test_gloss_message_for_the_learner_asks_for_no_translation():
    reply = '{"words": [{"surface": "swimming", "gloss": "nadar"}], "phrases": []}'
    with patch("app.translate.llm_translate.model_chat", return_value=reply) as mock_chat:
        _translation, words, _phrases = gloss_message("me gusta swimming", "en", "es", learner=True)

    assert words[0]["gloss"] == "nadar"
    prompt = mock_chat.call_args.args[0][0]["content"]
    assert "learner wrote" in prompt
    assert '"translation"' not in prompt


def test_gloss_message_strips_surrounding_prose_and_code_fences():
    reply = 'Sure:\n```json\n{"translation": "Hello", "words": [{"surface": "hola", "gloss": "hello"}], "phrases": []}\n```'
    with patch("app.translate.llm_translate.model_chat", return_value=reply):
        translation, words, _ = gloss_message("Hola", "en", "es")
    assert translation == "Hello"
    assert words == [{"surface": "hola", "gloss": "hello", "grammar": ""}]


def test_gloss_message_retries_on_unparseable_reply():
    with patch(
        "app.translate.llm_translate.model_chat",
        side_effect=["not json at all", '{"translation": "Hi", "words": [{"surface": "hola", "gloss": "hi"}]}'],
    ) as mock_chat:
        translation, words, _ = gloss_message("Hola", "en", "es")
    assert translation == "Hi"
    assert words == [{"surface": "hola", "gloss": "hi", "grammar": ""}]
    assert mock_chat.call_count == 2


def test_gloss_message_raises_after_exhausting_retries():
    with patch("app.translate.llm_translate.model_chat", return_value="not json at all"):
        with pytest.raises(TranslationUnavailableError):
            gloss_message("Hola", "en", "es")


def test_gloss_message_raises_on_garbled_reply():
    with patch("app.translate.llm_translate.model_chat", return_value="你好世界"):
        with pytest.raises(TranslationUnavailableError):
            gloss_message("Hola", "en", "es")


def test_gloss_message_wraps_model_server_error():
    with patch("app.translate.llm_translate.model_chat", side_effect=ModelServerUnavailableError("down")):
        with pytest.raises(TranslationUnavailableError, match="down"):
            gloss_message("Hola", "en", "es")


def test_tag_draft_empty_input_short_circuits():
    with patch("app.translate.llm_translate.model_chat") as mock_chat:
        assert tag_draft("   ", "en", "es") == ("", [])
    mock_chat.assert_not_called()


def test_tag_draft_parses_translation_and_ordered_spans():
    reply = (
        '{"translation": "Quiero ir a la playa.", "spans": ['
        '{"surface": "quiero", "gloss": "I want", "note": "", "translation": ""}, '
        '{"surface": "ir to", "gloss": "to go to", "note": "mixed English/Spanish", "translation": "ir a"}'
        "]}"
    )
    with patch("app.translate.llm_translate.model_chat", return_value=reply) as mock_chat:
        translation, spans = tag_draft("quiero ir to the beach", "en", "es")

    assert translation == "Quiero ir a la playa."
    assert spans == [
        {"surface": "quiero", "gloss": "I want", "note": "", "translation": "", "alternate_gloss": "", "literal": ""},
        {
            "surface": "ir to",
            "gloss": "to go to",
            "note": "mixed English/Spanish",
            "translation": "ir a",
            "alternate_gloss": "",
            "literal": "",
        },
    ]
    (messages,), kwargs = mock_chat.call_args
    assert messages[-1] == {"role": "user", "content": "quiero ir to the beach"}


def test_tag_draft_returns_duplicate_surface_spans_as_an_ordered_list():
    # Unlike gloss_reply's word_map (a dict), repeated surface text in a
    # draft must produce two separate span entries - a dict would silently
    # collapse them, and each occurrence needs its own later position-match.
    reply = (
        '{"translation": "the cat and the cat", "spans": ['
        '{"surface": "cat", "gloss": "a", "note": "", "translation": ""}, '
        '{"surface": "cat", "gloss": "b", "note": "", "translation": ""}'
        "]}"
    )
    with patch("app.translate.llm_translate.model_chat", return_value=reply):
        _translation, spans = tag_draft("the cat and the cat", "en", "es")

    assert len(spans) == 2
    assert [s["gloss"] for s in spans] == ["a", "b"]


def test_tag_draft_parses_alternate_gloss():
    reply = (
        '{"translation": "once veces", "spans": ['
        '{"surface": "once", "gloss": "eleven", "note": "", "translation": "", '
        '"alternate_gloss": "once (as in \\"at once\\") - an English word too"}'
        "]}"
    )
    with patch("app.translate.llm_translate.model_chat", return_value=reply):
        _translation, spans = tag_draft("once veces", "en", "es")

    assert spans[0]["alternate_gloss"] == 'once (as in "at once") - an English word too'


def test_tag_draft_parses_literal_breakdown_for_multi_word_span():
    reply = (
        '{"spans": [{"surface": "tener en cuenta", "gloss": "take into account", "note": "", '
        '"translation": "", "literal": "tener (to have) + en (in) + cuenta (account)"}]}'
    )
    with patch("app.translate.llm_translate.model_chat", return_value=reply):
        _translation, spans = tag_draft("tener en cuenta algo", "en", "es")

    assert spans[0]["literal"] == "tener (to have) + en (in) + cuenta (account)"


def test_tag_draft_literal_defaults_to_empty():
    reply = '{"spans": [{"surface": "hola", "gloss": "hi", "note": "", "translation": ""}]}'
    with patch("app.translate.llm_translate.model_chat", return_value=reply):
        _translation, spans = tag_draft("hola", "en", "es")

    assert spans[0]["literal"] == ""


def test_tag_draft_alternate_gloss_defaults_to_empty():
    reply = '{"translation": "hola", "spans": [{"surface": "hola", "gloss": "hi", "note": "", "translation": ""}]}'
    with patch("app.translate.llm_translate.model_chat", return_value=reply):
        _translation, spans = tag_draft("hola", "en", "es")

    assert spans[0]["alternate_gloss"] == ""


def test_tag_draft_drops_spans_with_no_surface_text():
    reply = '{"translation": "hola", "spans": [{"surface": "", "gloss": "x", "note": "", "translation": ""}]}'
    with patch("app.translate.llm_translate.model_chat", return_value=reply):
        _translation, spans = tag_draft("hola", "en", "es")
    assert spans == []


def test_tag_draft_retries_once_on_unparseable_reply():
    with patch(
        "app.translate.llm_translate.model_chat",
        side_effect=["not json at all", '{"translation": "Hi", "spans": [{"surface": "hola", "gloss": "hi"}]}'],
    ) as mock_chat:
        translation, spans = tag_draft("hola", "en", "es")

    assert translation == "Hi"
    assert spans == [
        {"surface": "hola", "gloss": "hi", "note": "", "translation": "", "alternate_gloss": "", "literal": ""}
    ]
    assert mock_chat.call_count == 2


def test_tag_draft_raises_after_exhausting_retries():
    with patch("app.translate.llm_translate.model_chat", return_value="not json at all") as mock_chat:
        with pytest.raises(TranslationUnavailableError):
            tag_draft("hola", "en", "es")
    assert mock_chat.call_count == 3


def test_tag_draft_retries_on_empty_reply():
    # Observed live: the model occasionally returns a fully empty
    # completion for a longer/more complex draft - not "garbled" (no
    # unexpected script), just blank, so this needs its own check rather
    # than relying on _looks_garbled to catch it.
    with patch(
        "app.translate.llm_translate.model_chat",
        side_effect=["", '{"translation": "Hi", "spans": [{"surface": "hola", "gloss": "hi"}]}'],
    ) as mock_chat:
        translation, spans = tag_draft("hola", "en", "es")

    assert translation == "Hi"
    assert mock_chat.call_count == 2


def test_tag_draft_raises_after_repeated_empty_replies():
    with patch("app.translate.llm_translate.model_chat", return_value="") as mock_chat:
        with pytest.raises(TranslationUnavailableError):
            tag_draft("hola", "en", "es")
    assert mock_chat.call_count == 3


def test_tag_draft_raises_on_garbled_reply():
    with patch("app.translate.llm_translate.model_chat", return_value="你好世界"):
        with pytest.raises(TranslationUnavailableError):
            tag_draft("hola", "en", "es")


def test_tag_draft_wraps_model_server_error():
    with patch("app.translate.llm_translate.model_chat", side_effect=ModelServerUnavailableError("down")):
        with pytest.raises(TranslationUnavailableError, match="down"):
            tag_draft("hola", "en", "es")


def test_interpret_user_input_falls_back_to_translate_text_on_garbled_reply():
    # Both interpret attempts come back garbled - falls back to a plain
    # translate_text call on the raw input rather than parsing garbage.
    # The raw input reads as majority-English, so native_text is used as-is
    # (no further translate_text call needed for it) and only the target
    # (Spanish) line needs a real translation call.
    with patch(
        "app.translate.llm_translate.model_chat",
        side_effect=["你好世界", "你好世界", "¿Hablas bien español?"],
    ) as mock_chat:
        native, target = interpret_user_input("Do you hablo the espanol good?", "en", "es")

    assert native == "Do you hablo the espanol good?"
    assert target == "¿Hablas bien español?"
    assert mock_chat.call_count == 3


# --- coach_reply_stream ---------------------------------------------------------

_MESSAGES = [
    {"number": 1, "role": "user", "text": "Hola"},
    {"number": 2, "role": "assistant", "text": "¿Cómo te llamas?", "glosses": "llamas = you call yourself"},
]


def _reply(chunks, **kwargs):
    args = dict(
        text=kwargs.pop("text", "x"),
        messages=kwargs.pop("messages", _MESSAGES),
        about_number=kwargs.pop("about_number", None),
        thread_turns=kwargs.pop("thread_turns", []),
        draft_turns=kwargs.pop("draft_turns", []),
        native_lang="en",
        target_lang="es",
        **kwargs,
    )
    with patch("app.translate.llm_translate.model_chat_stream", return_value=iter(chunks)) as mock_stream:
        events = list(coach_reply_stream(**args))
    return events, mock_stream


def test_coach_reply_stream_empty_input_short_circuits():
    with patch("app.translate.llm_translate.model_chat_stream") as mock_stream:
        assert list(coach_reply_stream("  ", _MESSAGES, None, [], [], "en", "es")) == []
    mock_stream.assert_not_called()


def test_coach_reply_stream_yields_ready_then_the_streamed_reply():
    events, mock_stream = _reply(
        ["no<<<REA", "DY>>>Nice. Try ", "[[Me gusta nadar. | I like swimming.]]"],
        text="me gusta swimming",
    )
    assert events[0] == ("ready", False)
    reply = "".join(e[1] for e in events if e[0] == "reply_chunk")
    assert reply == "Nice. Try [[Me gusta nadar. | I like swimming.]]"
    assert events[-1] == ("reply_complete", reply)
    system = mock_stream.call_args.args[0][0]["content"]
    assert "[2] Partner: ¿Cómo te llamas?" in system
    assert "llamas = you call yourself" in system
    assert "[[Spanish sentence | English meaning]]" in system
    assert "about what they'll say next" in system
    assert "ABOUT" not in system
    assert mock_stream.call_args.kwargs["model"]


def test_coach_reply_stream_ready_yes():
    events, _ = _reply(["Yes.<<<READY>>>That's right!"])
    assert events[0] == ("ready", True)
    assert events[-1] == ("reply_complete", "That's right!")


def test_coach_reply_stream_strips_echoed_field_labels():
    events, _ = _reply(["READY: yes<<<READY>>>REPLY: ", "Perfect, send it."])
    assert events[0] == ("ready", True)
    assert events[-1] == ("reply_complete", "Perfect, send it.")
    assert all(not e[1].startswith("REPLY") for e in events if e[0] == "reply_chunk")


def test_coach_reply_stream_without_the_marker_still_returns_the_reply():
    events, _ = _reply(["Just a plain answer."])
    assert events[0] == ("ready", False)
    assert events[-1] == ("reply_complete", "Just a plain answer.")


def test_coach_reply_stream_about_a_message_puts_that_message_and_its_thread_in_the_prompt():
    _, mock_stream = _reply(
        ["no<<<READY>>>ok"],
        about_number=2,
        thread_turns=[("learner", "what does it mean?"), ("coach", "What's your name?")],
    )
    system = mock_stream.call_args.args[0][0]["content"]
    assert "about message [2] (their partner's message)" in system
    assert "Coach: What's your name?" in system


def test_coach_reply_stream_next_message_context_and_check_reach_the_prompt():
    _, mock_stream = _reply(
        ["no<<<READY>>>ok"],
        draft_turns=[("learner", "I want to say I'm Kyle"), ("coach", "Try [[Me llamo Kyle. | My name is Kyle.]]")],
        check=True,
    )
    system = mock_stream.call_args.args[0][0]["content"]
    assert "Learner: I want to say I'm Kyle" in system
    assert "exact message they're about to send" in system


def test_coach_reply_stream_raises_on_empty_or_unreachable_model():
    with pytest.raises(TranslationUnavailableError):
        _reply(["no<<<READY>>>"])
    with patch("app.translate.llm_translate.model_chat_stream", side_effect=ModelServerUnavailableError("down")):
        with pytest.raises(TranslationUnavailableError):
            list(coach_reply_stream("q", _MESSAGES, None, [], [], "en", "es"))
