from unittest.mock import patch

import pytest

from app.chat.openai_client import ModelServerUnavailableError
from app.translate.llm_translate import (
    TranslationUnavailableError,
    coach_draft_stream,
    gloss_reply,
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


def test_gloss_reply_empty_input_short_circuits():
    with patch("app.translate.llm_translate.model_chat") as mock_chat:
        assert gloss_reply("   ", "es", "en") == ("", [])
    mock_chat.assert_not_called()


def test_gloss_reply_parses_translation_and_ordered_spans():
    reply = (
        '{"translation": "The bank is closed.", "spans": ['
        '{"surface": "El", "gloss": "the", "note": ""}, '
        '{"surface": "banco", "gloss": "bank", "note": "financial institution, not a bench"}, '
        '{"surface": "está", "gloss": "is", "note": ""}, '
        '{"surface": "cerrado", "gloss": "closed", "note": ""}]}'
    )
    with patch("app.translate.llm_translate.model_chat", return_value=reply) as mock_chat:
        translation, spans = gloss_reply("El banco está cerrado.", "es", "en")

    assert translation == "The bank is closed."
    assert spans == [
        {"surface": "El", "gloss": "the", "note": "", "translation": "", "alternate_gloss": ""},
        {
            "surface": "banco",
            "gloss": "bank",
            "note": "financial institution, not a bench",
            "translation": "",
            "alternate_gloss": "",
        },
        {"surface": "está", "gloss": "is", "note": "", "translation": "", "alternate_gloss": ""},
        {"surface": "cerrado", "gloss": "closed", "note": "", "translation": "", "alternate_gloss": ""},
    ]
    (messages,), kwargs = mock_chat.call_args
    assert kwargs["max_tokens"] > 300  # bigger budget than a plain translation
    assert messages[-1] == {"role": "user", "content": "El banco está cerrado."}


def test_gloss_reply_groups_a_multi_word_span():
    # "el tuyo" ("yours") must come back as ONE span covering both words,
    # not "el" and "tuyo" glossed (or left ungloss) separately.
    reply = '{"translation": "yours", "spans": [{"surface": "el tuyo", "gloss": "yours", "note": ""}]}'
    with patch("app.translate.llm_translate.model_chat", return_value=reply):
        _translation, spans = gloss_reply("el tuyo", "es", "en")

    assert spans == [{"surface": "el tuyo", "gloss": "yours", "note": "", "translation": "", "alternate_gloss": ""}]


def test_gloss_reply_strips_surrounding_prose_and_code_fences():
    reply = (
        'Sure, here you go:\n```json\n{"translation": "Hello", '
        '"spans": [{"surface": "hola", "gloss": "hello", "note": ""}]}\n```'
    )
    with patch("app.translate.llm_translate.model_chat", return_value=reply):
        translation, spans = gloss_reply("Hola", "es", "en")

    assert translation == "Hello"
    assert spans == [{"surface": "hola", "gloss": "hello", "note": "", "translation": "", "alternate_gloss": ""}]


def test_gloss_reply_retries_once_on_unparseable_reply():
    with patch(
        "app.translate.llm_translate.model_chat",
        side_effect=["not json at all", '{"translation": "Hi", "spans": [{"surface": "hola", "gloss": "hi"}]}'],
    ) as mock_chat:
        translation, spans = gloss_reply("Hola", "es", "en")

    assert translation == "Hi"
    assert spans == [{"surface": "hola", "gloss": "hi", "note": "", "translation": "", "alternate_gloss": ""}]
    assert mock_chat.call_count == 2


def test_gloss_reply_raises_after_exhausting_retries():
    with patch("app.translate.llm_translate.model_chat", return_value="not json at all"):
        with pytest.raises(TranslationUnavailableError):
            gloss_reply("Hola", "es", "en")


def test_gloss_reply_raises_on_garbled_reply():
    with patch("app.translate.llm_translate.model_chat", return_value="你好世界"):
        with pytest.raises(TranslationUnavailableError):
            gloss_reply("Hola", "es", "en")


def test_gloss_reply_wraps_model_server_error():
    with patch("app.translate.llm_translate.model_chat", side_effect=ModelServerUnavailableError("down")):
        with pytest.raises(TranslationUnavailableError, match="down"):
            gloss_reply("Hola", "es", "en")


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
        {"surface": "quiero", "gloss": "I want", "note": "", "translation": "", "alternate_gloss": ""},
        {
            "surface": "ir to",
            "gloss": "to go to",
            "note": "mixed English/Spanish",
            "translation": "ir a",
            "alternate_gloss": "",
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
    assert spans == [{"surface": "hola", "gloss": "hi", "note": "", "translation": "", "alternate_gloss": ""}]
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


def test_coach_draft_stream_empty_input_short_circuits():
    with patch("app.translate.llm_translate.model_chat_stream") as mock_stream:
        events = list(coach_draft_stream("   ", [], "en", "es"))
    assert events == []
    mock_stream.assert_not_called()


def test_coach_draft_stream_yields_core_then_translations():
    core_json = (
        '{"meaning": "I want to go to the beach tomorrow.", '
        '"feedback": "Good attempt - just a word order slip.", '
        '"options": ['
        '{"formality": "neutral", "spanish": "Quiero ir a la playa mañana."}, '
        '{"formality": "casual", "spanish": "Quiero ir a la playa mañana, ¿sí?"}'
        "]}"
    )
    translations_json = (
        '{"options": ['
        '{"english": "I want to go to the beach tomorrow.", '
        '"spans": [{"surface": "Quiero", "gloss": "I want", "note": "", "translation": ""}]}, '
        '{"english": "I want to go to the beach tomorrow, yeah?", "spans": []}'
        "]}"
    )
    chunks = [core_json, "<<<TRANSLATIONS>>>", translations_json]
    with patch("app.translate.llm_translate.model_chat_stream", return_value=iter(chunks)) as mock_stream:
        events = list(coach_draft_stream("quiero playa mañana ir", [], "en", "es"))

    assert [kind for kind, _ in events] == ["core", "translations"]
    core = events[0][1]
    assert core["meaning"] == "I want to go to the beach tomorrow."
    assert core["feedback"] == "Good attempt - just a word order slip."
    assert core["options"] == [
        ("neutral", "Quiero ir a la playa mañana."),
        ("casual", "Quiero ir a la playa mañana, ¿sí?"),
    ]
    translations = events[1][1]
    assert translations[0]["english"] == "I want to go to the beach tomorrow."
    assert translations[0]["spans"] == [
        {"surface": "Quiero", "gloss": "I want", "note": "", "translation": "", "alternate_gloss": ""}
    ]
    assert translations[1] == {"english": "I want to go to the beach tomorrow, yeah?", "spans": []}

    (messages,), kwargs = mock_stream.call_args
    assert messages[-1] == {"role": "user", "content": "quiero playa mañana ir"}


def test_coach_draft_stream_marker_split_across_chunks():
    # The marker can land anywhere relative to chunk boundaries - the
    # buffer accumulates across chunks regardless, so this must behave
    # identically to the marker arriving in one whole chunk.
    core_json = '{"meaning": "x", "feedback": "", "options": [{"formality": "neutral", "spanish": "y"}]}'
    marker = "<<<TRANSLATIONS>>>"
    translations_json = '{"options": [{"english": "z", "spans": []}]}'
    full = core_json + marker + translations_json
    chunks = [full[i : i + 7] for i in range(0, len(full), 7)]  # arbitrary small chunks
    with patch("app.translate.llm_translate.model_chat_stream", return_value=iter(chunks)):
        events = list(coach_draft_stream("algo", [], "en", "es"))

    assert [kind for kind, _ in events] == ["core", "translations"]
    assert events[0][1]["options"] == [("neutral", "y")]
    assert events[1][1] == [{"english": "z", "spans": []}]


def test_coach_draft_stream_core_only_when_marker_never_arrives():
    # Token budget pressure (or the model just not getting that far) can
    # leave PART 2 - and the marker - entirely unwritten. The whole buffer
    # should still work as PART 1 on its own.
    core_json = '{"meaning": "x", "feedback": "", "options": [{"formality": "neutral", "spanish": "y"}]}'
    with patch("app.translate.llm_translate.model_chat_stream", return_value=iter([core_json])):
        events = list(coach_draft_stream("algo", [], "en", "es"))

    assert [kind for kind, _ in events] == ["core"]
    assert events[0][1]["options"] == [("neutral", "y")]


def test_coach_draft_stream_defaults_unknown_formality_to_neutral():
    core_json = '{"meaning": "x", "feedback": "", "options": [{"formality": "sarcastic", "spanish": "y"}]}'
    with patch("app.translate.llm_translate.model_chat_stream", return_value=iter([core_json])):
        events = list(coach_draft_stream("algo", [], "en", "es"))

    assert events[0][1]["options"] == [("neutral", "y")]


def test_coach_draft_stream_translations_missing_is_not_fatal():
    # PART 2 garbled/missing shouldn't take PART 1 down with it - the
    # caller already has (and may have acted on) the core event.
    core_json = '{"meaning": "x", "feedback": "", "options": [{"formality": "neutral", "spanish": "y"}]}'
    chunks = [core_json, "<<<TRANSLATIONS>>>", "not json at all"]
    with patch("app.translate.llm_translate.model_chat_stream", return_value=iter(chunks)):
        events = list(coach_draft_stream("algo", [], "en", "es"))

    assert [kind for kind, _ in events] == ["core"]


def test_coach_draft_stream_raises_when_core_itself_never_parses():
    with patch("app.translate.llm_translate.model_chat_stream", return_value=iter(["not json at all"])):
        with pytest.raises(TranslationUnavailableError):
            list(coach_draft_stream("hola", [], "en", "es"))


def test_coach_draft_stream_includes_recent_history_in_prompt():
    core_json = '{"meaning": "x", "feedback": "", "options": [{"formality": "neutral", "spanish": "y"}]}'
    history = [("assistant", "¿Qué planes tienes para el fin de semana?"), ("user", "quiero ir playa")]
    with patch("app.translate.llm_translate.model_chat_stream", return_value=iter([core_json])) as mock_stream:
        list(coach_draft_stream("mañana quiero ir", history, "en", "es"))

    (messages,), _ = mock_stream.call_args
    prompt = messages[0]["content"]
    assert "¿Qué planes tienes para el fin de semana?" in prompt
    assert "quiero ir playa" in prompt


def test_coach_draft_stream_wraps_model_server_error():
    with patch("app.translate.llm_translate.model_chat_stream", side_effect=ModelServerUnavailableError("down")):
        with pytest.raises(TranslationUnavailableError, match="down"):
            list(coach_draft_stream("hola", [], "en", "es"))
