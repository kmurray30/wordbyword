from unittest.mock import patch

import pytest

from app.chat.openai_client import ModelServerUnavailableError
from app.config import OPENAI_TRANSLATE_MODEL
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
        {"surface": "El", "gloss": "the", "note": "", "translation": "", "alternate_gloss": "", "literal": ""},
        {
            "surface": "banco",
            "gloss": "bank",
            "note": "financial institution, not a bench",
            "translation": "",
            "alternate_gloss": "",
            "literal": "",
        },
        {"surface": "está", "gloss": "is", "note": "", "translation": "", "alternate_gloss": "", "literal": ""},
        {"surface": "cerrado", "gloss": "closed", "note": "", "translation": "", "alternate_gloss": "", "literal": ""},
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

    assert spans == [
        {"surface": "el tuyo", "gloss": "yours", "note": "", "translation": "", "alternate_gloss": "", "literal": ""}
    ]


def test_gloss_reply_strips_surrounding_prose_and_code_fences():
    reply = (
        'Sure, here you go:\n```json\n{"translation": "Hello", '
        '"spans": [{"surface": "hola", "gloss": "hello", "note": ""}]}\n```'
    )
    with patch("app.translate.llm_translate.model_chat", return_value=reply):
        translation, spans = gloss_reply("Hola", "es", "en")

    assert translation == "Hello"
    assert spans == [
        {"surface": "hola", "gloss": "hello", "note": "", "translation": "", "alternate_gloss": "", "literal": ""}
    ]


def test_gloss_reply_retries_once_on_unparseable_reply():
    with patch(
        "app.translate.llm_translate.model_chat",
        side_effect=["not json at all", '{"translation": "Hi", "spans": [{"surface": "hola", "gloss": "hi"}]}'],
    ) as mock_chat:
        translation, spans = gloss_reply("Hola", "es", "en")

    assert translation == "Hi"
    assert spans == [
        {"surface": "hola", "gloss": "hi", "note": "", "translation": "", "alternate_gloss": "", "literal": ""}
    ]
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


def test_coach_draft_stream_empty_input_short_circuits():
    with patch("app.translate.llm_translate.model_chat_stream") as mock_stream:
        events = list(coach_draft_stream("   ", [], "en", "es"))
    assert events == []
    mock_stream.assert_not_called()


def test_coach_draft_stream_clean_verdict_yields_verdict_then_meaning_only():
    chunks = ["CLEAN<<<VERDICT>>>", "Hello there<<<MEANING>>>"]
    with patch("app.translate.llm_translate.model_chat_stream", return_value=iter(chunks)) as mock_stream:
        events = list(coach_draft_stream("hola", [], "en", "es"))

    kinds = [e[0] for e in events]
    assert kinds[0] == "verdict"
    assert events[0] == ("verdict", "clean")
    assert kinds[-1] == "meaning_complete"
    assert events[-1] == ("meaning_complete", "Hello there")
    assert "".join(e[1] for e in events if e[0] == "meaning_chunk") == "Hello there"
    # Nothing else - no suggestion, no options - for a clean verdict.
    assert all(k in ("verdict", "meaning_chunk", "meaning_complete") for k in kinds)

    (messages,), _ = mock_stream.call_args
    assert messages[-1] == {"role": "user", "content": "hola"}


def test_coach_draft_stream_minor_verdict_streams_a_single_suggestion():
    chunks = ["MINOR<<<VERDICT>>>", "You mean hello<<<MEANING>>>", "Hola, ¿cómo estás?<<<SUGGESTION>>>"]
    with patch("app.translate.llm_translate.model_chat_stream", return_value=iter(chunks)):
        events = list(coach_draft_stream("hola como estas", [], "en", "es"))

    assert events[0] == ("verdict", "minor")
    assert "".join(e[1] for e in events if e[0] == "meaning_chunk") == "You mean hello"
    assert ("meaning_complete", "You mean hello") in events
    assert "".join(e[1] for e in events if e[0] == "suggestion_chunk") == "Hola, ¿cómo estás?"
    assert events[-1] == ("suggestion_complete", "Hola, ¿cómo estás?")


def test_coach_draft_stream_fix_verdict_streams_three_interleaved_options():
    chunks = [
        "FIX<<<VERDICT>>>",
        "I want to go to the beach<<<MEANING>>>",
        "Quiero ir a la playa.<<<OPT1_ES>>>",
        "I want to go to the beach.<<<OPT1_EN>>>",
        "Quiero ir a la playa, ¿va?<<<OPT2_ES>>>",
        "I want to go to the beach, right?<<<OPT2_EN>>>",
        "Desearía ir a la playa.<<<OPT3_ES>>>",
        "I would like to go to the beach.<<<OPT3_EN>>>",
    ]
    with patch("app.translate.llm_translate.model_chat_stream", return_value=iter(chunks)):
        events = list(coach_draft_stream("i want go beach", [], "en", "es"))

    complete_events = [e for e in events if e[0].endswith("_complete")]
    # Interleaved per-option order: each option's own Spanish completes
    # immediately before ITS OWN English - not all three Spanish texts
    # followed by all three English translations batched at the end.
    assert [e[0] for e in complete_events] == [
        "meaning_complete",
        "option_spanish_complete",
        "option_english_complete",
        "option_spanish_complete",
        "option_english_complete",
        "option_spanish_complete",
        "option_english_complete",
    ]
    spanish_by_index = {e[1]: e[2] for e in complete_events if e[0] == "option_spanish_complete"}
    english_by_index = {e[1]: e[2] for e in complete_events if e[0] == "option_english_complete"}
    assert spanish_by_index == {
        0: "Quiero ir a la playa.",
        1: "Quiero ir a la playa, ¿va?",
        2: "Desearía ir a la playa.",
    }
    assert english_by_index == {
        0: "I want to go to the beach.",
        1: "I want to go to the beach, right?",
        2: "I would like to go to the beach.",
    }


def test_coach_draft_stream_fix_option_swaps_mislabeled_spanish_and_english():
    # Observed live equivalent of interpret_user_input's own swap bug (see
    # test_interpret_user_input_swaps_mislabeled_content) but for a FIX
    # verdict's per-option fields - the model put the right content under
    # the wrong marker.
    chunks = [
        "FIX<<<VERDICT>>>",
        "meaning<<<MEANING>>>",
        "I want to go to the beach.<<<OPT1_ES>>>",
        "Quiero ir a la playa.<<<OPT1_EN>>>",
    ]
    with patch("app.translate.llm_translate.model_chat_stream", return_value=iter(chunks)):
        events = list(coach_draft_stream("algo", [], "en", "es"))

    spanish_complete = [e[2] for e in events if e[0] == "option_spanish_complete" and e[1] == 0]
    english_complete = [e[2] for e in events if e[0] == "option_english_complete" and e[1] == 0]
    # The original (wrong-language) spanish_complete is yielded immediately
    # as it arrives, then a corrective SECOND one follows once the matching
    # english_complete reveals the swap - a repeat "_complete" for the same
    # index/field is exactly what the frontend reducer already overwrites
    # on, so the corrected one being last is what the UI ends up showing.
    assert spanish_complete[-1] == "Quiero ir a la playa."
    assert english_complete[-1] == "I want to go to the beach."


def test_coach_draft_stream_fix_option_forces_translation_when_spanish_field_was_never_translated():
    # The model left BOTH fields in English - not a swap (Spanish never
    # appeared at all), so the Spanish field must be force-translated
    # rather than swapped with the (also-English) "English" field.
    chunks = [
        "FIX<<<VERDICT>>>",
        "meaning<<<MEANING>>>",
        "I want to go to the beach.<<<OPT1_ES>>>",
        "I want to go to the beach.<<<OPT1_EN>>>",
    ]
    with (
        patch("app.translate.llm_translate.model_chat_stream", return_value=iter(chunks)),
        patch("app.translate.llm_translate.model_chat", return_value="Quiero ir a la playa.") as mock_chat,
    ):
        events = list(coach_draft_stream("algo", [], "en", "es"))

    spanish_complete = [e[2] for e in events if e[0] == "option_spanish_complete" and e[1] == 0]
    english_complete = [e[2] for e in events if e[0] == "option_english_complete" and e[1] == 0]
    assert spanish_complete[-1] == "Quiero ir a la playa."
    assert english_complete[-1] == "I want to go to the beach."
    mock_chat.assert_called_once()


def test_coach_draft_stream_fix_option_forces_translation_when_english_field_was_never_translated():
    # The mirror case: both fields left in Spanish - the English back-
    # translation must be force-translated rather than left showing
    # Spanish text under an "English" label.
    chunks = [
        "FIX<<<VERDICT>>>",
        "meaning<<<MEANING>>>",
        "Quiero ir a la playa.<<<OPT1_ES>>>",
        "Quiero ir a la playa.<<<OPT1_EN>>>",
    ]
    with (
        patch("app.translate.llm_translate.model_chat_stream", return_value=iter(chunks)),
        patch("app.translate.llm_translate.model_chat", return_value="I want to go to the beach.") as mock_chat,
    ):
        events = list(coach_draft_stream("algo", [], "en", "es"))

    spanish_complete = [e[2] for e in events if e[0] == "option_spanish_complete" and e[1] == 0]
    english_complete = [e[2] for e in events if e[0] == "option_english_complete" and e[1] == 0]
    assert spanish_complete[-1] == "Quiero ir a la playa."
    assert english_complete[-1] == "I want to go to the beach."
    mock_chat.assert_called_once()


def test_coach_draft_stream_fix_option_correction_degrades_gracefully_on_translate_failure():
    # The correction check's own fallback translate_text call can itself
    # fail - that must not crash the whole stream, just leave the
    # (possibly still wrong-language) pair as the model originally wrote
    # it rather than losing the option entirely.
    chunks = [
        "FIX<<<VERDICT>>>",
        "meaning<<<MEANING>>>",
        "I want to go to the beach.<<<OPT1_ES>>>",
        "I want to go to the beach.<<<OPT1_EN>>>",
    ]
    with (
        patch("app.translate.llm_translate.model_chat_stream", return_value=iter(chunks)),
        patch("app.translate.llm_translate.model_chat", side_effect=ModelServerUnavailableError("down")),
    ):
        events = list(coach_draft_stream("algo", [], "en", "es"))

    spanish_complete = [e[2] for e in events if e[0] == "option_spanish_complete" and e[1] == 0]
    english_complete = [e[2] for e in events if e[0] == "option_english_complete" and e[1] == 0]
    assert spanish_complete == ["I want to go to the beach."]
    assert english_complete == ["I want to go to the beach."]


def test_coach_draft_stream_markers_split_across_chunks():
    # A marker can land anywhere relative to chunk boundaries - the buffer
    # accumulates across chunks regardless, so this must behave
    # identically to each marker arriving in one whole chunk.
    full = (
        "MINOR<<<VERDICT>>>"
        "meaning text<<<MEANING>>>"
        "suggestion text<<<SUGGESTION>>>"
    )
    chunks = [full[i : i + 5] for i in range(0, len(full), 5)]  # arbitrary small chunks
    with patch("app.translate.llm_translate.model_chat_stream", return_value=iter(chunks)):
        events = list(coach_draft_stream("algo", [], "en", "es"))

    assert events[0] == ("verdict", "minor")
    assert "".join(e[1] for e in events if e[0] == "meaning_chunk") == "meaning text"
    assert "".join(e[1] for e in events if e[0] == "suggestion_chunk") == "suggestion text"
    assert events[-1] == ("suggestion_complete", "suggestion text")


def test_coach_draft_stream_raises_when_verdict_marker_never_arrives():
    # Nothing downstream can be trusted without knowing which fields to
    # expect next - this is the one case that's a real error, not a
    # graceful degradation.
    with patch("app.translate.llm_translate.model_chat_stream", return_value=iter(["just some plain text, no marker"])):
        with pytest.raises(TranslationUnavailableError):
            list(coach_draft_stream("algo", [], "en", "es"))


def test_coach_draft_stream_never_leaks_a_marker_fragment_into_meaning_chunks():
    # A chunk boundary landing mid-marker must never leak a piece of the
    # marker itself into a "meaning_chunk" delta.
    # Split right after "<<<MEAN", mid-marker - the worst case for a naive
    # implementation that streams everything not yet proven to be the
    # marker.
    chunks = ["CLEAN<<<VERDICT>>>meaning text<<<MEAN", "ING>>>"]
    with patch("app.translate.llm_translate.model_chat_stream", return_value=iter(chunks)):
        events = list(coach_draft_stream("algo", [], "en", "es"))

    meaning = "".join(e[1] for e in events if e[0] == "meaning_chunk")
    assert meaning == "meaning text"
    assert "<" not in meaning


def test_coach_draft_stream_defaults_unrecognized_verdict_token_to_fix():
    # A malformed/unrecognized verdict must never silently become "clean"
    # (telling the learner a broken draft is fine to send) - defaulting to
    # the full correction flow is the safe direction to fail in.
    chunks = ["MAYBE<<<VERDICT>>>", "x<<<MEANING>>>", "y<<<OPT1_ES>>>", "z<<<OPT1_EN>>>", "a<<<OPT2_ES>>>", "b<<<OPT2_EN>>>", "c<<<OPT3_ES>>>", "d<<<OPT3_EN>>>"]
    # The single-letter option placeholders above have no real language
    # content for the Spanish/English correction check to key off (see
    # _correct_option_languages's own tests below) - model_chat is mocked
    # here purely so that check's fallback translate_text call, if it
    # fires, doesn't reach out over the network; this test only cares
    # about the verdict default.
    with (
        patch("app.translate.llm_translate.model_chat_stream", return_value=iter(chunks)),
        patch("app.translate.llm_translate.model_chat", return_value="n/a"),
    ):
        events = list(coach_draft_stream("algo", [], "en", "es"))

    assert events[0] == ("verdict", "fix")


def test_coach_draft_stream_degrades_gracefully_when_a_later_option_never_arrives():
    # Token budget pressure (or the model just not getting that far) can
    # leave the LAST option(s) unwritten - the earlier ones should still
    # be fully usable rather than the whole thing failing.
    chunks = [
        "FIX<<<VERDICT>>>",
        "meaning<<<MEANING>>>",
        "Quiero ir a la playa.<<<OPT1_ES>>>",
        "I want to go to the beach.<<<OPT1_EN>>>",
        "opt2 spanish, no marker yet",
        # opt2's own marker never arrives, nor does its english, nor opt3 -
        # the model just stopped generating.
    ]
    with patch("app.translate.llm_translate.model_chat_stream", return_value=iter(chunks)):
        events = list(coach_draft_stream("algo", [], "en", "es"))

    complete_kinds = [e[0] for e in events if e[0].endswith("_complete")]
    assert complete_kinds == ["meaning_complete", "option_spanish_complete", "option_english_complete"]
    # Option 1 (index 0) is fully usable; option 2 (index 1) never
    # completed - only whatever safely streamed of its Spanish text before
    # the model stopped, never an "_complete" for it.
    assert not any(e[0] == "option_spanish_complete" and e[1] == 1 for e in events)
    trailing_chunks = [e for e in events if e[0] == "option_spanish_chunk" and e[1] == 1]
    assert trailing_chunks
    assert "opt2 spanish, no marker yet".startswith("".join(e[2] for e in trailing_chunks))


def test_coach_draft_stream_includes_recent_history_in_prompt():
    chunks = ["CLEAN<<<VERDICT>>>", "x<<<MEANING>>>"]
    history = [("assistant", "¿Qué planes tienes para el fin de semana?"), ("user", "quiero ir playa")]
    with patch("app.translate.llm_translate.model_chat_stream", return_value=iter(chunks)) as mock_stream:
        list(coach_draft_stream("mañana quiero ir", history, "en", "es"))

    (messages,), _ = mock_stream.call_args
    prompt = messages[0]["content"]
    assert "¿Qué planes tienes para el fin de semana?" in prompt
    assert "quiero ir playa" in prompt


def test_coach_draft_stream_wraps_model_server_error():
    with patch("app.translate.llm_translate.model_chat_stream", side_effect=ModelServerUnavailableError("down")):
        with pytest.raises(TranslationUnavailableError, match="down"):
            list(coach_draft_stream("hola", [], "en", "es"))
