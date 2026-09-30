from unittest.mock import patch

import pytest

from app.chat.llama_client import ModelServerUnavailableError
from app.translate.llm_translate import (
    TranslationUnavailableError,
    gloss_reply,
    interpret_user_input,
    translate_text,
)


def test_translate_text_empty_input_short_circuits():
    with patch("app.translate.llm_translate.llama_chat") as mock_chat:
        assert translate_text("   ", "es", "en") == ""
    mock_chat.assert_not_called()


def test_translate_text_calls_llm_with_language_names():
    with patch("app.translate.llm_translate.llama_chat", return_value="Hello, how are you?") as mock_chat:
        result = translate_text("Hola, ¿cómo estás?", "es", "en")

    assert result == "Hello, how are you?"
    (messages,), kwargs = mock_chat.call_args
    assert kwargs["logit_bias"] == {}
    assert messages[-1] == {"role": "user", "content": "Hola, ¿cómo estás?"}
    assert "Spanish" in messages[0]["content"]
    assert "English" in messages[0]["content"]


def test_translate_text_strips_whitespace():
    with patch("app.translate.llm_translate.llama_chat", return_value="  Hello  \n"):
        assert translate_text("Hola", "es", "en") == "Hello"


def test_translate_text_wraps_model_server_error():
    with patch("app.translate.llm_translate.llama_chat", side_effect=ModelServerUnavailableError("down")):
        with pytest.raises(TranslationUnavailableError, match="down"):
            translate_text("Hola", "es", "en")


def test_interpret_user_input_empty_short_circuits():
    with patch("app.translate.llm_translate.llama_chat") as mock_chat:
        assert interpret_user_input("   ", "en", "es") == ("", "")
    mock_chat.assert_not_called()


def test_interpret_user_input_parses_labeled_lines():
    reply = "English: Do you speak Spanish well?\nSpanish: ¿Hablas bien español?"
    with patch("app.translate.llm_translate.llama_chat", return_value=reply) as mock_chat:
        native, target = interpret_user_input("Do you hablo the espanol good?", "en", "es")

    assert native == "Do you speak Spanish well?"
    assert target == "¿Hablas bien español?"
    (messages,), kwargs = mock_chat.call_args
    assert kwargs["logit_bias"] == {}
    assert messages[-1] == {"role": "user", "content": "Do you hablo the espanol good?"}


def test_interpret_user_input_falls_back_when_format_not_followed():
    # Model ignores the "Label: text" instruction and just answers in prose
    # (but not so much longer than the input that the hallucination backstop
    # below also kicks in - that's covered by its own test).
    with patch(
        "app.translate.llm_translate.llama_chat",
        side_effect=["just some text", "un poco de texto"],
    ) as mock_chat:
        native, target = interpret_user_input("some input", "en", "es")

    assert native == "just some text"
    assert target == "un poco de texto"
    assert mock_chat.call_count == 2  # interpret call, then translate_text fallback


def test_interpret_user_input_wraps_model_server_error():
    with patch("app.translate.llm_translate.llama_chat", side_effect=ModelServerUnavailableError("down")):
        with pytest.raises(TranslationUnavailableError, match="down"):
            interpret_user_input("hola", "en", "es")


def test_interpret_user_input_swaps_mislabeled_content():
    # Observed live: the model labeled its two lines correctly but put the
    # wrong language's content under each one - the "English" row came back
    # showing Spanish text (and vice versa).
    reply = "English: ¿Hablas bien español?\nSpanish: Do you speak Spanish well?"
    with patch("app.translate.llm_translate.llama_chat", return_value=reply):
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
        "app.translate.llm_translate.llama_chat",
        side_effect=[reply, "qué tal, bro. había una vez un mono rojo"],
    ) as mock_chat:
        native, target = interpret_user_input("sup bro. there once was a red monkey", "en", "es")

    assert native == "sup bro. there once was a red monkey"
    assert target == "qué tal, bro. había una vez un mono rojo"
    assert mock_chat.call_count == 2  # interpret call, then the forced translate_text call


def test_interpret_user_input_backfills_missing_native_line():
    # The model produced only the target-language line.
    with patch(
        "app.translate.llm_translate.llama_chat",
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
        "app.translate.llm_translate.llama_chat",
        side_effect=["你好世界", "Hello"],
    ) as mock_chat:
        assert translate_text("Hola", "es", "en") == "Hello"
    assert mock_chat.call_count == 2


def test_translate_text_raises_if_still_garbled_after_retry():
    with patch("app.translate.llm_translate.llama_chat", return_value="你好世界"):
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
        "app.translate.llm_translate.llama_chat",
        side_effect=[reply, "Hola, señor."],
    ) as mock_chat:
        native, target = interpret_user_input("hello sir", "en", "es")

    assert native == "hello sir"
    assert target == "Hola, señor."
    assert mock_chat.call_count == 2  # interpret call, then the translate_text fallback


def test_interpret_user_input_prompt_warns_against_answering():
    with patch("app.translate.llm_translate.llama_chat", return_value="English: hi\nSpanish: hola") as mock_chat:
        interpret_user_input("hi", "en", "es")

    (messages,), _ = mock_chat.call_args
    prompt = messages[0]["content"]
    assert "hola bro" in prompt.lower()
    assert "not having a" in prompt.lower() or "not have a" in prompt.lower()


def test_gloss_reply_empty_input_short_circuits():
    with patch("app.translate.llm_translate.llama_chat") as mock_chat:
        assert gloss_reply("   ", "es", "en") == ("", {})
    mock_chat.assert_not_called()


def test_gloss_reply_parses_translation_and_word_map():
    reply = (
        '{"translation": "The bank is closed.", "words": {'
        '"banco": {"gloss": "bank", "note": "financial institution, not a bench"}, '
        '"cerrado": {"gloss": "closed", "note": ""}}}'
    )
    with patch("app.translate.llm_translate.llama_chat", return_value=reply) as mock_chat:
        translation, words = gloss_reply("El banco está cerrado.", "es", "en")

    assert translation == "The bank is closed."
    assert words == {
        "banco": ("bank", "financial institution, not a bench"),
        "cerrado": ("closed", ""),
    }
    (messages,), kwargs = mock_chat.call_args
    assert kwargs["logit_bias"] == {}
    assert kwargs["max_tokens"] > 300  # bigger budget than a plain translation
    assert messages[-1] == {"role": "user", "content": "El banco está cerrado."}


def test_gloss_reply_tolerates_plain_string_word_values():
    # The {"gloss": ..., "note": ...} wrapper is the requested shape, but a
    # small model can drop it for a word with nothing to note - still a
    # usable gloss, just without a description.
    reply = '{"translation": "Hello", "words": {"hola": "hello"}}'
    with patch("app.translate.llm_translate.llama_chat", return_value=reply):
        translation, words = gloss_reply("Hola", "es", "en")

    assert translation == "Hello"
    assert words == {"hola": ("hello", "")}


def test_gloss_reply_strips_surrounding_prose_and_code_fences():
    reply = (
        'Sure, here you go:\n```json\n{"translation": "Hello", '
        '"words": {"hola": {"gloss": "hello", "note": ""}}}\n```'
    )
    with patch("app.translate.llm_translate.llama_chat", return_value=reply):
        translation, words = gloss_reply("Hola", "es", "en")

    assert translation == "Hello"
    assert words == {"hola": ("hello", "")}


def test_gloss_reply_retries_once_on_unparseable_reply():
    with patch(
        "app.translate.llm_translate.llama_chat",
        side_effect=["not json at all", '{"translation": "Hi", "words": {"hola": {"gloss": "hi", "note": ""}}}'],
    ) as mock_chat:
        translation, words = gloss_reply("Hola", "es", "en")

    assert translation == "Hi"
    assert words == {"hola": ("hi", "")}
    assert mock_chat.call_count == 2


def test_gloss_reply_raises_after_exhausting_retries():
    with patch("app.translate.llm_translate.llama_chat", return_value="not json at all"):
        with pytest.raises(TranslationUnavailableError):
            gloss_reply("Hola", "es", "en")


def test_gloss_reply_raises_on_garbled_reply():
    with patch("app.translate.llm_translate.llama_chat", return_value="你好世界"):
        with pytest.raises(TranslationUnavailableError):
            gloss_reply("Hola", "es", "en")


def test_gloss_reply_wraps_model_server_error():
    with patch("app.translate.llm_translate.llama_chat", side_effect=ModelServerUnavailableError("down")):
        with pytest.raises(TranslationUnavailableError, match="down"):
            gloss_reply("Hola", "es", "en")


def test_interpret_user_input_falls_back_to_translate_text_on_garbled_reply():
    # Both interpret attempts come back garbled - falls back to a plain
    # translate_text call on the raw input rather than parsing garbage.
    # The raw input reads as majority-English, so native_text is used as-is
    # (no further translate_text call needed for it) and only the target
    # (Spanish) line needs a real translation call.
    with patch(
        "app.translate.llm_translate.llama_chat",
        side_effect=["你好世界", "你好世界", "¿Hablas bien español?"],
    ) as mock_chat:
        native, target = interpret_user_input("Do you hablo the espanol good?", "en", "es")

    assert native == "Do you hablo the espanol good?"
    assert target == "¿Hablas bien español?"
    assert mock_chat.call_count == 3
