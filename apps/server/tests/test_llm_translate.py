from unittest.mock import patch

import pytest

from app.chat.llama_client import ModelServerUnavailableError
from app.translate.llm_translate import TranslationUnavailableError, interpret_user_input, translate_text


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
    # Model ignores the "Label: text" instruction and just answers in prose.
    with patch(
        "app.translate.llm_translate.llama_chat",
        side_effect=["Just a plain sentence with no labels.", "Solo una oración simple."],
    ) as mock_chat:
        native, target = interpret_user_input("some input", "en", "es")

    assert native == "Just a plain sentence with no labels."
    assert target == "Solo una oración simple."
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
