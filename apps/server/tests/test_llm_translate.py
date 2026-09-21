from unittest.mock import patch

import pytest

from app.chat.llama_client import ModelServerUnavailableError
from app.translate.llm_translate import TranslationUnavailableError, translate_text


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
