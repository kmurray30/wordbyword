from unittest.mock import patch

from app.chat.model_client import chat


def test_chat_routes_to_llama_client_by_default():
    with (
        patch("app.chat.model_client.llama_client.chat", return_value="hola") as mock_llama,
        patch("app.chat.model_client.openai_client.chat") as mock_openai,
    ):
        result = chat([{"role": "user", "content": "hi"}], {})

    assert result == "hola"
    mock_llama.assert_called_once()
    mock_openai.assert_not_called()


def test_chat_routes_to_llama_client_when_provider_explicitly_local():
    with (
        patch("app.chat.model_client.llama_client.chat", return_value="hola") as mock_llama,
        patch("app.chat.model_client.openai_client.chat") as mock_openai,
    ):
        chat([{"role": "user", "content": "hi"}], {}, provider="local")

    mock_llama.assert_called_once()
    mock_openai.assert_not_called()


def test_chat_routes_to_openai_client_when_provider_is_openai():
    with (
        patch("app.chat.model_client.llama_client.chat") as mock_llama,
        patch("app.chat.model_client.openai_client.chat", return_value="hello") as mock_openai,
    ):
        result = chat([{"role": "user", "content": "hi"}], {}, provider="openai")

    assert result == "hello"
    mock_openai.assert_called_once()
    mock_llama.assert_not_called()


def test_chat_falls_back_to_config_default_when_provider_omitted():
    with (
        patch("app.chat.model_client.MODEL_PROVIDER", "openai"),
        patch("app.chat.model_client.llama_client.chat") as mock_llama,
        patch("app.chat.model_client.openai_client.chat", return_value="hello") as mock_openai,
    ):
        result = chat([{"role": "user", "content": "hi"}], {})

    assert result == "hello"
    mock_openai.assert_called_once()
    mock_llama.assert_not_called()


def test_chat_passes_through_timeout_and_max_tokens():
    with (
        patch("app.chat.model_client.llama_client.chat", return_value="hola") as mock_llama,
    ):
        chat([{"role": "user", "content": "hi"}], {"1": 0.5}, timeout=5.0, max_tokens=42)

    _, kwargs = mock_llama.call_args
    assert kwargs["timeout"] == 5.0
    assert kwargs["max_tokens"] == 42
