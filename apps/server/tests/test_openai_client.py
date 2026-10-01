from unittest.mock import Mock, patch

import httpx
import pytest

from app.chat.openai_client import ModelServerUnavailableError, chat


def _response(content: str = "hola"):
    resp = Mock()
    resp.raise_for_status = Mock()
    resp.json = Mock(return_value={"choices": [{"message": {"content": content}}]})
    return resp


def test_chat_raises_when_key_missing():
    with patch("app.chat.openai_client.OPENAI_API_KEY", ""):
        with pytest.raises(ModelServerUnavailableError, match="OPENAI_API_KEY"):
            chat([{"role": "user", "content": "hi"}])


def test_chat_returns_content_and_sends_configured_model():
    with (
        patch("app.chat.openai_client.OPENAI_API_KEY", "sk-fake"),
        patch("app.chat.openai_client.OPENAI_CHAT_MODEL", "gpt-6-luna"),
        patch("httpx.post", return_value=_response("hola")) as mock_post,
    ):
        result = chat([{"role": "user", "content": "hi"}])

    assert result == "hola"
    _, kwargs = mock_post.call_args
    assert kwargs["json"]["model"] == "gpt-6-luna"
    assert kwargs["headers"]["Authorization"] == "Bearer sk-fake"


def test_chat_ignores_logit_bias_without_erroring():
    # logit_bias is accepted for a uniform call signature with
    # llama_client.chat (see model_client.py) but is meaningless against
    # this API's own tokenizer - just silently not sent.
    with (
        patch("app.chat.openai_client.OPENAI_API_KEY", "sk-fake"),
        patch("httpx.post", return_value=_response("hola")) as mock_post,
    ):
        result = chat([{"role": "user", "content": "hi"}], logit_bias={123: 1.0})

    assert result == "hola"
    assert "logit_bias" not in mock_post.call_args.kwargs["json"]


def test_chat_wraps_http_status_error():
    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    response = httpx.Response(401, text="invalid api key", request=request)
    error = httpx.HTTPStatusError("unauthorized", request=request, response=response)

    bad = Mock()
    bad.raise_for_status = Mock(side_effect=error)

    with (
        patch("app.chat.openai_client.OPENAI_API_KEY", "sk-fake"),
        patch("httpx.post", return_value=bad),
    ):
        with pytest.raises(ModelServerUnavailableError, match="invalid api key"):
            chat([{"role": "user", "content": "hi"}])


def test_chat_wraps_connection_error():
    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    with (
        patch("app.chat.openai_client.OPENAI_API_KEY", "sk-fake"),
        patch("httpx.post", side_effect=httpx.ConnectError("refused", request=request)),
    ):
        with pytest.raises(ModelServerUnavailableError, match="Could not reach"):
            chat([{"role": "user", "content": "hi"}])


def test_chat_returns_empty_string_when_no_choices():
    with (
        patch("app.chat.openai_client.OPENAI_API_KEY", "sk-fake"),
        patch("httpx.post", return_value=Mock(raise_for_status=Mock(), json=Mock(return_value={"choices": []}))),
    ):
        assert chat([{"role": "user", "content": "hi"}]) == ""
