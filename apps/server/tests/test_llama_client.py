from unittest.mock import Mock, patch

import httpx
import pytest

from app.chat.llama_client import ModelServerUnavailableError, chat


def _response(content: str = "hola"):
    resp = Mock()
    resp.raise_for_status = Mock()
    resp.json = Mock(return_value={"choices": [{"message": {"content": content}}]})
    return resp


def test_chat_succeeds_on_first_attempt_no_retry():
    with (
        patch("httpx.post", return_value=_response("hola")) as mock_post,
        patch("app.chat.llama_client.time.sleep") as mock_sleep,
    ):
        result = chat([{"role": "user", "content": "hi"}], {})

    assert result == "hola"
    assert mock_post.call_count == 1
    mock_sleep.assert_not_called()


def test_chat_retries_on_502_then_succeeds():
    # model-server runs with Railway's Serverless mode - a 502 right after
    # a quiet period is the documented "waking up a sleeping container"
    # signature, not a real outage.
    request = httpx.Request("POST", "http://model-server/v1/chat/completions")
    error_response = httpx.Response(502, request=request)
    error = httpx.HTTPStatusError("bad gateway", request=request, response=error_response)

    bad = Mock()
    bad.raise_for_status = Mock(side_effect=error)
    good = _response("hola")

    with (
        patch("httpx.post", side_effect=[bad, good]) as mock_post,
        patch("app.chat.llama_client.time.sleep") as mock_sleep,
    ):
        result = chat([{"role": "user", "content": "hi"}], {})

    assert result == "hola"
    assert mock_post.call_count == 2
    mock_sleep.assert_called_once()


def test_chat_raises_after_exhausting_retries_on_persistent_502():
    request = httpx.Request("POST", "http://model-server/v1/chat/completions")
    error_response = httpx.Response(502, request=request)
    error = httpx.HTTPStatusError("bad gateway", request=request, response=error_response)

    bad = Mock()
    bad.raise_for_status = Mock(side_effect=error)

    with (
        patch("httpx.post", return_value=bad) as mock_post,
        patch("app.chat.llama_client.time.sleep"),
    ):
        with pytest.raises(ModelServerUnavailableError):
            chat([{"role": "user", "content": "hi"}], {})

    assert mock_post.call_count == 3  # _MAX_ATTEMPTS


def test_chat_does_not_retry_on_non_gateway_status_error():
    request = httpx.Request("POST", "http://model-server/v1/chat/completions")
    error_response = httpx.Response(400, request=request)
    error = httpx.HTTPStatusError("bad request", request=request, response=error_response)

    bad = Mock()
    bad.raise_for_status = Mock(side_effect=error)

    with (
        patch("httpx.post", return_value=bad) as mock_post,
        patch("app.chat.llama_client.time.sleep") as mock_sleep,
    ):
        with pytest.raises(ModelServerUnavailableError):
            chat([{"role": "user", "content": "hi"}], {})

    assert mock_post.call_count == 1  # a real 4xx won't fix itself on retry
    mock_sleep.assert_not_called()


def test_chat_does_not_retry_on_timeout():
    request = httpx.Request("POST", "http://model-server/v1/chat/completions")

    with (
        patch("httpx.post", side_effect=httpx.ReadTimeout("timed out", request=request)) as mock_post,
        patch("app.chat.llama_client.time.sleep") as mock_sleep,
    ):
        with pytest.raises(ModelServerUnavailableError, match="didn't respond within"):
            chat([{"role": "user", "content": "hi"}], {}, timeout=5.0)

    assert mock_post.call_count == 1
    mock_sleep.assert_not_called()


def test_chat_retries_on_connection_error():
    request = httpx.Request("POST", "http://model-server/v1/chat/completions")
    with (
        patch(
            "httpx.post",
            side_effect=[httpx.ConnectError("refused", request=request), _response("hola")],
        ) as mock_post,
        patch("app.chat.llama_client.time.sleep"),
    ):
        result = chat([{"role": "user", "content": "hi"}], {})

    assert result == "hola"
    assert mock_post.call_count == 2
