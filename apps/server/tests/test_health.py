from unittest.mock import Mock, patch

import httpx

from app.main import health_llm


def test_health_llm_ready_when_model_server_responds_200():
    with patch("httpx.get", return_value=Mock(status_code=200)):
        assert health_llm().ready is True


def test_health_llm_not_ready_when_model_server_still_loading():
    # llama-server itself reports 503 while the model weights are still
    # loading into memory, distinct from being unreachable altogether.
    with patch("httpx.get", return_value=Mock(status_code=503)):
        assert health_llm().ready is False


def test_health_llm_not_ready_when_model_server_unreachable():
    # The Railway-serverless-sleep case: the container hasn't even woken up
    # yet, so the request fails outright rather than getting any response.
    request = httpx.Request("GET", "http://model-server/health")
    with patch("httpx.get", side_effect=httpx.ConnectError("refused", request=request)):
        assert health_llm().ready is False
