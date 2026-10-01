from unittest.mock import Mock, patch

import httpx
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db import Base
from app.main import health_llm


def _session():
    engine = create_engine("sqlite:///:memory:")
    from app import models  # noqa: F401  (register tables on Base)

    Base.metadata.create_all(bind=engine)
    return sessionmaker(bind=engine)()


def test_health_llm_ready_when_model_server_responds_200():
    with patch("httpx.get", return_value=Mock(status_code=200)):
        assert health_llm(session=_session()).ready is True


def test_health_llm_not_ready_when_model_server_still_loading():
    # llama-server itself reports 503 while the model weights are still
    # loading into memory, distinct from being unreachable altogether.
    with patch("httpx.get", return_value=Mock(status_code=503)):
        assert health_llm(session=_session()).ready is False


def test_health_llm_not_ready_when_model_server_unreachable():
    # The Railway-serverless-sleep case: the container hasn't even woken up
    # yet, so the request fails outright rather than getting any response.
    request = httpx.Request("GET", "http://model-server/health")
    with patch("httpx.get", side_effect=httpx.ConnectError("refused", request=request)):
        assert health_llm(session=_session()).ready is False


def test_health_llm_ready_under_openai_provider_without_pinging_model_server():
    # No self-hosted model server to wait on in this mode - ready as soon
    # as a key is configured, and httpx.get (model-server's own /health)
    # should never even be called.
    session = _session()
    from app import settings_store

    settings_store.update_settings(session, model_provider="openai")
    with (
        patch("httpx.get") as mock_get,
        patch("app.main.OPENAI_API_KEY", "sk-fake"),
    ):
        assert health_llm(session=session).ready is True
    mock_get.assert_not_called()


def test_health_llm_not_ready_under_openai_provider_without_a_key():
    session = _session()
    from app import settings_store

    settings_store.update_settings(session, model_provider="openai")
    with patch("app.main.OPENAI_API_KEY", ""):
        assert health_llm(session=session).ready is False
