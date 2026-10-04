from unittest.mock import patch

from app.main import health_llm


def test_health_llm_ready_when_key_configured():
    with patch("app.main.OPENAI_API_KEY", "sk-fake"):
        assert health_llm().ready is True


def test_health_llm_not_ready_without_a_key():
    with patch("app.main.OPENAI_API_KEY", ""):
        assert health_llm().ready is False
