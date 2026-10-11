import asyncio
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import httpx
import pytest

from app.tts.deepinfra_client import TTSUnavailableError
from app.tts.elevenlabs_client import synthesize


def _mock_async_client(fake_response=None, side_effect=None):
    client = MagicMock()
    client.post = AsyncMock(return_value=fake_response, side_effect=side_effect)
    context_manager = MagicMock()
    context_manager.__aenter__ = AsyncMock(return_value=client)
    context_manager.__aexit__ = AsyncMock(return_value=False)
    return patch("httpx.AsyncClient", return_value=context_manager), client


def test_synthesize_raises_when_key_missing():
    with patch("app.tts.elevenlabs_client.ELEVENLABS_API_KEY", ""):
        with pytest.raises(TTSUnavailableError, match="not configured"):
            asyncio.run(synthesize("hello"))


def test_synthesize_posts_to_the_coach_voice_with_the_multilingual_model():
    fake_response = Mock(content=b"mp3-bytes")
    fake_response.raise_for_status = Mock()
    client_patch, client = _mock_async_client(fake_response)

    with (
        patch("app.tts.elevenlabs_client.ELEVENLABS_API_KEY", "xi-fake"),
        patch("app.tts.elevenlabs_client.ELEVENLABS_COACH_VOICE_ID", "voice123"),
        patch("app.tts.elevenlabs_client.ELEVENLABS_TTS_MODEL", "eleven_multilingual_v2"),
        client_patch,
    ):
        result = asyncio.run(synthesize("Say: ¿Te gusta correr?"))

    assert result == b"mp3-bytes"
    (url,), kwargs = client.post.call_args
    assert url == "https://api.elevenlabs.io/v1/text-to-speech/voice123"
    assert kwargs["headers"]["xi-api-key"] == "xi-fake"
    assert kwargs["json"]["text"] == "Say: ¿Te gusta correr?"
    assert kwargs["json"]["model_id"] == "eleven_multilingual_v2"
    assert "language_code" not in kwargs["json"]  # mixed-language text: let the model detect
    assert kwargs["params"]["output_format"].startswith("mp3")


def test_synthesize_does_not_retry_an_http_error():
    fake_response = Mock(status_code=401, text="bad key")
    fake_response.raise_for_status = Mock(
        side_effect=httpx.HTTPStatusError("401", request=Mock(), response=fake_response)
    )
    client_patch, client = _mock_async_client(fake_response)

    with patch("app.tts.elevenlabs_client.ELEVENLABS_API_KEY", "xi-fake"), client_patch:
        with pytest.raises(TTSUnavailableError, match="401"):
            asyncio.run(synthesize("hello"))
    assert client.post.call_count == 1


def test_synthesize_retries_transport_errors_three_times_then_raises():
    client_patch, client = _mock_async_client(side_effect=httpx.ConnectTimeout("slow"))

    with patch("app.tts.elevenlabs_client.ELEVENLABS_API_KEY", "xi-fake"), client_patch:
        with pytest.raises(TTSUnavailableError, match="after 3 attempts"):
            asyncio.run(synthesize("hello"))
    assert client.post.call_count == 3
