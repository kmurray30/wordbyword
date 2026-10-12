import asyncio
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import httpx
import pytest

from app.tts import elevenlabs_client
from app.tts.deepinfra_client import TTSUnavailableError
from app.tts.elevenlabs_client import pick_voice, synthesize


def _mock_async_client(fake_response=None, side_effect=None, voices_response=None):
    client = MagicMock()
    client.post = AsyncMock(return_value=fake_response, side_effect=side_effect)
    client.get = AsyncMock(return_value=voices_response)
    context_manager = MagicMock()
    context_manager.__aenter__ = AsyncMock(return_value=client)
    context_manager.__aexit__ = AsyncMock(return_value=False)
    return patch("httpx.AsyncClient", return_value=context_manager), client


@pytest.fixture(autouse=True)
def _reset_discovered_voice():
    elevenlabs_client._discovered_voice_id = None
    yield
    elevenlabs_client._discovered_voice_id = None


def test_synthesize_raises_when_key_missing():
    with patch("app.tts.elevenlabs_client.ELEVENLABS_API_KEY", ""):
        with pytest.raises(TTSUnavailableError, match="not configured"):
            asyncio.run(synthesize("hello"))


def test_synthesize_posts_to_the_configured_voice_with_the_multilingual_model():
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
    client.get.assert_not_awaited()  # a configured id skips discovery


def test_pick_voice_prefers_the_accounts_premade_english_voice():
    voices = [
        {"voice_id": "lib1", "category": "professional", "labels": {"language": "en"}},
        {"voice_id": "pre-es", "category": "premade", "labels": {"language": "es"}},
        {"voice_id": "pre-en", "category": "premade", "labels": {"language": "en"}},
    ]
    assert pick_voice(voices) == "pre-en"
    assert pick_voice([{"voice_id": "only", "category": "cloned"}]) == "only"
    assert pick_voice([]) is None


def test_synthesize_discovers_a_voice_once_when_none_is_configured():
    # Observed live: the free plan rejects a hard-coded library voice id
    # with a 402 - the account's own listed voices are what work.
    fake_response = Mock(content=b"mp3-bytes")
    fake_response.raise_for_status = Mock()
    voices_response = Mock()
    voices_response.raise_for_status = Mock()
    voices_response.json = Mock(
        return_value={"voices": [{"voice_id": "pre-en", "category": "premade", "labels": {"language": "en"}}]}
    )
    client_patch, client = _mock_async_client(fake_response, voices_response=voices_response)

    with (
        patch("app.tts.elevenlabs_client.ELEVENLABS_API_KEY", "xi-fake"),
        patch("app.tts.elevenlabs_client.ELEVENLABS_COACH_VOICE_ID", ""),
        client_patch,
    ):
        asyncio.run(synthesize("hello"))
        asyncio.run(synthesize("hello again"))

    assert client.get.await_count == 1  # cached after the first resolve
    (url,), _ = client.post.call_args
    assert url == "https://api.elevenlabs.io/v1/text-to-speech/pre-en"


def test_synthesize_fails_clearly_when_the_account_has_no_voices():
    voices_response = Mock()
    voices_response.raise_for_status = Mock()
    voices_response.json = Mock(return_value={"voices": []})
    client_patch, _ = _mock_async_client(voices_response=voices_response)

    with (
        patch("app.tts.elevenlabs_client.ELEVENLABS_API_KEY", "xi-fake"),
        patch("app.tts.elevenlabs_client.ELEVENLABS_COACH_VOICE_ID", ""),
        client_patch,
    ):
        with pytest.raises(TTSUnavailableError, match="ELEVENLABS_COACH_VOICE_ID"):
            asyncio.run(synthesize("hello"))


def test_synthesize_does_not_retry_an_http_error():
    fake_response = Mock(status_code=401, text="bad key")
    fake_response.raise_for_status = Mock(
        side_effect=httpx.HTTPStatusError("401", request=Mock(), response=fake_response)
    )
    client_patch, client = _mock_async_client(fake_response)

    with (
        patch("app.tts.elevenlabs_client.ELEVENLABS_API_KEY", "xi-fake"),
        patch("app.tts.elevenlabs_client.ELEVENLABS_COACH_VOICE_ID", "voice123"),
        client_patch,
    ):
        with pytest.raises(TTSUnavailableError, match="401"):
            asyncio.run(synthesize("hello"))
    assert client.post.call_count == 1


def test_synthesize_retries_transport_errors_three_times_then_raises():
    client_patch, client = _mock_async_client(side_effect=httpx.ConnectTimeout("slow"))

    with (
        patch("app.tts.elevenlabs_client.ELEVENLABS_API_KEY", "xi-fake"),
        patch("app.tts.elevenlabs_client.ELEVENLABS_COACH_VOICE_ID", "voice123"),
        client_patch,
    ):
        with pytest.raises(TTSUnavailableError, match="after 3 attempts"):
            asyncio.run(synthesize("hello"))
    assert client.post.call_count == 3


def test_synthesize_retries_a_429_then_succeeds():
    # Observed live: the third of three parallel coach sentences failed -
    # ElevenLabs caps concurrent requests per plan. A 429 is retried after
    # a short wait rather than surfaced as a failed clip.
    limited = Mock(status_code=429, text="too_many_concurrent_requests")
    limited.raise_for_status = Mock(side_effect=httpx.HTTPStatusError("429", request=Mock(), response=limited))
    ok = Mock(content=b"mp3-bytes")
    ok.raise_for_status = Mock()
    client_patch, client = _mock_async_client()
    client.post = AsyncMock(side_effect=[limited, ok])

    with (
        patch("app.tts.elevenlabs_client.ELEVENLABS_API_KEY", "xi-fake"),
        patch("app.tts.elevenlabs_client.ELEVENLABS_COACH_VOICE_ID", "voice123"),
        patch("app.tts.elevenlabs_client._RETRY_AFTER_429_S", 0),
        client_patch,
    ):
        assert asyncio.run(synthesize("hello")) == b"mp3-bytes"
    assert client.post.call_count == 2
