import asyncio
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import httpx
import pytest

from app.tts.deepinfra_client import TTSUnavailableError, synthesize


def _mock_async_client(fake_response):
    """synthesize() now uses `async with httpx.AsyncClient(...) as client:
    await client.post(...)` (see deepinfra_client.py's docstring on why -
    keeps a slow DeepInfra call from tying up a thread another endpoint
    needs from FastAPI's shared sync threadpool). Patches httpx.AsyncClient
    itself so the `async with ...` context manager yields a client whose
    `post` resolves to `fake_response`, without needing pytest-asyncio -
    each test just wraps the call in asyncio.run()."""
    client = MagicMock()
    client.post = AsyncMock(return_value=fake_response)
    context_manager = MagicMock()
    context_manager.__aenter__ = AsyncMock(return_value=client)
    context_manager.__aexit__ = AsyncMock(return_value=False)
    return patch("httpx.AsyncClient", return_value=context_manager), client


def test_synthesize_raises_when_token_missing():
    with patch("app.tts.deepinfra_client.DEEPINFRA_API_TOKEN", ""):
        with pytest.raises(TTSUnavailableError, match="not configured"):
            asyncio.run(synthesize("hola"))


def test_synthesize_returns_audio_bytes_on_success():
    fake_response = Mock(content=b"fake-mp3-bytes")
    fake_response.raise_for_status = Mock()
    client_patch, client = _mock_async_client(fake_response)

    with patch("app.tts.deepinfra_client.DEEPINFRA_API_TOKEN", "fake-token"):
        with client_patch:
            result = asyncio.run(synthesize("hola"))

    assert result == b"fake-mp3-bytes"
    _, kwargs = client.post.call_args
    assert kwargs["headers"]["Authorization"] == "Bearer fake-token"
    assert kwargs["json"]["input"] == "hola"


def test_synthesize_picks_voice_from_language():
    fake_response = Mock(content=b"fake-mp3-bytes")
    fake_response.raise_for_status = Mock()
    client_patch, client = _mock_async_client(fake_response)

    with patch("app.tts.deepinfra_client.DEEPINFRA_API_TOKEN", "fake-token"):
        with client_patch:
            asyncio.run(synthesize("bonjour", language="fr"))

    assert client.post.call_args.kwargs["json"]["voice"] == "ff_siwis"


def test_synthesize_explicit_voice_beats_language_default():
    fake_response = Mock(content=b"fake-mp3-bytes")
    fake_response.raise_for_status = Mock()
    client_patch, client = _mock_async_client(fake_response)

    with patch("app.tts.deepinfra_client.DEEPINFRA_API_TOKEN", "fake-token"):
        with client_patch:
            asyncio.run(synthesize("hola", language="es", voice="em_alex"))

    assert client.post.call_args.kwargs["json"]["voice"] == "em_alex"


def test_synthesize_env_override_beats_everything():
    fake_response = Mock(content=b"fake-mp3-bytes")
    fake_response.raise_for_status = Mock()
    client_patch, client = _mock_async_client(fake_response)

    with patch("app.tts.deepinfra_client.DEEPINFRA_API_TOKEN", "fake-token"):
        with patch("app.tts.deepinfra_client.DEEPINFRA_TTS_VOICE", "am_puck"):
            with client_patch:
                asyncio.run(synthesize("hola", language="es", voice="em_alex"))

    assert client.post.call_args.kwargs["json"]["voice"] == "am_puck"


def test_synthesize_wraps_http_status_error():
    request = httpx.Request("POST", "https://api.deepinfra.com/v1/audio/speech")
    response = httpx.Response(400, text="invalid voice", request=request)
    error = httpx.HTTPStatusError("bad request", request=request, response=response)

    fake_response = Mock()
    fake_response.raise_for_status = Mock(side_effect=error)
    client_patch, client = _mock_async_client(fake_response)

    with patch("app.tts.deepinfra_client.DEEPINFRA_API_TOKEN", "fake-token"):
        with client_patch:
            with pytest.raises(TTSUnavailableError, match="invalid voice"):
                asyncio.run(synthesize("hola"))

    # A real API error (not a transport failure) shouldn't be retried.
    assert client.post.call_count == 1


def test_synthesize_retries_once_on_timeout_then_succeeds():
    # Observed live twice: DeepInfra occasionally stalls past the per-
    # attempt timeout - a second attempt on a fresh connection often just
    # works.
    fake_response = Mock(content=b"fake-mp3-bytes")
    fake_response.raise_for_status = Mock()
    client = MagicMock()
    client.post = AsyncMock(side_effect=[httpx.ReadTimeout("timed out"), fake_response])
    context_manager = MagicMock()
    context_manager.__aenter__ = AsyncMock(return_value=client)
    context_manager.__aexit__ = AsyncMock(return_value=False)

    with patch("app.tts.deepinfra_client.DEEPINFRA_API_TOKEN", "fake-token"):
        with patch("httpx.AsyncClient", return_value=context_manager):
            result = asyncio.run(synthesize("hola"))

    assert result == b"fake-mp3-bytes"
    assert client.post.call_count == 2


def test_synthesize_raises_with_detail_after_exhausting_retries():
    client = MagicMock()
    client.post = AsyncMock(side_effect=httpx.ReadTimeout("timed out"))
    context_manager = MagicMock()
    context_manager.__aenter__ = AsyncMock(return_value=client)
    context_manager.__aexit__ = AsyncMock(return_value=False)

    with patch("app.tts.deepinfra_client.DEEPINFRA_API_TOKEN", "fake-token"):
        with patch("httpx.AsyncClient", return_value=context_manager):
            # repr(exc), not str(exc) - httpx's timeout exceptions often
            # stringify to "", which previously produced a blank-looking
            # error message with no clue what actually failed.
            with pytest.raises(TTSUnavailableError, match="ReadTimeout"):
                asyncio.run(synthesize("hola"))

    assert client.post.call_count == 3
