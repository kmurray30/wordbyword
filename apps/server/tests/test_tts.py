from unittest.mock import Mock, patch

import httpx
import pytest

from app.tts.deepinfra_client import TTSUnavailableError, synthesize


def test_synthesize_raises_when_token_missing():
    with patch("app.tts.deepinfra_client.DEEPINFRA_API_TOKEN", ""):
        with pytest.raises(TTSUnavailableError, match="not configured"):
            synthesize("hola")


def test_synthesize_returns_audio_bytes_on_success():
    fake_response = Mock(content=b"fake-mp3-bytes")
    fake_response.raise_for_status = Mock()

    with patch("app.tts.deepinfra_client.DEEPINFRA_API_TOKEN", "fake-token"):
        with patch("httpx.post", return_value=fake_response) as mock_post:
            result = synthesize("hola")

    assert result == b"fake-mp3-bytes"
    _, kwargs = mock_post.call_args
    assert kwargs["headers"]["Authorization"] == "Bearer fake-token"
    assert kwargs["json"]["input"] == "hola"


def test_synthesize_picks_voice_from_language():
    fake_response = Mock(content=b"fake-mp3-bytes")
    fake_response.raise_for_status = Mock()

    with patch("app.tts.deepinfra_client.DEEPINFRA_API_TOKEN", "fake-token"):
        with patch("httpx.post", return_value=fake_response) as mock_post:
            synthesize("bonjour", language="fr")

    assert mock_post.call_args.kwargs["json"]["voice"] == "ff_siwis"


def test_synthesize_voice_override_beats_language_map():
    fake_response = Mock(content=b"fake-mp3-bytes")
    fake_response.raise_for_status = Mock()

    with patch("app.tts.deepinfra_client.DEEPINFRA_API_TOKEN", "fake-token"):
        with patch("app.tts.deepinfra_client.DEEPINFRA_TTS_VOICE", "am_puck"):
            with patch("httpx.post", return_value=fake_response) as mock_post:
                synthesize("hola", language="es")

    assert mock_post.call_args.kwargs["json"]["voice"] == "am_puck"


def test_synthesize_wraps_http_status_error():
    request = httpx.Request("POST", "https://api.deepinfra.com/v1/audio/speech")
    response = httpx.Response(400, text="invalid voice", request=request)
    error = httpx.HTTPStatusError("bad request", request=request, response=response)

    fake_response = Mock()
    fake_response.raise_for_status = Mock(side_effect=error)

    with patch("app.tts.deepinfra_client.DEEPINFRA_API_TOKEN", "fake-token"):
        with patch("httpx.post", return_value=fake_response):
            with pytest.raises(TTSUnavailableError, match="invalid voice"):
                synthesize("hola")
