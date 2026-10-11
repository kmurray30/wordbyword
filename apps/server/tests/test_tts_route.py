import asyncio
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException

from app.routes.tts import speak
from app.schemas import TTSRequest
from app.tts.deepinfra_client import TTSUnavailableError


def test_partner_role_uses_the_kokoro_path_by_default():
    with (
        patch("app.routes.tts.synthesize", AsyncMock(return_value=b"kokoro")) as partner,
        patch("app.routes.tts.elevenlabs_client.synthesize", AsyncMock(return_value=b"eleven")) as coach,
    ):
        response = asyncio.run(speak(TTSRequest(text="Hola", language="es", voice="ef_dora")))

    assert response.body == b"kokoro"
    assert response.media_type == "audio/mpeg"
    partner.assert_awaited_once_with("Hola", language="es", voice="ef_dora")
    coach.assert_not_awaited()


def test_coach_role_uses_elevenlabs_and_ignores_partner_voice_fields():
    with (
        patch("app.routes.tts.synthesize", AsyncMock(return_value=b"kokoro")) as partner,
        patch("app.routes.tts.elevenlabs_client.synthesize", AsyncMock(return_value=b"eleven")) as coach,
    ):
        # `voice` would be invalid for a partner request; the coach path
        # must not even look at it.
        response = asyncio.run(speak(TTSRequest(text="Say: ¿Te gusta correr?", role="coach", voice="not-a-voice")))

    assert response.body == b"eleven"
    coach.assert_awaited_once_with("Say: ¿Te gusta correr?")
    partner.assert_not_awaited()


def test_unknown_role_is_a_400():
    with pytest.raises(HTTPException) as excinfo:
        asyncio.run(speak(TTSRequest(text="hola", role="narrator")))
    assert excinfo.value.status_code == 400


def test_coach_provider_unavailable_is_a_503():
    with patch("app.routes.tts.elevenlabs_client.synthesize", AsyncMock(side_effect=TTSUnavailableError("no key"))):
        with pytest.raises(HTTPException) as excinfo:
            asyncio.run(speak(TTSRequest(text="hola", role="coach")))
    assert excinfo.value.status_code == 503
    assert "no key" in excinfo.value.detail
