import asyncio
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException

from app.routes.tts import list_voices, speak
from app.schemas import TTSRequest
from app.tts.elevenlabs_client import pick_partner_voice, picker_order
from app.tts.errors import TTSUnavailableError

VOICES = [
    {"voice_id": "rachel", "name": "Rachel", "category": "premade", "labels": {"accent": "american"}},
    {"voice_id": "mine", "name": "My clone", "category": "cloned", "labels": {}},
    {"voice_id": "sofia", "name": "Sofía", "category": "premade", "labels": {"accent": "latin american"}},
]


def _patch_voices(coach="rachel", partner="sofia"):
    return (
        patch("app.routes.tts.elevenlabs_client.coach_voice_id", AsyncMock(return_value=coach)),
        patch("app.routes.tts.elevenlabs_client.partner_voice_id", AsyncMock(return_value=partner)),
    )


def test_partner_speaks_in_the_chosen_voice():
    c, p = _patch_voices()
    with c, p, patch("app.routes.tts.elevenlabs_client.synthesize", AsyncMock(return_value=b"mp3")) as synth:
        response = asyncio.run(speak(TTSRequest(text="Hola", voice="mine")))
    assert response.body == b"mp3"
    assert response.media_type == "audio/mpeg"
    synth.assert_awaited_once_with("Hola", voice_id="mine")


def test_partner_without_a_voice_uses_the_partner_default():
    c, p = _patch_voices()
    with c, p, patch("app.routes.tts.elevenlabs_client.synthesize", AsyncMock(return_value=b"mp3")) as synth:
        asyncio.run(speak(TTSRequest(text="Hola")))
    synth.assert_awaited_once_with("Hola", voice_id="sofia")


def test_coach_uses_the_coach_voice_and_ignores_voice():
    c, p = _patch_voices()
    with c, p, patch("app.routes.tts.elevenlabs_client.synthesize", AsyncMock(return_value=b"mp3")) as synth:
        asyncio.run(speak(TTSRequest(text="Say: ¿Te gusta correr?", role="coach", voice="mine")))
    synth.assert_awaited_once_with("Say: ¿Te gusta correr?", voice_id="rachel")


def test_unknown_role_is_a_400():
    with pytest.raises(HTTPException) as excinfo:
        asyncio.run(speak(TTSRequest(text="hola", role="narrator")))
    assert excinfo.value.status_code == 400


def test_provider_unavailable_is_a_503():
    c, p = _patch_voices()
    with c, p, patch("app.routes.tts.elevenlabs_client.synthesize", AsyncMock(side_effect=TTSUnavailableError("quota"))):
        with pytest.raises(HTTPException) as excinfo:
            asyncio.run(speak(TTSRequest(text="hola", role="coach")))
    assert excinfo.value.status_code == 503
    assert "quota" in excinfo.value.detail


def test_voices_lists_the_accounts_voices_with_the_partner_default():
    _, p = _patch_voices()
    with p, patch("app.routes.tts.elevenlabs_client.account_voices", AsyncMock(return_value=VOICES)):
        res = asyncio.run(list_voices())
    assert [v.id for v in res.voices] == ["mine", "sofia", "rachel"]
    assert res.voices[1].name == "Sofía"
    assert res.default == "sofia"


def test_voices_unavailable_is_a_503():
    with patch("app.routes.tts.elevenlabs_client.account_voices", AsyncMock(side_effect=TTSUnavailableError("no key"))):
        with pytest.raises(HTTPException) as excinfo:
            asyncio.run(list_voices())
    assert excinfo.value.status_code == 503


def test_picker_order_puts_own_then_spanish_voices_first():
    assert [v["voice_id"] for v in picker_order(VOICES)] == ["mine", "sofia", "rachel"]


def test_partner_voice_is_never_the_coach_voice():
    assert pick_partner_voice(VOICES, coach_id="mine") == "sofia"
    assert pick_partner_voice([VOICES[0]], coach_id="rachel") == "rachel"  # only one voice: share it


def test_kokoro_provider_speaks_with_its_language_voice():
    with patch("app.routes.tts.deepinfra_client.synthesize", AsyncMock(return_value=b"kokoro")) as kokoro:
        response = asyncio.run(speak(TTSRequest(text="Hola", language="es", voice="ef_dora", provider="kokoro")))
    assert response.body == b"kokoro"
    kokoro.assert_awaited_once_with("Hola", language="es", voice="ef_dora")


def test_kokoro_rejects_a_voice_of_another_language():
    with pytest.raises(HTTPException) as excinfo:
        asyncio.run(speak(TTSRequest(text="Hola", language="es", voice="af_heart", provider="kokoro")))
    assert excinfo.value.status_code == 400


def test_kokoro_coach_uses_the_language_default_voice():
    with patch("app.routes.tts.deepinfra_client.synthesize", AsyncMock(return_value=b"k")) as kokoro:
        asyncio.run(speak(TTSRequest(text="Say this", language="en", voice="mine", role="coach", provider="kokoro")))
    kokoro.assert_awaited_once_with("Say this", language="en", voice=None)


def test_kokoro_voices_are_that_languages_voices():
    res = asyncio.run(list_voices(language="es", provider="kokoro"))
    assert res.provider == "kokoro"
    assert "ef_dora" in [v.id for v in res.voices]
    assert next(v.name for v in res.voices if v.id == "ef_dora") == "Dora"
    assert res.default == "ef_dora"


def test_default_provider_is_elevenlabs():
    c, p = _patch_voices()
    with (
        c,
        p,
        patch("app.routes.tts.elevenlabs_client.synthesize", AsyncMock(return_value=b"e")) as eleven,
        patch("app.routes.tts.deepinfra_client.synthesize", AsyncMock(return_value=b"k")) as kokoro,
    ):
        asyncio.run(speak(TTSRequest(text="Hola")))
    eleven.assert_awaited_once()
    kokoro.assert_not_awaited()


def test_unknown_provider_is_a_400():
    with pytest.raises(HTTPException) as excinfo:
        asyncio.run(speak(TTSRequest(text="hola", provider="espeak")))
    assert excinfo.value.status_code == 400
