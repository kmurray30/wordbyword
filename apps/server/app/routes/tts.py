from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

from app.config import TTS_PROVIDER, TTS_PROVIDERS
from app.schemas import TTSRequest, TTSVoice, TTSVoicesResponse
from app.tts import deepinfra_client, elevenlabs_client
from app.tts.errors import TTSUnavailableError
from app.tts.kokoro_voices import voice_for_language, voices_for_language

router = APIRouter(prefix="/tts", tags=["tts"])


def _provider(name: str | None) -> str:
    provider = name or TTS_PROVIDER
    if provider not in TTS_PROVIDERS:
        raise HTTPException(status_code=400, detail=f"unknown TTS provider: {provider!r}")
    return provider


def _kokoro_name(voice_id: str) -> str:
    # "ef_dora" -> "Dora": Kokoro voice ids are "<lang><gender>_<name>".
    name = voice_id.split("_", 1)[-1]
    return name[:1].upper() + name[1:]


@router.get("/voices", response_model=TTSVoicesResponse)
async def list_voices(language: str = "es", provider: str | None = None) -> TTSVoicesResponse:
    """The partner voices a provider offers, and its default. ElevenLabs:
    the account's voices (see picker_order) - multilingual, so `language`
    doesn't narrow them. Kokoro: that language's voices."""
    provider = _provider(provider)
    if provider == "kokoro":
        voices = voices_for_language(language)
        if not voices:
            raise HTTPException(status_code=400, detail=f"unsupported language: {language!r}")
        return TTSVoicesResponse(
            provider=provider,
            voices=[TTSVoice(id=v, name=_kokoro_name(v)) for v in voices],
            default=voice_for_language(language),
        )
    try:
        voices = elevenlabs_client.picker_order(await elevenlabs_client.account_voices())
        default = await elevenlabs_client.partner_voice_id()
    except TTSUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    offered = voices[: elevenlabs_client.MAX_PICKER_VOICES]
    if default not in {v["voice_id"] for v in offered}:
        default_voice = next((v for v in voices if v["voice_id"] == default), None)
        offered = ([default_voice] if default_voice else []) + offered[: elevenlabs_client.MAX_PICKER_VOICES - 1]
    return TTSVoicesResponse(
        provider=provider,
        voices=[TTSVoice(id=v["voice_id"], name=v.get("name") or v["voice_id"]) for v in offered],
        default=default,
    )


@router.post("/speak")
async def speak(req: TTSRequest) -> Response:
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="text must not be empty")
    if req.role not in ("partner", "coach"):
        raise HTTPException(status_code=400, detail=f"unknown role: {req.role!r}")
    provider = _provider(req.provider)

    try:
        if provider == "kokoro":
            # One language per Kokoro voice: the coach speaks in the request
            # language's default voice; the partner's voice must match it.
            voice = None if req.role == "coach" else req.voice
            if voice is not None and voice not in voices_for_language(req.language):
                raise HTTPException(
                    status_code=400,
                    detail=f"{voice!r} is not a valid voice for language {req.language!r}",
                )
            audio = await deepinfra_client.synthesize(req.text, language=req.language, voice=voice)
        else:
            if req.role == "coach":
                voice_id = await elevenlabs_client.coach_voice_id()
            else:
                voice_id = req.voice or await elevenlabs_client.partner_voice_id()
            audio = await elevenlabs_client.synthesize(req.text, voice_id=voice_id)
    except TTSUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return Response(content=audio, media_type="audio/mpeg")
