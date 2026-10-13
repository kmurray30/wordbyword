from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

from app.schemas import TTSRequest, TTSVoice, TTSVoicesResponse
from app.tts import elevenlabs_client
from app.tts.errors import TTSUnavailableError

router = APIRouter(prefix="/tts", tags=["tts"])


@router.get("/voices", response_model=TTSVoicesResponse)
async def list_voices(language: str = "es") -> TTSVoicesResponse:
    """The partner voices the app offers (the account's ElevenLabs voices,
    see picker_order), and which one it uses by default. `language` is
    accepted for older clients and ignored - every voice is multilingual."""
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
        voices=[TTSVoice(id=v["voice_id"], name=v.get("name") or v["voice_id"]) for v in offered],
        default=default,
    )


@router.post("/speak")
async def speak(req: TTSRequest) -> Response:
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="text must not be empty")
    if req.role not in ("partner", "coach"):
        raise HTTPException(status_code=400, detail=f"unknown role: {req.role!r}")

    try:
        if req.role == "coach":
            voice_id = await elevenlabs_client.coach_voice_id()
        else:
            voice_id = req.voice or await elevenlabs_client.partner_voice_id()
        audio = await elevenlabs_client.synthesize(req.text, voice_id=voice_id)
    except TTSUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return Response(content=audio, media_type="audio/mpeg")
