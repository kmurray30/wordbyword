from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

from app.schemas import TTSRequest, TTSVoicesResponse
from app.tts.deepinfra_client import TTSUnavailableError, synthesize
from app.tts.kokoro_voices import voice_for_language, voices_for_language

router = APIRouter(prefix="/tts", tags=["tts"])


@router.get("/voices", response_model=TTSVoicesResponse)
def list_voices(language: str = "es") -> TTSVoicesResponse:
    voices = voices_for_language(language)
    if not voices:
        raise HTTPException(status_code=400, detail=f"unsupported language: {language!r}")
    return TTSVoicesResponse(voices=voices, default=voice_for_language(language))


@router.post("/speak")
def speak(req: TTSRequest) -> Response:
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="text must not be empty")

    if req.voice is not None and req.voice not in voices_for_language(req.language):
        raise HTTPException(
            status_code=400,
            detail=f"{req.voice!r} is not a valid voice for language {req.language!r}",
        )

    try:
        audio = synthesize(req.text, language=req.language, voice=req.voice)
    except TTSUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return Response(content=audio, media_type="audio/mpeg")
