from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

from app.schemas import TTSRequest
from app.tts.deepinfra_client import TTSUnavailableError, synthesize

router = APIRouter(prefix="/tts", tags=["tts"])


@router.post("/speak")
def speak(req: TTSRequest) -> Response:
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="text must not be empty")

    try:
        audio = synthesize(req.text)
    except TTSUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return Response(content=audio, media_type="audio/mpeg")
