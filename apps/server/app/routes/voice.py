"""Voice mode: live microphone streaming and the spoken-conversation coach.

The microphone path is a WebSocket rather than request/response because
turn-taking has to come from the audio itself (server-side endpointing on
the transcription provider) - with the phone's screen off there is no
"stop" button to tap.
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

router = APIRouter(prefix="/voice", tags=["voice"])


@router.websocket("/ping")
async def ping(websocket: WebSocket) -> None:
    """Echo socket for the on-device keep-alive spike
    (apps/web/public/voice-spike.html): proves whether a WebSocket from a
    web page survives the phone being locked, independent of any speech
    provider."""
    await websocket.accept()
    try:
        while True:
            message = await websocket.receive_text()
            await websocket.send_text(f"pong {message}")
    except WebSocketDisconnect:
        return
