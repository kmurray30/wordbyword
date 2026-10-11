"""Voice mode: live microphone streaming and the spoken-conversation coach.

The microphone path is a WebSocket rather than request/response because
turn-taking has to come from the audio itself (server-side endpointing on
the transcription provider) - with the phone's screen off there is no
"stop" button to tap.
"""

import asyncio
import json
import logging
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect
from sqlalchemy.orm import Session

from app.db import get_session
from app.stt import deepgram_client
from app.stt.deepgram_client import STTUnavailableError
from app.wordbank import store

logger = logging.getLogger(__name__)

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


def keyterms_for(session: Session, limit: int = deepgram_client.MAX_KEYTERMS) -> list[str]:
    """The learner's most recently seen word-bank lemmas - fed to Deepgram's
    keyterm prompting so the exact words they're studying are the ones the
    recognizer is biased toward hearing."""
    entries = sorted(store.list_all(session), key=lambda e: e.last_seen_at, reverse=True)
    return [e.lemma for e in entries if e.lemma.isalpha()][:limit]


@router.websocket("/listen")
async def listen(websocket: WebSocket, session: Session = Depends(get_session)) -> None:
    """Proxies the browser's microphone to Deepgram and relays transcripts
    back (keeps the Deepgram key server-side).

    Client -> server: binary frames are 16kHz mono 16-bit PCM audio; JSON
    text frames are control messages - {"type":"start"} (must come first),
    {"type":"mute"} / {"type":"unmute"} (frames are dropped while muted -
    the app's own TTS is playing and the mic would just hear it; dropping
    server-side is cheaper than reconnecting to Deepgram every turn), and
    {"type":"stop"} (flush and finish).

    Server -> client: JSON {"type": "interim"|"final"|"utterance_end"|
    "speech_started", "text", "words"} (see deepgram_client.TranscriptEvent)
    and {"type":"error","message"}."""
    await websocket.accept()

    try:
        first = json.loads(await websocket.receive_text())
    except WebSocketDisconnect:
        return
    except (json.JSONDecodeError, TypeError):
        await websocket.send_json({"type": "error", "message": "expected a {\"type\":\"start\"} control frame first"})
        await websocket.close()
        return
    if first.get("type") != "start":
        await websocket.send_json({"type": "error", "message": "expected a {\"type\":\"start\"} control frame first"})
        await websocket.close()
        return

    frames: asyncio.Queue[bytes | None] = asyncio.Queue()
    muted = False

    async def read_client() -> None:
        nonlocal muted
        try:
            while True:
                message = await websocket.receive()
                if message.get("type") == "websocket.disconnect":
                    break
                if message.get("bytes") is not None:
                    if not muted:
                        await frames.put(message["bytes"])
                    continue
                text = message.get("text")
                if not text:
                    continue
                try:
                    control = json.loads(text)
                except json.JSONDecodeError:
                    continue
                kind = control.get("type")
                if kind == "mute":
                    muted = True
                elif kind == "unmute":
                    muted = False
                elif kind == "stop":
                    break
        except WebSocketDisconnect:
            pass
        finally:
            await frames.put(None)

    async def frame_iter() -> AsyncIterator[bytes | None]:
        while True:
            frame = await frames.get()
            yield frame
            if frame is None:
                return

    reader = asyncio.create_task(read_client())
    try:
        async for event in deepgram_client.stream_transcripts(frame_iter(), keyterms_for(session)):
            await websocket.send_json({"type": event.kind, "text": event.text, "words": event.words})
    except STTUnavailableError as exc:
        await _send_error(websocket, str(exc))
    except WebSocketDisconnect:
        pass
    except Exception as exc:  # a provider hiccup must not take the whole server down
        logger.exception("voice/listen failed")
        await _send_error(websocket, f"transcription failed: {exc!r}")
    finally:
        reader.cancel()
        try:
            await reader
        except (asyncio.CancelledError, Exception):
            pass
        try:
            await websocket.close()
        except Exception:
            pass


async def _send_error(websocket: WebSocket, message: str) -> None:
    try:
        await websocket.send_json({"type": "error", "message": message})
    except Exception:
        pass
