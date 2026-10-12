"""Streaming speech-to-text via Deepgram's live /v1/listen WebSocket.

A separate provider from chat (OpenAI) and TTS (DeepInfra), with its own key
and its own failure mode, same as the other hosted clients in this app.

Why streaming rather than record-then-upload: turn-taking has to come from
the audio itself. Deepgram's server-side endpointing / UtteranceEnd events
are what let the learner just stop talking and have the app react, with no
"stop" button - which is the whole point of voice mode working with the
phone's screen off.

Why `language=multi`: the learner's speech is genuinely mixed Spanish and
English inside one utterance ("¿te gusta running?"). Nova-3's multilingual
code-switching mode transcribes both in one pass; a single-language model
would mangle whichever half it wasn't told about.
"""

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from urllib.parse import urlencode

import websockets

from app.config import DEEPGRAM_API_KEY, DEEPGRAM_STT_MODEL

logger = logging.getLogger(__name__)

DEEPGRAM_LISTEN_URL = "wss://api.deepgram.com/v1/listen"

# The browser's AudioWorklet (apps/web/public/pcm-worklet.js) downsamples to
# exactly this: 16kHz mono 16-bit little-endian PCM. Deepgram has to be told
# the raw format since there's no container header.
SAMPLE_RATE = 16000

# Deepgram drops a live connection that goes ~10s without audio. While the
# client is muted (our own TTS is playing - see /voice/listen's half-duplex
# muting) no audio arrives, so the sender pings a KeepAlive well inside
# that window instead.
_KEEPALIVE_INTERVAL_S = 5.0

# Deepgram caps keyterm prompting at 100 terms per request; 50 leaves
# headroom and is already more vocabulary than one utterance could use.
MAX_KEYTERMS = 50


class STTUnavailableError(RuntimeError):
    pass


@dataclass
class TranscriptEvent:
    # "interim": a provisional transcript of the segment still being spoken
    #   (replaces the previous interim, never appended).
    # "final": Deepgram's settled transcript for a segment (is_final) -
    #   these DO accumulate across one utterance.
    # "utterance_end": silence long enough (utterance_end_ms) to call the
    #   turn over - the caller's cue to act on everything "final" so far.
    # "speech_started": the learner began talking (vad_events).
    kind: str
    text: str = ""
    # Per-word {"word", "confidence"} - the "sounded like X, did you mean
    # Y?" signal the voice coach reads (low-confidence near-homophones).
    words: list[dict] = field(default_factory=list)


def build_listen_url(keyterms: list[str] = ()) -> str:
    params = [
        ("model", DEEPGRAM_STT_MODEL),
        ("language", "multi"),
        ("encoding", "linear16"),
        ("sample_rate", str(SAMPLE_RATE)),
        ("channels", "1"),
        ("interim_results", "true"),
        ("smart_format", "true"),
        # endpointing is Deepgram's own "speech_final" silence threshold;
        # utterance_end_ms is the more conservative end-of-TURN signal this
        # app actually acts on - a learner pausing to think mid-sentence
        # shouldn't get cut off at a few hundred ms of silence.
        ("endpointing", "300"),
        ("utterance_end_ms", "1200"),
        ("vad_events", "true"),
    ]
    params.extend(("keyterm", term) for term in list(keyterms)[:MAX_KEYTERMS])
    return f"{DEEPGRAM_LISTEN_URL}?{urlencode(params)}"


def normalize_message(raw: dict) -> TranscriptEvent | None:
    """Deepgram's live message shapes -> this app's TranscriptEvent, or None
    for the ones nothing here acts on (Metadata, empty interims). Pure, so
    the wire-format handling is unit-testable without a socket."""
    kind = raw.get("type")
    if kind == "UtteranceEnd":
        return TranscriptEvent(kind="utterance_end")
    if kind == "SpeechStarted":
        return TranscriptEvent(kind="speech_started")
    if kind != "Results":
        return None

    alternatives = (raw.get("channel") or {}).get("alternatives") or []
    if not alternatives:
        return None
    best = alternatives[0]
    text = (best.get("transcript") or "").strip()
    if not text:
        # Deepgram emits empty Results frames for pure silence - noise.
        return None
    words = [
        {
            # punctuated_word carries smart_format's casing/punctuation; the
            # bare `word` is the fallback for providers/options without it.
            "word": w.get("punctuated_word") or w.get("word") or "",
            "confidence": float(w.get("confidence") or 0.0),
        }
        for w in best.get("words") or []
    ]
    return TranscriptEvent(kind="final" if raw.get("is_final") else "interim", text=text, words=words)


async def stream_transcripts(
    audio_frames: AsyncIterator[bytes | None],
    keyterms: list[str] = (),
) -> AsyncIterator[TranscriptEvent]:
    """Pipes `audio_frames` (raw PCM chunks; a `None` item means "the caller
    is done, flush and finish") into Deepgram and yields TranscriptEvents
    as they come back. Raises STTUnavailableError if the connection can't
    be established at all; a connection that drops mid-stream just ends
    the iterator (whatever was transcribed is already yielded)."""
    if not DEEPGRAM_API_KEY:
        raise STTUnavailableError("DEEPGRAM_API_KEY is not configured on the server")

    url = build_listen_url(keyterms)
    headers = {"Authorization": f"Token {DEEPGRAM_API_KEY}"}
    try:
        ws = await websockets.connect(url, additional_headers=headers, open_timeout=10)
    except (OSError, websockets.exceptions.WebSocketException, asyncio.TimeoutError) as exc:
        raise STTUnavailableError(f"Could not connect to Deepgram: {exc!r}") from exc

    async def send_audio() -> None:
        # Plain iteration, deliberately NOT asyncio.wait_for() around the
        # iterator's __anext__ to detect gaps: the timeout cancels that
        # __anext__, which leaves an async generator unusable ("already
        # running" on the next call) - observed live: the first muted gap
        # killed this task, no KeepAlive ever went out, and Deepgram closed
        # the connection 10s later ("Voice connection closed" in the UI).
        # Gaps are covered by the separate keepalive task below instead.
        try:
            async for frame in audio_frames:
                if frame is None:
                    break
                await ws.send(frame)
            closing.set()
            await ws.send(json.dumps({"type": "CloseStream"}))
        except websockets.exceptions.ConnectionClosed:
            return

    async def keepalive() -> None:
        # Deepgram ignores a KeepAlive that arrives while audio is flowing,
        # so sending one on a fixed cadence is simpler and safer than
        # trying to send it only during gaps.
        try:
            while not closing.is_set():
                await asyncio.sleep(_KEEPALIVE_INTERVAL_S)
                if closing.is_set():
                    return
                await ws.send(json.dumps({"type": "KeepAlive"}))
        except websockets.exceptions.ConnectionClosed:
            return

    closing = asyncio.Event()

    sender = asyncio.create_task(send_audio())
    pinger = asyncio.create_task(keepalive())
    try:
        async for message in ws:
            if isinstance(message, bytes):
                continue
            try:
                raw = json.loads(message)
            except json.JSONDecodeError:
                logger.warning("Deepgram sent non-JSON text frame: %r", message[:200])
                continue
            if raw.get("type") == "Error":
                logger.warning("Deepgram error frame: %s", raw)
                continue
            event = normalize_message(raw)
            if event is not None:
                yield event
    except websockets.exceptions.ConnectionClosed as exc:
        logger.info("Deepgram connection closed: %r", exc)
    finally:
        for task in (sender, pinger):
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
        await ws.close()
