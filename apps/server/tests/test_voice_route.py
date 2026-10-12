import json
from datetime import datetime, timedelta
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.db import Base, get_session
from app.routes import voice
from app.stt.deepgram_client import STTUnavailableError, TranscriptEvent
from app.wordbank import store


def _session():
    # StaticPool + check_same_thread=False: an in-memory SQLite DB lives on
    # ONE connection, and TestClient runs the app on a separate thread from
    # the test - without pinning every thread to that same connection, the
    # route would see a fresh, empty database with no tables.
    engine = create_engine(
        "sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    from app import models  # noqa: F401  (register tables on Base)

    Base.metadata.create_all(bind=engine)
    return sessionmaker(bind=engine)()


def _client(session=None) -> TestClient:
    # A bare app with just this router - app.main's startup hook would
    # otherwise create the real on-disk SQLite file as a side effect.
    app = FastAPI()
    app.include_router(voice.router)
    db = session or _session()
    app.dependency_overrides[get_session] = lambda: db
    return TestClient(app)


def test_ping_echoes_each_message():
    with _client().websocket_connect("/voice/ping") as ws:
        ws.send_text("ping 1")
        assert ws.receive_text() == "pong ping 1"
        ws.send_text("ping 2")
        assert ws.receive_text() == "pong ping 2"


def test_keyterms_are_most_recent_alphabetic_lemmas():
    session = _session()
    old = store.get_or_create(session, "viejo")
    old.last_seen_at = datetime.utcnow() - timedelta(days=3)
    store.get_or_create(session, "nuevo")
    store.get_or_create(session, "x1")  # not alphabetic - never a useful keyterm
    session.commit()

    assert voice.keyterms_for(session, limit=1) == ["nuevo"]
    assert voice.keyterms_for(session) == ["nuevo", "viejo"]


def _fake_stream(received: list):
    """Stands in for deepgram_client.stream_transcripts: drains the frame
    iterator (recording what the route forwarded) and emits a canned
    interim -> final -> utterance_end sequence once audio shows up."""

    async def fake(audio_frames, keyterms=()):
        received.append(("keyterms", list(keyterms)))
        async for frame in audio_frames:
            if frame is None:
                return
            received.append(("frame", frame))
            if len([r for r in received if r[0] == "frame"]) == 1:
                yield TranscriptEvent(kind="interim", text="hola")
                yield TranscriptEvent(kind="final", text="Hola, ¿cómo estás?", words=[{"word": "Hola,", "confidence": 0.99}])
                yield TranscriptEvent(kind="utterance_end")

    return fake


def test_listen_relays_audio_and_transcript_events():
    received: list = []
    with patch("app.routes.voice.deepgram_client.stream_transcripts", _fake_stream(received)):
        with _client().websocket_connect("/voice/listen") as ws:
            ws.send_text(json.dumps({"type": "start"}))
            ws.send_bytes(b"\x00\x01" * 10)
            assert ws.receive_json() == {"type": "interim", "text": "hola", "words": []}
            final = ws.receive_json()
            assert final["type"] == "final"
            assert final["text"] == "Hola, ¿cómo estás?"
            assert final["words"] == [{"word": "Hola,", "confidence": 0.99}]
            assert ws.receive_json()["type"] == "utterance_end"
            ws.send_text(json.dumps({"type": "stop"}))

    assert ("frame", b"\x00\x01" * 10) in received


def test_listen_drops_audio_while_muted():
    received: list = []

    async def fake(audio_frames, keyterms=()):
        async for frame in audio_frames:
            if frame is None:
                return
            received.append(frame)
            yield TranscriptEvent(kind="interim", text=f"got {len(received)}")

    with patch("app.routes.voice.deepgram_client.stream_transcripts", fake):
        with _client().websocket_connect("/voice/listen") as ws:
            ws.send_text(json.dumps({"type": "start"}))
            ws.send_text(json.dumps({"type": "mute"}))
            ws.send_bytes(b"muted-audio")
            ws.send_text(json.dumps({"type": "unmute"}))
            ws.send_bytes(b"live-audio")
            assert ws.receive_json()["text"] == "got 1"
            ws.send_text(json.dumps({"type": "stop"}))

    assert received == [b"live-audio"]


def test_listen_requires_a_start_frame_first():
    with _client().websocket_connect("/voice/listen") as ws:
        ws.send_text(json.dumps({"type": "mute"}))
        error = ws.receive_json()
        assert error["type"] == "error"
        assert "start" in error["message"]


def test_listen_reports_provider_unavailable_as_an_error_event():
    async def unavailable(audio_frames, keyterms=()):
        raise STTUnavailableError("DEEPGRAM_API_KEY is not configured on the server")
        yield  # pragma: no cover

    with patch("app.routes.voice.deepgram_client.stream_transcripts", unavailable):
        with _client().websocket_connect("/voice/listen") as ws:
            ws.send_text(json.dumps({"type": "start"}))
            error = ws.receive_json()
            assert error["type"] == "error"
            assert "DEEPGRAM_API_KEY" in error["message"]


# --- POST /voice/coach ------------------------------------------------------

import asyncio  # noqa: E402

from app.models import ChatMessage  # noqa: E402
from app.schemas import VoiceCoachRequest, VoiceDialogTurn, VoiceWord  # noqa: E402
from app.translate.llm_translate import TranslationUnavailableError  # noqa: E402


async def _collect(body_iterator):
    return "".join([chunk async for chunk in body_iterator])


def _parse_sse(raw: str):
    parsed = []
    for block in raw.split("\n\n"):
        if not block.strip() or block.startswith(":"):
            continue
        lines = block.splitlines()
        name = next(line[len("event:") :].strip() for line in lines if line.startswith("event:"))
        data = next(line[len("data:") :].strip() for line in lines if line.startswith("data:"))
        parsed.append((name, json.loads(data)))
    return parsed


def test_voice_coach_sse_sequence_and_context_plumbing():
    session = _session()
    session.add(ChatMessage(session_id="s1", role="assistant", text="¿Qué te gusta hacer?"))
    session.add(ChatMessage(session_id="other", role="assistant", text="not this session"))
    session.commit()

    events = [
        ("intent", "attempt"),
        ("verdict", "revise"),
        ("draft", "¿Te gusta correr?"),
        ("say_chunk", "Did you mean: "),
        ("say_chunk", "do you like running?"),
        ("say_complete", "Did you mean: do you like running?"),
    ]
    req = VoiceCoachRequest(
        transcript="te gusta running",
        words=[VoiceWord(word="gusta", confidence=0.6)],
        coach_dialog=[VoiceDialogTurn(role="learner", text="hola")],
        session_id="s1",
    )
    with patch("app.routes.voice.llm_translate.voice_coach_stream", return_value=iter(events)) as mock_stream:
        response = voice.voice_coach(req, session=session)
        parsed = _parse_sse(asyncio.run(_collect(response.body_iterator)))

    assert response.media_type == "text/event-stream"
    assert [name for name, _ in parsed] == ["intent", "verdict", "draft", "say_chunk", "say_chunk", "say_complete"]
    assert parsed[0][1] == {"intent": "attempt"}
    assert parsed[2][1] == {"draft": "¿Te gusta correr?"}
    assert parsed[-1][1] == {"say": "Did you mean: do you like running?"}

    args, _ = mock_stream.call_args
    transcript, words, coach_dialog, history, native, target = args
    assert transcript == "te gusta running"
    assert words == [{"word": "gusta", "confidence": 0.6}]
    assert coach_dialog == [("learner", "hola")]
    assert history == [("assistant", "¿Qué te gusta hacer?")]  # only this session's rows
    assert (native, target) == ("en", "es")


def test_voice_coach_reports_unavailable_as_error_event():
    def boom(*args, **kwargs):
        raise TranslationUnavailableError("no usable intent")
        yield  # pragma: no cover

    with patch("app.routes.voice.llm_translate.voice_coach_stream", boom):
        response = voice.voice_coach(VoiceCoachRequest(transcript="hola", session_id="s1"), session=_session())
        parsed = _parse_sse(asyncio.run(_collect(response.body_iterator)))

    assert parsed == [("error", {"message": "no usable intent"})]
