import asyncio
import json
from unittest.mock import patch

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.chat.openai_client import ModelServerUnavailableError
from app.db import Base
from app.routes.chat import take_turn_stream
from app.schemas import ChatTurnRequest


def _session():
    # check_same_thread=False, matching app/db.py's real engine: unlike
    # test_coach_route.py's mocks (which never touch the DB), this route's
    # generator actually persists via _gloss_and_persist_reply - and
    # Starlette's StreamingResponse runs a plain (sync) generator's steps
    # in a worker thread (iterate_in_threadpool), not the calling thread,
    # so the session's SQLite connection gets used from a different thread
    # than the one that created it once body_iterator is consumed below.
    # StaticPool on top of that: a plain :memory: DB is otherwise per-
    # CONNECTION, so a second thread opening its own connection would see
    # a blank, separate database (no tables at all) rather than the one
    # Base.metadata.create_all just set up here - StaticPool keeps every
    # connection on this engine pointed at the exact same one.
    engine = create_engine(
        "sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    from app import models  # noqa: F401  (register tables on Base)

    Base.metadata.create_all(bind=engine)
    return sessionmaker(bind=engine)()


async def _collect(body_iterator):
    chunks = []
    async for chunk in body_iterator:
        chunks.append(chunk)
    return chunks


def _run(session, deltas):
    # Mirrors test_coach_route.py's pattern: the route only builds the
    # StreamingResponse - model_chat_stream isn't actually called until
    # body_iterator is consumed below, so the patch has to stay active for
    # that too.
    with (
        patch("app.routes.chat.model_chat_stream", return_value=iter(deltas)),
        patch("app.translate.llm_translate.model_chat", side_effect=ModelServerUnavailableError("down")),
    ):
        response = take_turn_stream(ChatTurnRequest(session_id="s1", message="hola"), session=session)
        raw = "".join(asyncio.run(_collect(response.body_iterator)))
    parsed = []
    for block in raw.split("\n\n"):
        if not block.strip():
            continue
        lines = block.splitlines()
        name = next(line[len("event:") :].strip() for line in lines if line.startswith("event:"))
        data = next(line[len("data:") :].strip() for line in lines if line.startswith("data:"))
        parsed.append((name, json.loads(data)))
    return response, parsed


def test_chunks_then_done_carries_the_full_reply():
    response, parsed = _run(_session(), ["Hola", ", ", "¿qué tal?"])

    assert response.media_type == "text/event-stream"
    assert [name for name, _ in parsed[:3]] == ["chunk", "chunk", "chunk"]
    assert [d["delta"] for _, d in parsed[:3]] == ["Hola", ", ", "¿qué tal?"]
    assert parsed[-1][0] == "done"
    assert parsed[-1][1]["text"] == "Hola, ¿qué tal?"


def test_model_unavailable_yields_an_error_event_with_no_chunks():
    with patch("app.routes.chat.model_chat_stream", side_effect=ModelServerUnavailableError("down")):
        response = take_turn_stream(ChatTurnRequest(session_id="s1", message="hola"), session=_session())
        raw = "".join(asyncio.run(_collect(response.body_iterator)))
    assert "event: error" in raw
    assert "down" in raw
    assert "event: chunk" not in raw
    assert "event: done" not in raw


def test_zero_chunks_yields_an_error_not_a_blank_done():
    # The actual live bug this guards: a reasoning-style model can burn its
    # whole generation budget on invisible reasoning tokens and return with
    # NO visible content at all - not an exception, just an empty stream.
    # Previously this silently "succeeded" with a "done" event carrying an
    # empty `text`, persisting a vacant chat bubble with no indication
    # anything went wrong.
    session = _session()
    response, parsed = _run(session, [])

    assert [name for name, _ in parsed] == ["error"]
    assert parsed[0][1]["message"] == "Model returned an empty reply"

    # Nothing should have been persisted - not even the user's own turn,
    # which was staged (session.add) before the stream ran but never
    # committed once the reply came back empty. Rolling back first mirrors
    # what the real request lifecycle does (get_session's dependency closes
    # the session without ever committing on this path) - querying the
    # SAME still-open session without it would see the flushed-but-
    # uncommitted insert via read-your-own-writes, which isn't what ends
    # up durably persisted once the session actually closes.
    from app.models import ChatMessage

    session.rollback()
    assert session.query(ChatMessage).count() == 0


def test_whitespace_only_reply_also_counts_as_empty():
    # The individual deltas still stream live as "chunk" events as they
    # arrive (the whole point of streaming) - only the FINAL decision
    # (done vs. error) depends on whether they add up to anything once
    # stripped, which happens after every delta is already in hand.
    response, parsed = _run(_session(), ["   ", "\n"])

    assert [name for name, _ in parsed] == ["chunk", "chunk", "error"]
    assert parsed[-1][1]["message"] == "Model returned an empty reply"
