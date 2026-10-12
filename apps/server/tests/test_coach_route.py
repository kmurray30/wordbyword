import asyncio
import json
from unittest.mock import patch

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.db import Base
from app.models import ChatMessage, ThreadEntry
from app.routes.chat import get_history
from app.routes.translate import coach_draft
from app.schemas import CoachDraftRequest
from app.translate.llm_translate import TranslationUnavailableError


def _session():
    # The route persists from Starlette's worker thread - StaticPool keeps
    # every connection on the same in-memory DB (see test_chat_stream_route).
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    from app import models  # noqa: F401

    Base.metadata.create_all(bind=engine)
    return sessionmaker(bind=engine)()


def _seed(session):
    rows = [
        ChatMessage(session_id="s1", role="user", text="Hola"),
        ChatMessage(session_id="s1", role="assistant", text="¿Cómo te llamas?"),
        ChatMessage(session_id="other", role="assistant", text="not mine"),
    ]
    session.add_all(rows)
    session.commit()
    return rows


async def _collect(body_iterator):
    return [chunk async for chunk in body_iterator]


def _run(session, req, events):
    with patch("app.routes.translate.llm_translate.coach_reply_stream", return_value=iter(events)) as stream:
        response = coach_draft(req, session=session)
        raw = "".join(asyncio.run(_collect(response.body_iterator)))
    parsed = []
    for block in raw.split("\n\n"):
        lines = [line for line in block.splitlines() if line.startswith(("event:", "data:"))]
        if len(lines) == 2:
            parsed.append((lines[0][6:].strip(), json.loads(lines[1][5:].strip())))
    return response, parsed, stream


def test_reply_about_the_next_message_streams_and_is_not_stored():
    session = _session()
    _, partner, _ = _seed(session)
    response, parsed, stream = _run(
        session,
        CoachDraftRequest(
            text="me gusta swimming",
            session_id="s1",
            draft_thread=[{"role": "learner", "text": "I want to say I like swimming"}],
        ),
        [
            ("about", None),
            ("ready", False),
            ("reply_chunk", "Try "),
            ("reply_chunk", "[[Me gusta nadar. | I like swimming.]]"),
            ("reply_complete", "Try [[Me gusta nadar. | I like swimming.]]"),
        ],
    )
    assert response.media_type == "text/event-stream"
    assert [n for n, _ in parsed] == ["about", "ready", "reply_chunk", "reply_chunk", "reply_complete"]
    assert parsed[0][1] == {"message_id": None}
    assert parsed[1][1] == {"ready": False}
    assert parsed[-1][1]["entries"] == []
    assert session.get(ChatMessage, partner.id).thread == []
    args = stream.call_args
    assert [m["text"] for m in args.args[1]] == ["Hola", "¿Cómo te llamas?"]
    assert args.args[4] == [("learner", "I want to say I like swimming")]
    assert args.kwargs["check"] is False


def test_reply_about_a_partner_message_is_stored_in_its_thread():
    session = _session()
    _, partner, _ = _seed(session)
    _, parsed, _ = _run(
        session,
        CoachDraftRequest(text="what does llamas mean?", session_id="s1"),
        [("about", 2), ("ready", False), ("reply_complete", "It means you call yourself.")],
    )
    assert parsed[0][1] == {"message_id": partner.id}
    assert [e["kind"] for e in parsed[-1][1]["entries"]] == ["question", "answer"]
    thread = next(m for m in get_history("s1", session=session).messages if m.id == partner.id).thread
    assert [(e.role, e.kind, e.text) for e in thread] == [
        ("learner", "question", "what does llamas mean?"),
        ("coach", "answer", "It means you call yourself."),
    ]


def test_open_thread_is_the_default_and_its_turns_are_context():
    session = _session()
    _, partner, _ = _seed(session)
    session.add(ThreadEntry(message_id=partner.id, position=0, role="learner", kind="question", text="q1"))
    session.add(ThreadEntry(message_id=partner.id, position=1, role="coach", kind="answer", text="a1"))
    session.commit()
    _, parsed, stream = _run(
        session,
        CoachDraftRequest(text="llamas?", session_id="s1", thread_message_id=partner.id),
        [("about", 2), ("ready", False), ("reply_complete", "Yes, llamas.")],
    )
    assert stream.call_args.args[2] == 2
    assert stream.call_args.args[3] == [("learner", "q1"), ("coach", "a1")]
    assert [e.position for e in session.get(ChatMessage, partner.id).thread] == [0, 1, 2, 3]


def test_a_thread_id_from_another_session_is_ignored():
    session = _session()
    _, _, other = _seed(session)
    _, _, stream = _run(
        session,
        CoachDraftRequest(text="hola", session_id="s1", thread_message_id=other.id),
        [("about", None), ("ready", True), ("reply_complete", "Good.")],
    )
    assert stream.call_args.args[2] is None
    assert all(m["text"] != "not mine" for m in stream.call_args.args[1])


def test_a_check_is_always_about_the_next_message():
    session = _session()
    _, partner, _ = _seed(session)
    _, parsed, stream = _run(
        session,
        CoachDraftRequest(text="Me llamo Kyle.", session_id="s1", check=True),
        [("about", 2), ("ready", True), ("reply_complete", "Perfect.")],
    )
    assert stream.call_args.kwargs["check"] is True
    assert parsed[0][1] == {"message_id": None}
    assert parsed[1][1] == {"ready": True}
    assert session.get(ChatMessage, partner.id).thread == []


def test_coach_failure_is_an_error_event_and_nothing_is_stored():
    session = _session()
    _, partner, _ = _seed(session)

    def failing():
        yield ("about", 2)
        raise TranslationUnavailableError("down")

    with patch("app.routes.translate.llm_translate.coach_reply_stream", return_value=failing()):
        response = coach_draft(CoachDraftRequest(text="why?", session_id="s1"), session=session)
        raw = "".join(asyncio.run(_collect(response.body_iterator)))
    assert "event: error" in raw
    assert "reply_complete" not in raw
    assert session.get(ChatMessage, partner.id).thread == []
