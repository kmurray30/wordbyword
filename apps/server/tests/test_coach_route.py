import asyncio
import json
from unittest.mock import patch

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db import Base
from app.routes.translate import coach_draft
from app.schemas import CoachDraftRequest
from app.translate.llm_translate import TranslationUnavailableError


def _session():
    engine = create_engine("sqlite:///:memory:")
    from app import models  # noqa: F401  (register tables on Base)

    Base.metadata.create_all(bind=engine)
    return sessionmaker(bind=engine)()


async def _collect(body_iterator):
    chunks = []
    async for chunk in body_iterator:
        chunks.append(chunk)
    return chunks


def _run(text: str, events: list[tuple]):
    # The route function only builds the StreamingResponse - its generator
    # is lazy and isn't actually iterated (and coach_draft_stream isn't
    # actually called) until body_iterator is consumed below, so the
    # patch has to stay active for that too, not just for the route call.
    with patch("app.routes.translate.llm_translate.coach_draft_stream", return_value=iter(events)):
        response = coach_draft(CoachDraftRequest(text=text, session_id="s1"), session=_session())
        raw = "".join(asyncio.run(_collect(response.body_iterator)))
    # Parse the SSE wire format ("event: <name>\ndata: <json>\n\n") back
    # into (name, parsed_json) pairs.
    parsed = []
    for block in raw.split("\n\n"):
        if not block.strip():
            continue
        lines = block.splitlines()
        name = next(line[len("event:") :].strip() for line in lines if line.startswith("event:"))
        data = next(line[len("data:") :].strip() for line in lines if line.startswith("data:"))
        parsed.append((name, json.loads(data)))
    return response, parsed


def test_clean_verdict_sse_sequence():
    events = [
        ("verdict", "clean"),
        ("meaning_chunk", "Hello "),
        ("meaning_chunk", "there"),
        ("meaning_complete", "Hello there"),
    ]
    response, parsed = _run("hola", events)

    assert response.media_type == "text/event-stream"
    assert [name for name, _ in parsed] == ["verdict", "meaning_chunk", "meaning_chunk", "meaning_complete"]
    assert parsed[0][1] == {"verdict": "clean"}
    assert parsed[1][1] == {"delta": "Hello "}
    assert parsed[3][1] == {"meaning": "Hello there"}


def test_minor_verdict_sse_sequence():
    events = [
        ("verdict", "minor"),
        ("meaning_chunk", "You mean hello"),
        ("meaning_complete", "You mean hello"),
        ("suggestion_chunk", "Hola, "),
        ("suggestion_chunk", "¿cómo estás?"),
        ("suggestion_complete", "Hola, ¿cómo estás?"),
    ]
    response, parsed = _run("hola como estas", events)

    assert [name for name, _ in parsed] == [
        "verdict",
        "meaning_chunk",
        "meaning_complete",
        "suggestion_chunk",
        "suggestion_chunk",
        "suggestion_complete",
    ]
    assert parsed[0][1] == {"verdict": "minor"}
    assert parsed[-1][1] == {"suggestion": "Hola, ¿cómo estás?"}


def test_fix_verdict_sse_sequence_is_interleaved_per_option():
    events = [
        ("verdict", "fix"),
        ("meaning_chunk", "I want to go to the beach"),
        ("meaning_complete", "I want to go to the beach"),
        ("option_spanish_chunk", 0, "Quiero ir a la playa."),
        ("option_spanish_complete", 0, "Quiero ir a la playa."),
        ("option_english_chunk", 0, "I want to go to the beach."),
        ("option_english_complete", 0, "I want to go to the beach."),
        ("option_spanish_chunk", 1, "Quiero ir a la playa, ¿va?"),
        ("option_spanish_complete", 1, "Quiero ir a la playa, ¿va?"),
        ("option_english_chunk", 1, "I want to go to the beach, right?"),
        ("option_english_complete", 1, "I want to go to the beach, right?"),
    ]
    response, parsed = _run("quiero ir playa", events)

    names = [name for name, _ in parsed]
    assert names == [
        "verdict",
        "meaning_chunk",
        "meaning_complete",
        "option_chunk",
        "option_complete",
        "option_chunk",
        "option_complete",
        "option_chunk",
        "option_complete",
        "option_chunk",
        "option_complete",
    ]
    # Interleaved order confirmed: option 0's spanish+english complete
    # before option 1's spanish even starts, not all-spanish-then-all-
    # english.
    option_complete_events = [data for name, data in parsed if name == "option_complete"]
    assert option_complete_events == [
        {"index": 0, "field": "spanish", "text": "Quiero ir a la playa."},
        {"index": 0, "field": "english", "text": "I want to go to the beach."},
        {"index": 1, "field": "spanish", "text": "Quiero ir a la playa, ¿va?"},
        {"index": 1, "field": "english", "text": "I want to go to the beach, right?"},
    ]


def test_coach_failure_yields_an_error_event():
    with patch(
        "app.routes.translate.llm_translate.coach_draft_stream",
        side_effect=TranslationUnavailableError("model is down"),
    ):
        response = coach_draft(CoachDraftRequest(text="hola", session_id="s1"), session=_session())
        raw = "".join(asyncio.run(_collect(response.body_iterator)))
    assert "event: error" in raw
    assert "model is down" in raw


# --- the "ask" branch: questions for the tutor ---------------------------------

from app.models import ChatMessage, ThreadEntry  # noqa: E402
from app.routes.chat import get_history  # noqa: E402
from sqlalchemy.pool import StaticPool  # noqa: E402


def _shared_session():
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


def _run_request(session, req, coach_events, tutor_events):
    with (
        patch("app.routes.translate.llm_translate.coach_draft_stream", return_value=iter(coach_events)) as coach,
        patch("app.routes.translate.llm_translate.tutor_answer_stream", return_value=iter(tutor_events)) as tutor,
    ):
        response = coach_draft(req, session=session)
        raw = "".join(asyncio.run(_collect(response.body_iterator)))
    parsed = []
    for block in raw.split("\n\n"):
        lines = [line for line in block.splitlines() if line.startswith(("event:", "data:"))]
        if len(lines) == 2:
            parsed.append((lines[0][6:].strip(), json.loads(lines[1][5:].strip())))
    return parsed, coach, tutor


def test_ask_verdict_hands_off_to_the_tutor_and_persists_the_exchange():
    session = _shared_session()
    user_row, partner_row, _ = _seed(session)
    parsed, coach, tutor = _run_request(
        session,
        CoachDraftRequest(text="what does llamas mean?", session_id="s1"),
        [("verdict", "ask")],
        [("about", 2), ("answer_chunk", "It means "), ("answer_chunk", "you call yourself."), ("answer_complete", "It means you call yourself.")],
    )
    names = [n for n, _ in parsed]
    assert names == ["verdict", "thread_anchor", "answer_chunk", "answer_chunk", "answer_complete"]
    assert parsed[0][1] == {"verdict": "ask"}
    assert parsed[1][1] == {"message_id": partner_row.id, "question": "what does llamas mean?"}
    assert [e["kind"] for e in parsed[-1][1]["entries"]] == ["question", "answer"]
    # The tutor got the numbered conversation, defaulting to the partner's message.
    args = tutor.call_args.args
    assert [m["text"] for m in args[1]] == ["Hola", "¿Cómo te llamas?"]
    assert args[2] == 2
    assert coach.call_args.kwargs["allow_ask"] is True
    # Stored on the partner's message and returned by /chat/history.
    history = get_history("s1", session=session)
    thread = next(m for m in history.messages if m.id == partner_row.id).thread
    assert [(e.kind, e.text) for e in thread] == [
        ("question", "what does llamas mean?"),
        ("answer", "It means you call yourself."),
    ]


def test_open_thread_is_passed_to_the_coach_and_follow_ups_continue_it():
    session = _shared_session()
    user_row, partner_row, other_row = _seed(session)
    session.add(ThreadEntry(message_id=partner_row.id, position=0, role="learner", kind="question", text="q1"))
    session.add(ThreadEntry(message_id=partner_row.id, position=1, role="tutor", kind="answer", text="a1"))
    session.commit()
    parsed, coach, tutor = _run_request(
        session,
        CoachDraftRequest(text="llamas?", session_id="s1", thread_message_id=partner_row.id),
        [("verdict", "ask")],
        [("about", 2), ("answer_complete", "Yes, llamas.")],
    )
    assert coach.call_args.kwargs["open_thread"] == ("¿Cómo te llamas?", [("learner", "q1"), ("tutor", "a1")])
    assert tutor.call_args.args[3] == [("learner", "q1"), ("tutor", "a1")]
    assert parsed[1][1]["message_id"] == partner_row.id
    positions = [e.position for e in session.get(ChatMessage, partner_row.id).thread]
    assert positions == [0, 1, 2, 3]


def test_a_thread_id_from_another_session_is_ignored():
    session = _shared_session()
    _, partner_row, other_row = _seed(session)
    _, coach, _ = _run_request(
        session,
        CoachDraftRequest(text="hola", session_id="s1", thread_message_id=other_row.id),
        [("verdict", "clean"), ("meaning_complete", "Hi")],
        [],
    )
    assert coach.call_args.kwargs["open_thread"] is None


def test_force_ask_skips_the_coach_and_force_attempt_disables_ask():
    session = _shared_session()
    _seed(session)
    parsed, coach, tutor = _run_request(
        session,
        CoachDraftRequest(text="hola", session_id="s1", force="ask"),
        [],
        [("about", 2), ("answer_complete", "It's hello.")],
    )
    coach.assert_not_called()
    assert parsed[0] == ("verdict", {"verdict": "ask"})
    _, coach, tutor = _run_request(
        session,
        CoachDraftRequest(text="what does hola mean", session_id="s1", force="attempt"),
        [("verdict", "fix")],
        [],
    )
    assert coach.call_args.kwargs["allow_ask"] is False
    tutor.assert_not_called()


def test_tutor_failure_is_an_error_event_and_nothing_is_persisted():
    session = _shared_session()
    _, partner_row, _ = _seed(session)

    def failing():
        yield ("about", 2)
        raise TranslationUnavailableError("down")

    with (
        patch("app.routes.translate.llm_translate.coach_draft_stream", return_value=iter([("verdict", "ask")])),
        patch("app.routes.translate.llm_translate.tutor_answer_stream", return_value=failing()),
    ):
        response = coach_draft(CoachDraftRequest(text="why?", session_id="s1"), session=session)
        raw = "".join(asyncio.run(_collect(response.body_iterator)))
    assert "event: error" in raw
    assert session.get(ChatMessage, partner_row.id).thread == []



def test_a_question_about_the_next_message_is_not_stored_on_any_message():
    session = _shared_session()
    _, partner_row, _ = _seed(session)
    parsed, coach, tutor = _run_request(
        session,
        CoachDraftRequest(
            text="how do I say I'm sleepy?",
            session_id="s1",
            draft_thread=[{"role": "learner", "text": "I want to say I'm tired"}],
        ),
        [("verdict", "ask")],
        [("about", None), ("answer_complete", "Say tengo sueño.")],
    )
    assert parsed[1][1]["message_id"] is None
    assert parsed[-1][1]["entries"] == []
    assert session.get(ChatMessage, partner_row.id).thread == []
    assert coach.call_args.kwargs["draft_turns"] == [("learner", "I want to say I'm tired")]
    assert tutor.call_args.kwargs["draft_turns"] == [("learner", "I want to say I'm tired")]
