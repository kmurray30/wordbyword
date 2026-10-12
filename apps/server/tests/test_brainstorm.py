"""The Brainstorm side chat (app/routes/brainstorm.py) and how it ties into
sending a reply (routes/chat.py): turns are persisted as an open log,
sending stamps the log onto the reply, and folding turns it into notes on
the right bubbles. All LLM calls are patched."""

import asyncio
import json
from unittest.mock import patch

import pytest
from fastapi import HTTPException
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.chat.openai_client import ModelServerUnavailableError
from app.db import Base
from app.models import BrainstormTurn, ChatMessage, MessageNote
from app.routes import brainstorm
from app.routes.chat import clear_history, get_history, take_turn
from app.schemas import BrainstormTurnRequest, ChatTurnRequest, FoldBrainstormRequest, VoiceWord
from app.translate.llm_translate import TranslationUnavailableError


def _session():
    # StaticPool + check_same_thread=False: the SSE body runs on a worker
    # thread, and an in-memory DB only exists on its one connection.
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    from app import models  # noqa: F401  (register tables on Base)

    Base.metadata.create_all(bind=engine)
    return sessionmaker(bind=engine)()


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


COACH_EVENTS = [
    ("intent", "attempt"),
    ("verdict", "revise"),
    ("draft", "¿Te gusta correr?"),
    ("say_chunk", "Did you mean: "),
    ("say_chunk", "do you like running?"),
    ("say_complete", "Did you mean: do you like running?"),
]


def _run_turn(session, req, events=COACH_EVENTS):
    with patch("app.routes.brainstorm.llm_translate.brainstorm_stream", return_value=iter(events)) as mock_stream:
        response = brainstorm.brainstorm_turn(req, session=session)
        parsed = _parse_sse(asyncio.run(_collect(response.body_iterator)))
    return parsed, mock_stream


# --- /brainstorm/turn --------------------------------------------------------


def test_turn_streams_coach_events_and_persists_both_sides():
    session = _session()
    session.add(ChatMessage(session_id="s1", role="assistant", text="¿Qué te gusta hacer?"))
    session.add(ChatMessage(session_id="other", role="assistant", text="not this session"))
    session.add(BrainstormTurn(session_id="s1", role="learner", text="earlier question"))
    session.add(BrainstormTurn(session_id="s1", role="learner", text="already sent", user_message_id=99))
    session.commit()

    req = BrainstormTurnRequest(
        session_id="s1",
        text="te gusta running",
        words=[VoiceWord(word="gusta", confidence=0.6)],
        reply_draft="Me gusta",
        source="voice",
    )
    parsed, mock_stream = _run_turn(session, req)

    assert [name for name, _ in parsed] == [
        "intent", "verdict", "draft", "say_chunk", "say_chunk", "say_complete", "saved",
    ]
    assert parsed[2][1] == {"draft": "¿Te gusta correr?"}

    args, kwargs = mock_stream.call_args
    text, words, dialog, history, native, target = args
    assert text == "te gusta running"
    assert words == [{"word": "gusta", "confidence": 0.6}]
    assert dialog == [("learner", "earlier question")]  # open turns only, not the new one
    assert history == [("assistant", "¿Qué te gusta hacer?")]
    assert kwargs == {"reply_draft": "Me gusta", "spoken": True}

    turns = brainstorm.open_turns(session, "s1")
    assert [(t.role, t.text, t.draft) for t in turns] == [
        ("learner", "earlier question", ""),
        ("learner", "te gusta running", ""),
        ("coach", "Did you mean: do you like running?", "¿Te gusta correr?"),
    ]
    assert parsed[-1][1] == {"learner_turn_id": turns[1].id, "coach_turn_id": turns[2].id}


def test_typed_turn_is_not_treated_as_speech():
    _, mock_stream = _run_turn(_session(), BrainstormTurnRequest(session_id="s1", text="what does 'gol' mean?"))
    assert mock_stream.call_args.kwargs["spoken"] is False


def test_turn_error_keeps_the_learner_line_and_saves_no_coach_line():
    def boom(*args, **kwargs):
        raise TranslationUnavailableError("no usable intent")
        yield  # pragma: no cover

    session = _session()
    with patch("app.routes.brainstorm.llm_translate.brainstorm_stream", boom):
        response = brainstorm.brainstorm_turn(BrainstormTurnRequest(session_id="s1", text="hola"), session=session)
        parsed = _parse_sse(asyncio.run(_collect(response.body_iterator)))

    assert parsed == [("error", {"message": "no usable intent"})]
    assert [t.role for t in brainstorm.open_turns(session, "s1")] == ["learner"]


def test_empty_turn_is_rejected():
    with pytest.raises(HTTPException) as exc_info:
        brainstorm.brainstorm_turn(BrainstormTurnRequest(session_id="s1", text="   "), session=_session())
    assert exc_info.value.status_code == 400


def test_open_log_lists_only_unsent_turns_for_the_session():
    session = _session()
    session.add_all(
        [
            BrainstormTurn(session_id="s1", role="learner", text="open"),
            BrainstormTurn(session_id="s1", role="learner", text="sent", user_message_id=1),
            BrainstormTurn(session_id="s2", role="learner", text="other session"),
        ]
    )
    session.commit()
    assert [t.text for t in brainstorm.get_open("s1", session=session).turns] == ["open"]


# --- sending + folding --------------------------------------------------------


def _send(session, message="Me gusta correr.", reply="¡Qué bien!"):
    with (
        patch("app.routes.chat.model_chat", return_value=reply),
        patch("app.translate.llm_translate.model_chat", side_effect=ModelServerUnavailableError("down")),
    ):
        return take_turn(ChatTurnRequest(session_id="s1", message=message), session=session)


def _brainstormed_session():
    """A partner message, then an open brainstorm log about the reply."""
    session = _session()
    session.add(ChatMessage(session_id="s1", role="assistant", text="¿Qué te gusta hacer?"))
    session.add_all(
        [
            BrainstormTurn(session_id="s1", role="learner", text="what does 'hacer' mean here?"),
            BrainstormTurn(session_id="s1", role="coach", text="'hacer' means 'to do'."),
            BrainstormTurn(session_id="s1", role="learner", text="how do I say I like running"),
            BrainstormTurn(session_id="s1", role="coach", text="Say: 'Me gusta correr.'", draft="Me gusta correr."),
        ]
    )
    session.commit()
    return session


def test_sending_stamps_the_open_log_onto_the_reply():
    session = _brainstormed_session()
    result = _send(session)

    assert result.has_brainstorm
    assert brainstorm.open_turns(session, "s1") == []
    stamped = session.query(BrainstormTurn).filter_by(user_message_id=result.user_message_id).count()
    assert stamped == 4

    # The next send has no open log left.
    assert not _send(session, message="Y tú?").has_brainstorm


def test_brainstorming_while_the_reply_streams_stays_open_for_the_next_reply():
    # The learner can start asking about the partner's answer while it's
    # still streaming in - before the send that's in flight gets stamped.
    session = _brainstormed_session()

    def reply_then_brainstorm(messages):
        session.add(BrainstormTurn(session_id="s1", role="learner", text="what does 'bien' mean?"))
        session.commit()
        return "¡Qué bien!"

    with (
        patch("app.routes.chat.model_chat", side_effect=reply_then_brainstorm),
        patch("app.translate.llm_translate.model_chat", side_effect=ModelServerUnavailableError("down")),
    ):
        result = take_turn(ChatTurnRequest(session_id="s1", message="Me gusta correr."), session=session)

    assert session.query(BrainstormTurn).filter_by(user_message_id=result.user_message_id).count() == 4
    assert [t.text for t in brainstorm.open_turns(session, "s1")] == ["what does 'bien' mean?"]


def test_fold_splits_notes_between_partner_and_reply_bubbles_and_is_idempotent():
    session = _brainstormed_session()
    result = _send(session)
    partner_id = session.query(ChatMessage).filter_by(role="assistant").order_by(ChatMessage.id).first().id
    req = FoldBrainstormRequest(session_id="s1", user_message_id=result.user_message_id)

    with patch(
        "app.routes.brainstorm.llm_translate.fold_brainstorm",
        return_value=(["'hacer' = to do"], ["'Me gusta correr' = I like running", "gustar + infinitive"]),
    ) as fold:
        first = brainstorm.fold(req, session=session)
        second = brainstorm.fold(req, session=session)

    assert fold.call_count == 1
    log, partner_text, sent, _, _ = fold.call_args.args
    assert partner_text == "¿Qué te gusta hacer?" and sent == "Me gusta correr."
    assert log[0] == ("learner", "what does 'hacer' mean here?")
    by_message = {m.message_id: m.notes for m in first.messages}
    assert by_message == {
        partner_id: ["'hacer' = to do"],
        result.user_message_id: ["'Me gusta correr' = I like running", "gustar + infinitive"],
    }
    assert second == first


def test_fold_without_a_partner_message_keeps_all_notes_on_the_reply():
    session = _session()
    session.add(BrainstormTurn(session_id="s1", role="learner", text="how do I say hi"))
    session.commit()
    result = _send(session, message="Hola")
    with patch("app.routes.brainstorm.llm_translate.fold_brainstorm", return_value=(["partner note"], ["reply note"])):
        folded = brainstorm.fold(FoldBrainstormRequest(session_id="s1", user_message_id=result.user_message_id), session=session)
    assert [(m.message_id, m.notes) for m in folded.messages] == [(result.user_message_id, ["partner note", "reply note"])]


def test_fold_with_no_log_skips_the_llm_and_marks_folded():
    session = _session()
    result = _send(session, message="Hola")
    with patch("app.routes.brainstorm.llm_translate.fold_brainstorm", side_effect=AssertionError("no LLM call")):
        folded = brainstorm.fold(FoldBrainstormRequest(session_id="s1", user_message_id=result.user_message_id), session=session)
    assert folded.messages == []
    assert session.get(ChatMessage, result.user_message_id).notes_folded


def test_fold_failure_leaves_notes_pending_for_a_retry():
    session = _brainstormed_session()
    result = _send(session)
    with patch(
        "app.routes.brainstorm.llm_translate.fold_brainstorm", side_effect=TranslationUnavailableError("down")
    ):
        with pytest.raises(HTTPException) as exc_info:
            brainstorm.fold(FoldBrainstormRequest(session_id="s1", user_message_id=result.user_message_id), session=session)
    assert exc_info.value.status_code == 503

    user_row = next(m for m in get_history("s1", session=session).messages if m.id == result.user_message_id)
    assert user_row.notes_pending
    assert len(user_row.brainstorm) == 4


def test_fold_rejects_another_sessions_message():
    session = _brainstormed_session()
    result = _send(session)
    with pytest.raises(HTTPException) as exc_info:
        brainstorm.fold(FoldBrainstormRequest(session_id="intruder", user_message_id=result.user_message_id), session=session)
    assert exc_info.value.status_code == 404


def test_history_carries_notes_and_the_raw_log():
    session = _brainstormed_session()
    result = _send(session)
    with patch("app.routes.brainstorm.llm_translate.fold_brainstorm", return_value=(["about partner"], ["about reply"])):
        brainstorm.fold(FoldBrainstormRequest(session_id="s1", user_message_id=result.user_message_id), session=session)

    messages = get_history("s1", session=session).messages
    partner, user, reply = messages[0], messages[1], messages[2]
    assert partner.notes == ["about partner"] and partner.brainstorm == []
    assert user.notes == ["about reply"] and not user.notes_pending
    assert [t.role for t in user.brainstorm] == ["learner", "coach", "learner", "coach"]
    assert user.brainstorm[-1].draft == "Me gusta correr."
    assert reply.notes == []


def test_clear_history_drops_brainstorm_turns_and_notes():
    session = _brainstormed_session()
    session.add(BrainstormTurn(session_id="s2", role="learner", text="keep me"))
    session.commit()
    result = _send(session)
    with patch("app.routes.brainstorm.llm_translate.fold_brainstorm", return_value=(["a"], ["b"])):
        brainstorm.fold(FoldBrainstormRequest(session_id="s1", user_message_id=result.user_message_id), session=session)

    clear_history("s1", session=session)

    assert session.query(BrainstormTurn).filter_by(session_id="s1").count() == 0
    assert session.query(BrainstormTurn).filter_by(session_id="s2").count() == 1
    assert session.query(MessageNote).count() == 0
