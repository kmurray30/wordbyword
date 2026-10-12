"""The Brainstorm field: the learner's side chat with the coach about their
next message, kept apart from the official conversation with the partner.

- POST /brainstorm/turn: one typed or spoken brainstorm line -> the coach's
  streamed answer (and possibly a draft for the Reply box). Both sides are
  persisted as the open log.
- GET /brainstorm/open: the open log, so it survives a reload.
- POST /brainstorm/fold: once a reply is sent (which stamps the open log
  onto that reply - see routes/chat.py), summarize the log into short notes
  on the reply's bubble and on the partner message before it.

The partner model never sees any of this.
"""

import logging

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import NATIVE_LANGUAGE, TARGET_LANGUAGE
from app.db import get_session
from app.models import BrainstormTurn, ChatMessage, MessageNote
from app.schemas import (
    BrainstormDraftEvent,
    BrainstormErrorEvent,
    BrainstormIntentEvent,
    BrainstormLogTurn,
    BrainstormOpenResponse,
    BrainstormSavedEvent,
    BrainstormSayChunkEvent,
    BrainstormSayCompleteEvent,
    BrainstormTurnRequest,
    BrainstormVerdictEvent,
    FoldBrainstormRequest,
    FoldBrainstormResponse,
    MessageNotesOut,
)
from app.sse_streaming import KEEPALIVE_LINE, iter_with_keepalive
from app.translate import llm_translate

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/brainstorm", tags=["brainstorm"])

_HISTORY_TURNS = 8
_SSE_HEADERS = {"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"}


def _sse(event: str, data: str) -> str:
    return f"event: {event}\ndata: {data}\n\n"


def open_turns(session: Session, session_id: str) -> list[BrainstormTurn]:
    return list(
        session.scalars(
            select(BrainstormTurn)
            .where(BrainstormTurn.session_id == session_id, BrainstormTurn.user_message_id.is_(None))
            .order_by(BrainstormTurn.id)
        ).all()
    )


def to_log_turn(turn: BrainstormTurn) -> BrainstormLogTurn:
    return BrainstormLogTurn(id=turn.id, role=turn.role, text=turn.text, draft=turn.draft)


@router.get("/open", response_model=BrainstormOpenResponse)
def get_open(session_id: str, session: Session = Depends(get_session)) -> BrainstormOpenResponse:
    return BrainstormOpenResponse(turns=[to_log_turn(t) for t in open_turns(session, session_id)])


@router.post("/turn")
def brainstorm_turn(req: BrainstormTurnRequest, session: Session = Depends(get_session)) -> StreamingResponse:
    """SSE: "intent", "verdict", "draft" (for the Reply box, possibly ""),
    "say_chunk"/"say_complete" (the coach's answer), then "saved" once both
    turns are persisted. "error" means nothing usable came back. See
    llm_translate.brainstorm_stream for the field semantics."""
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="empty brainstorm turn")

    history_rows = session.scalars(
        select(ChatMessage)
        .where(ChatMessage.session_id == req.session_id)
        .order_by(ChatMessage.id.desc())
        .limit(_HISTORY_TURNS)
    ).all()
    history = [(row.role, row.text) for row in reversed(history_rows)]
    dialog = [(t.role, t.text) for t in open_turns(session, req.session_id)]
    words = [{"word": w.word, "confidence": w.confidence} for w in req.words]

    # Persisted (and committed) before the model call, as its own short
    # write - nothing may hold SQLite's write lock while the model streams
    # (see routes/chat.py's _gloss_and_persist_reply).
    learner_turn = BrainstormTurn(session_id=req.session_id, role="learner", text=req.text.strip())
    session.add(learner_turn)
    session.commit()
    learner_turn_id = learner_turn.id

    def events():
        draft = ""
        say = ""
        try:
            for item in iter_with_keepalive(
                llm_translate.brainstorm_stream(
                    req.text,
                    words,
                    dialog,
                    history,
                    NATIVE_LANGUAGE,
                    TARGET_LANGUAGE,
                    reply_draft=req.reply_draft,
                    spoken=req.source == "voice",
                )
            ):
                if item is None:
                    yield KEEPALIVE_LINE
                    continue
                kind, *rest = item
                if kind == "intent":
                    yield _sse("intent", BrainstormIntentEvent(intent=rest[0]).model_dump_json())
                elif kind == "verdict":
                    yield _sse("verdict", BrainstormVerdictEvent(verdict=rest[0]).model_dump_json())
                elif kind == "draft":
                    draft = rest[0]
                    yield _sse("draft", BrainstormDraftEvent(draft=draft).model_dump_json())
                elif kind == "say_chunk":
                    yield _sse("say_chunk", BrainstormSayChunkEvent(delta=rest[0]).model_dump_json())
                elif kind == "say_complete":
                    say = rest[0]
                    yield _sse("say_complete", BrainstormSayCompleteEvent(say=say).model_dump_json())
        except llm_translate.TranslationUnavailableError as exc:
            yield _sse("error", BrainstormErrorEvent(message=str(exc)).model_dump_json())
            return

        coach_turn = BrainstormTurn(session_id=req.session_id, role="coach", text=say, draft=draft)
        session.add(coach_turn)
        session.commit()
        yield _sse(
            "saved", BrainstormSavedEvent(learner_turn_id=learner_turn_id, coach_turn_id=coach_turn.id).model_dump_json()
        )

    return StreamingResponse(events(), media_type="text/event-stream", headers=_SSE_HEADERS)


def _fold_result(session: Session, user_message_id: int) -> FoldBrainstormResponse:
    notes = session.scalars(
        select(MessageNote)
        .where(MessageNote.source_user_message_id == user_message_id)
        .order_by(MessageNote.message_id, MessageNote.position)
    ).all()
    grouped: dict[int, list[str]] = {}
    for note in notes:
        grouped.setdefault(note.message_id, []).append(note.text)
    return FoldBrainstormResponse(
        messages=[MessageNotesOut(message_id=mid, notes=texts) for mid, texts in grouped.items()]
    )


@router.post("/fold", response_model=FoldBrainstormResponse)
def fold(req: FoldBrainstormRequest, session: Session = Depends(get_session)) -> FoldBrainstormResponse:
    """Idempotent: a message that's already folded just returns its notes,
    so the frontend can safely re-ask after a reload (notes_pending)."""
    message = session.get(ChatMessage, req.user_message_id)
    if message is None or message.role != "user" or message.session_id != req.session_id:
        raise HTTPException(status_code=404, detail="user message not found")
    if message.notes_folded:
        return _fold_result(session, message.id)

    turns = session.scalars(
        select(BrainstormTurn).where(BrainstormTurn.user_message_id == message.id).order_by(BrainstormTurn.id)
    ).all()
    partner = session.scalar(
        select(ChatMessage)
        .where(ChatMessage.session_id == message.session_id, ChatMessage.role == "assistant", ChatMessage.id < message.id)
        .order_by(ChatMessage.id.desc())
        .limit(1)
    )

    about_partner: list[str] = []
    about_reply: list[str] = []
    if turns:
        try:
            about_partner, about_reply = llm_translate.fold_brainstorm(
                [(t.role, t.text) for t in turns],
                partner.text if partner else "",
                message.text,
                NATIVE_LANGUAGE,
                TARGET_LANGUAGE,
            )
        except llm_translate.TranslationUnavailableError as exc:
            # notes_folded stays False, so history keeps reporting it as
            # pending and the frontend can retry later.
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    session.refresh(message)
    if message.notes_folded:  # a concurrent fold won the race - keep its notes
        return _fold_result(session, message.id)

    if not partner:
        # Nothing to hang them on - keep them with the reply instead.
        about_reply = about_partner + about_reply
        about_partner = []
    for target_id, texts in ((partner.id if partner else None, about_partner), (message.id, about_reply)):
        for position, text in enumerate(texts):
            session.add(
                MessageNote(message_id=target_id, source_user_message_id=message.id, position=position, text=text)
            )
    message.notes_folded = True
    session.commit()
    return _fold_result(session, message.id)
