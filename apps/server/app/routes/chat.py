from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.chat.openai_client import ModelServerUnavailableError, chat as model_chat, chat_stream as model_chat_stream
from app.chat.prompt_builder import build_messages
from app.config import NATIVE_LANGUAGE, TARGET_LANGUAGE
from app.db import get_session
from app.models import ChatMessage, MessageToken
from app.schemas import (
    ChatHistoryMessage,
    ChatHistoryResponse,
    ChatTurnChunkEvent,
    ChatTurnErrorEvent,
    ChatTurnRequest,
    ChatTurnResponse,
    ClearHistoryResponse,
    TokenAnnotation,
)
from app.translate import llm_translate
from app.translate.lemmatizer import Token, analyze
from app.translate.span_matching import MatchedSpan, match_spans
from app.wordbank import store

router = APIRouter(prefix="/chat", tags=["chat"])


def _sse(event: str, data: str) -> str:
    return f"event: {event}\ndata: {data}\n\n"


HISTORY_TURNS = 10


def _match_token_glosses(tokens: list[Token], matched_spans: list[MatchedSpan]) -> list[tuple[str, str, str]]:
    """Both `tokens` (spaCy, left-to-right) and `matched_spans` (left-to-
    right, non-overlapping - see span_matching.match_spans) are ordered by
    position, so a single left-to-right walk finds each token's covering
    span, if any, without re-scanning from the start each time. A
    multi-word span (e.g. "el tuyo") covers more than one token - every
    token inside it gets that same span's gloss/note/literal, rather than
    only the first one or splitting the group across two separate
    glosses."""
    result: list[tuple[str, str, str]] = []
    span_idx = 0
    for tok in tokens:
        while span_idx < len(matched_spans) and matched_spans[span_idx].end <= tok.start:
            span_idx += 1
        if span_idx < len(matched_spans) and matched_spans[span_idx].start <= tok.start < matched_spans[span_idx].end:
            span = matched_spans[span_idx]
            result.append((span.gloss, span.note, span.literal))
        else:
            result.append(("", "", ""))
    return result


def _recent_history(session: Session, session_id: str) -> list[tuple[str, str]]:
    rows = session.scalars(
        select(ChatMessage)
        .where(ChatMessage.session_id == session_id)
        .order_by(ChatMessage.id.desc())
        .limit(HISTORY_TURNS)
    ).all()
    rows = list(reversed(rows))
    return [(row.role, row.text) for row in rows]


@router.get("/history", response_model=ChatHistoryResponse)
def get_history(session_id: str, session: Session = Depends(get_session)) -> ChatHistoryResponse:
    rows = session.scalars(
        select(ChatMessage).where(ChatMessage.session_id == session_id).order_by(ChatMessage.id)
    ).all()
    messages = [
        ChatHistoryMessage(
            id=row.id,
            role=row.role,
            text=row.text,
            tokens=[
                TokenAnnotation(
                    surface=t.surface,
                    lemma=t.lemma,
                    pos=t.pos,
                    gloss=t.gloss,
                    is_new=t.is_new,
                    note=t.note,
                    literal=t.literal,
                )
                for t in row.tokens
            ],
        )
        for row in rows
    ]
    return ChatHistoryResponse(messages=messages)


@router.post("/history/clear", response_model=ClearHistoryResponse)
def clear_history(session_id: str, session: Session = Depends(get_session)) -> ClearHistoryResponse:
    rows = session.scalars(select(ChatMessage).where(ChatMessage.session_id == session_id)).all()
    for row in rows:
        session.delete(row)
    session.commit()
    return ClearHistoryResponse(cleared=len(rows))


def _gloss_and_persist_reply(session: Session, session_id: str, reply_text: str) -> ChatTurnResponse:
    """Shared by both /chat/turn and /chat/turn/stream - everything that
    happens once the model's full reply text is known: one LLM call glosses
    every word/group in the reply using its meaning IN CONTEXT, and returns
    the whole-sentence translation from that exact same pass - one
    consistent source for both. No dictionary/MT fallback for a word it
    missed or a failed call - that word just gets no gloss until a later
    turn supplies one (see store.record_exposure below)."""
    try:
        translation, raw_spans = llm_translate.gloss_reply(reply_text, TARGET_LANGUAGE, NATIVE_LANGUAGE)
    except llm_translate.TranslationUnavailableError:
        translation, raw_spans = "", []

    tokens = analyze(reply_text)
    matched_spans = match_spans(reply_text, raw_spans)
    token_glosses = _match_token_glosses(tokens, matched_spans)
    annotations: list[TokenAnnotation] = []
    token_rows: list[MessageToken] = []

    for position, (tok, (word_gloss, word_note, word_literal)) in enumerate(zip(tokens, token_glosses)):
        # tok.is_spanish is a heuristic built for the learner's own possibly-
        # English-mixed input (see lemmatizer.py); it's gated on a small
        # ~300-word frequency list, so applying it here to the agent's own
        # reply - which is always Spanish - silently dropped the gloss (and
        # word-bank tracking) for any real Spanish word outside that list,
        # e.g. "tormenta". Every alphabetic token in the agent's reply is
        # Spanish by construction; only punctuation has nothing to gloss.
        if not tok.surface.isalpha():
            annotations.append(TokenAnnotation(surface=tok.surface, lemma=tok.lemma, pos=tok.pos, gloss="", is_new=False))
            # Persisted too (not just returned live) - a MessageToken row
            # with no gloss, same as any other token with nothing to show.
            # Skipping this used to silently drop every punctuation mark
            # from a message's token array on the very next history
            # reload (GET /chat/history reconstructs tokens purely from
            # these rows), which joinTokens.ts then rendered as words
            # mashed directly together with no sign anything was missing.
            token_rows.append(MessageToken(position=position, surface=tok.surface, lemma=tok.lemma, pos=tok.pos))
            continue

        word_note = word_note if word_gloss else ""
        word_literal = word_literal if word_gloss else ""
        entry = store.record_exposure(session, tok.lemma, pos=tok.pos, translation=word_gloss)
        is_new = entry.exposure_count == 1

        annotations.append(
            TokenAnnotation(
                surface=tok.surface,
                lemma=tok.lemma,
                pos=tok.pos,
                gloss=word_gloss,
                is_new=is_new,
                note=word_note,
                literal=word_literal,
            )
        )
        token_rows.append(
            MessageToken(
                position=position,
                surface=tok.surface,
                lemma=tok.lemma,
                pos=tok.pos,
                gloss=word_gloss,
                is_new=is_new,
                note=word_note,
                literal=word_literal,
            )
        )

    assistant_message = ChatMessage(session_id=session_id, role="assistant", text=reply_text, tokens=token_rows)
    session.add(assistant_message)
    session.commit()
    session.refresh(assistant_message)

    return ChatTurnResponse(message_id=assistant_message.id, text=reply_text, tokens=annotations, translation=translation)


@router.post("/turn", response_model=ChatTurnResponse)
def take_turn(req: ChatTurnRequest, session: Session = Depends(get_session)) -> ChatTurnResponse:
    history = _recent_history(session, req.session_id)
    messages = build_messages(history, req.message)

    try:
        reply_text = model_chat(messages)
    except ModelServerUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    if not reply_text.strip():
        # The model returned a 200 with literally no content - seen live
        # against a reasoning-style model whose internal reasoning tokens
        # can exhaust the whole generation budget before any visible
        # output, on a harder turn (see MAX_REPLY_TOKENS's comment in
        # config.py). Surfacing this as a real error rather than silently
        # persisting a blank assistant turn - a vacant chat bubble the
        # learner can't tell apart from an actual (empty) reply, sitting
        # in history forever once saved.
        raise HTTPException(status_code=502, detail="Model returned an empty reply")

    session.add(ChatMessage(session_id=req.session_id, role="user", text=req.message))

    return _gloss_and_persist_reply(session, req.session_id, reply_text)


@router.post("/turn/stream")
def take_turn_stream(req: ChatTurnRequest, session: Session = Depends(get_session)) -> StreamingResponse:
    """Same turn as POST /chat/turn, but the model's reply streams to the
    client as it's generated instead of waiting for the whole thing - for a
    live, typing-style display. Streamed as Server-Sent Events: a "chunk"
    event per token/delta of the reply's own text, then (once generation
    finishes) the exact same gloss/translate/persist work /chat/turn does,
    sent as one final "done" event carrying the exact same ChatTurnResponse
    shape /chat/turn returns - so the frontend ends up with one consistent
    payload either way, just delivered progressively here."""
    history = _recent_history(session, req.session_id)
    messages = build_messages(history, req.message)
    session.add(ChatMessage(session_id=req.session_id, role="user", text=req.message))

    def events():
        chunks: list[str] = []
        try:
            for delta in model_chat_stream(messages):
                chunks.append(delta)
                yield _sse("chunk", ChatTurnChunkEvent(delta=delta).model_dump_json())
        except ModelServerUnavailableError as exc:
            yield _sse("error", ChatTurnErrorEvent(message=str(exc)).model_dump_json())
            return

        reply_text = "".join(chunks).strip()
        if not reply_text:
            # Same empty-reply case take_turn guards against (see its own
            # comment) - here it shows up as a stream with zero "chunk"
            # events at all (the frontend's "…" placeholder never fills
            # in), so surface it the same way: an "error" event instead of
            # silently persisting (and then "done"-ing with) a blank
            # assistant turn. The pending user-message add above is never
            # committed in this branch, so it doesn't leave a dangling
            # user-only turn in history either.
            yield _sse("error", ChatTurnErrorEvent(message="Model returned an empty reply").model_dump_json())
            return
        result = _gloss_and_persist_reply(session, req.session_id, reply_text)
        yield _sse("done", result.model_dump_json())

    return StreamingResponse(events(), media_type="text/event-stream")
