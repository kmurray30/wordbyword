from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.chat.llama_client import ModelServerUnavailableError, chat as llama_chat
from app.chat.logit_bias import build_logit_bias
from app.chat.prompt_builder import build_messages
from app.config import NATIVE_LANGUAGE, TARGET_LANGUAGE, WORD_WEIGHTING_ENABLED
from app.db import get_session
from app.models import ChatMessage, MessageToken
from app.schemas import (
    ChatHistoryMessage,
    ChatHistoryResponse,
    ChatTurnRequest,
    ChatTurnResponse,
    ClearHistoryResponse,
    TokenAnnotation,
)
from app.translate.lemmatizer import analyze
from app.translate.service import gloss
from app.wordbank import store

router = APIRouter(prefix="/chat", tags=["chat"])

HISTORY_TURNS = 10


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
                TokenAnnotation(surface=t.surface, lemma=t.lemma, pos=t.pos, gloss=t.gloss, is_new=t.is_new)
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


@router.post("/turn", response_model=ChatTurnResponse)
def take_turn(req: ChatTurnRequest, session: Session = Depends(get_session)) -> ChatTurnResponse:
    reinforce_lemmas, new_lemmas, reinforce_urgency = store.pick_turn_vocabulary_if_enabled(
        session, WORD_WEIGHTING_ENABLED
    )
    history = _recent_history(session, req.session_id)
    messages = build_messages(history, reinforce_lemmas, new_lemmas, req.message)
    logit_bias = build_logit_bias(reinforce_lemmas, new_lemmas, reinforce_urgency)

    try:
        reply_text = llama_chat(messages, logit_bias)
    except ModelServerUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    session.add(ChatMessage(session_id=req.session_id, role="user", text=req.message))

    new_lemma_set = set(new_lemmas)
    tokens = analyze(reply_text)
    annotations: list[TokenAnnotation] = []
    token_rows: list[MessageToken] = []

    for position, tok in enumerate(tokens):
        if not tok.is_spanish:
            annotations.append(TokenAnnotation(surface=tok.surface, lemma=tok.lemma, pos=tok.pos, gloss="", is_new=False))
            continue

        word_gloss = gloss(tok.lemma, TARGET_LANGUAGE, NATIVE_LANGUAGE)
        entry = store.record_exposure(session, tok.lemma, pos=tok.pos, translation=word_gloss)
        is_new = tok.lemma in new_lemma_set or entry.exposure_count == 1

        annotations.append(
            TokenAnnotation(surface=tok.surface, lemma=tok.lemma, pos=tok.pos, gloss=word_gloss, is_new=is_new)
        )
        token_rows.append(
            MessageToken(
                position=position,
                surface=tok.surface,
                lemma=tok.lemma,
                pos=tok.pos,
                gloss=word_gloss,
                is_new=is_new,
            )
        )

    assistant_message = ChatMessage(session_id=req.session_id, role="assistant", text=reply_text, tokens=token_rows)
    session.add(assistant_message)
    session.commit()
    session.refresh(assistant_message)

    return ChatTurnResponse(message_id=assistant_message.id, text=reply_text, tokens=annotations)
