from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.chat.ollama_client import OllamaUnavailableError, chat as ollama_chat
from app.chat.prompt_builder import build_messages
from app.config import NATIVE_LANGUAGE, TARGET_LANGUAGE
from app.db import get_session
from app.models import ChatMessage, MessageToken
from app.schemas import ChatTurnRequest, ChatTurnResponse, TokenAnnotation
from app.translate.lemmatizer import analyze
from app.translate.service import gloss
from app.wordbank import store

router = APIRouter(prefix="/chat", tags=["chat"])

HISTORY_TURNS = 10


def _recent_history(session: Session) -> list[tuple[str, str]]:
    rows = session.scalars(
        select(ChatMessage).order_by(ChatMessage.id.desc()).limit(HISTORY_TURNS)
    ).all()
    rows = list(reversed(rows))
    return [(row.role, row.text) for row in rows]


@router.post("/turn", response_model=ChatTurnResponse)
def take_turn(req: ChatTurnRequest, session: Session = Depends(get_session)) -> ChatTurnResponse:
    reinforce_lemmas, new_lemmas = store.pick_turn_vocabulary(session)
    history = _recent_history(session)
    messages = build_messages(history, reinforce_lemmas, new_lemmas, req.message)

    try:
        reply_text = ollama_chat(messages)
    except OllamaUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    session.add(ChatMessage(role="user", text=req.message))

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

    assistant_message = ChatMessage(role="assistant", text=reply_text, tokens=token_rows)
    session.add(assistant_message)
    session.commit()
    session.refresh(assistant_message)

    return ChatTurnResponse(message_id=assistant_message.id, text=reply_text, tokens=annotations)
