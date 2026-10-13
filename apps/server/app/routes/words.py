"""The learner's saved words (see models.SavedWord)."""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db import get_session
from app.models import SavedWord
from app.schemas import RemoveSavedWordResponse, SavedWordIn, SavedWordOut, SavedWordsResponse

router = APIRouter(prefix="/words", tags=["words"])


def _key(word: str) -> str:
    return word.strip().lower()


def _out(row: SavedWord) -> SavedWordOut:
    return SavedWordOut(
        id=row.id,
        word=row.word,
        gloss=row.gloss,
        grammar=row.grammar,
        sentence=row.sentence,
        message_id=row.message_id,
        created_at=row.created_at,
    )


@router.get("", response_model=SavedWordsResponse)
def list_words(session_id: str, session: Session = Depends(get_session)) -> SavedWordsResponse:
    rows = session.scalars(
        select(SavedWord).where(SavedWord.session_id == session_id).order_by(SavedWord.created_at.desc())
    ).all()
    return SavedWordsResponse(words=[_out(r) for r in rows])


@router.post("", response_model=SavedWordOut)
def save_word(req: SavedWordIn, session: Session = Depends(get_session)) -> SavedWordOut:
    """Saving a word that's already saved returns the existing entry."""
    key = _key(req.word)
    if not key:
        raise HTTPException(status_code=422, detail="empty word")
    existing = session.scalars(
        select(SavedWord).where(SavedWord.session_id == req.session_id, SavedWord.word_key == key)
    ).first()
    if existing is not None:
        return _out(existing)
    row = SavedWord(
        session_id=req.session_id,
        word=req.word.strip()[:128],
        word_key=key[:128],
        gloss=req.gloss[:256],
        grammar=req.grammar[:512],
        sentence=req.sentence[:2000],
        message_id=req.message_id,
    )
    session.add(row)
    session.commit()
    return _out(row)


@router.delete("", response_model=RemoveSavedWordResponse)
def remove_word(session_id: str, word: str, session: Session = Depends(get_session)) -> RemoveSavedWordResponse:
    rows = session.scalars(
        select(SavedWord).where(SavedWord.session_id == session_id, SavedWord.word_key == _key(word))
    ).all()
    for row in rows:
        session.delete(row)
    session.commit()
    return RemoveSavedWordResponse(removed=len(rows))
