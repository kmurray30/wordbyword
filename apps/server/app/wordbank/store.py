from datetime import datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import DEFAULT_REVIEW_INTERVAL_DAYS
from app.models import WordBankEntry
from app.wordbank.scoring import WordState, apply_active_recall, apply_hover_penalty, apply_passive_exposure


def _to_state(entry: WordBankEntry) -> WordState:
    return WordState(
        lemma=entry.lemma,
        familiarity=entry.familiarity,
        last_reviewed_at=entry.last_reviewed_at,
        review_interval_days=entry.review_interval_days,
    )


def get_or_create(session: Session, lemma: str, pos: str = "", translation: str = "") -> WordBankEntry:
    entry = session.scalar(select(WordBankEntry).where(WordBankEntry.lemma == lemma))
    if entry is None:
        now = datetime.utcnow()
        entry = WordBankEntry(
            lemma=lemma,
            pos=pos,
            primary_translation=translation,
            familiarity=0.0,
            exposure_count=0,
            introduced_at=now,
            last_seen_at=now,
            last_reviewed_at=now,
            review_interval_days=DEFAULT_REVIEW_INTERVAL_DAYS,
        )
        session.add(entry)
        session.flush()
    return entry


def record_exposure(session: Session, lemma: str, pos: str = "", translation: str = "") -> WordBankEntry:
    """Called for every lemma the agent produces this turn, whether or not
    it was one of the words we deliberately steered toward."""
    entry = get_or_create(session, lemma, pos=pos, translation=translation)
    entry.exposure_count += 1
    entry.last_seen_at = datetime.utcnow()
    if translation and not entry.primary_translation:
        entry.primary_translation = translation
    return entry


def _apply(session: Session, lemma: str, transform) -> WordBankEntry:
    # get_or_create, not a plain lookup: userTypedSpanishWord can name a word
    # the agent has never produced (so it has no word_bank row yet) but the
    # user clearly already knows - that's still real active-recall evidence
    # and shouldn't be dropped just because nothing introduced the word first.
    entry = get_or_create(session, lemma)
    now = datetime.utcnow()
    updated = transform(_to_state(entry), now)
    entry.familiarity = updated.familiarity
    entry.last_reviewed_at = updated.last_reviewed_at
    entry.review_interval_days = updated.review_interval_days
    return entry


def apply_reward_event(session: Session, event_type: str, lemma: str) -> WordBankEntry:
    transforms = {
        "wordSeenNoHover": apply_passive_exposure,
        "wordHovered": apply_hover_penalty,
        "userTypedSpanishWord": apply_active_recall,
    }
    transform = transforms.get(event_type)
    if transform is None:
        raise ValueError(f"unknown reward event type: {event_type}")
    return _apply(session, lemma, transform)


def list_all(session: Session) -> list[WordBankEntry]:
    return list(session.scalars(select(WordBankEntry).order_by(WordBankEntry.lemma)).all())
