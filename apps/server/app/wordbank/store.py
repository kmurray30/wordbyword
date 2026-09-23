from datetime import datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import DEFAULT_REVIEW_INTERVAL_DAYS, NEW_WORDS_PER_TURN, REINFORCE_WORDS_PER_TURN
from app.models import WordBankEntry
from app.wordbank.frequency_list import NEW_WORD_CANDIDATES_ES
from app.wordbank.function_words import EXCLUDED_REINFORCE_POS
from app.wordbank.scoring import (
    WordState,
    apply_active_recall,
    apply_hover_penalty,
    apply_passive_exposure,
    reinforce_urgency,
    select_new_words,
    select_reinforce_words,
)


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


def _reinforce_candidates(entries: list[WordBankEntry]) -> list[WordBankEntry]:
    """Word-bank entries eligible to be force-boosted via logit_bias this
    turn - excludes closed-class (function) words. See function_words.py
    for why: they have a narrow grammatical slot, so hard-boosting one
    tends to produce a broken sentence rather than a natural one. Entries
    created before a POS was known (e.g. via a reward event for a word the
    agent never produced) have pos == "" and stay eligible by default."""
    return [e for e in entries if e.pos not in EXCLUDED_REINFORCE_POS]


def pick_turn_vocabulary(session: Session) -> tuple[list[str], list[str], dict[str, float]]:
    """Returns (reinforce_lemmas, new_lemmas, reinforce_urgency) for building
    this turn's prompt and its logit_bias. `reinforce_urgency` is a [0, 1]
    per-lemma score (only for the reinforce list) used to scale how strongly
    each word's logit_bias nudges generation."""
    entries = session.scalars(select(WordBankEntry)).all()
    now = datetime.utcnow()

    reinforce_states = [_to_state(e) for e in _reinforce_candidates(entries)]
    reinforce = select_reinforce_words(reinforce_states, now, REINFORCE_WORDS_PER_TURN)
    urgency = reinforce_urgency(reinforce_states, now)
    known_lemmas = {e.lemma for e in entries}
    new_words = select_new_words(NEW_WORD_CANDIDATES_ES, known_lemmas, NEW_WORDS_PER_TURN)
    return reinforce, new_words, {lemma: urgency[lemma] for lemma in reinforce}


def pick_turn_vocabulary_if_enabled(
    session: Session, enabled: bool
) -> tuple[list[str], list[str], dict[str, float]]:
    """Same as pick_turn_vocabulary, but returns nothing to steer toward
    when `enabled` is False (config.WORD_WEIGHTING_ENABLED) - the kill
    switch for comparing raw vs. steered model behavior. Doesn't touch the
    session at all in that case, so it's safe to call with disabled=True
    outside of a real request too."""
    if not enabled:
        return [], [], {}
    return pick_turn_vocabulary(session)


def list_all(session: Session) -> list[WordBankEntry]:
    return list(session.scalars(select(WordBankEntry).order_by(WordBankEntry.lemma)).all())
