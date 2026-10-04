"""The word-bank familiarity/exposure scoring algorithm.

Every lemma has a `familiarity` score that decays over time (a forgetting
curve) and is nudged up or down by reward events tied directly to the
hover-translation feature:

  - the agent uses a word and the user does NOT hover it   -> small boost
    (passive recognition)
  - the user hovers/translates a word the agent used       -> penalty, and
    the word becomes due for review sooner
  - the user correctly types the word themselves            -> larger boost
    (active recall is stronger evidence of mastery)

Nothing here talks to the LLM or the DB directly, so it's cheap to unit
test in isolation.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from app.config import (
    ACTIVE_RECALL_BOOST,
    HOVER_PENALTY,
    INTERVAL_GROWTH_ON_SUCCESS,
    INTERVAL_SHRINK_ON_FAILURE,
    MAX_REVIEW_INTERVAL_DAYS,
    MIN_REVIEW_INTERVAL_DAYS,
    PASSIVE_EXPOSURE_BOOST,
)


@dataclass
class WordState:
    """A plain-data mirror of `models.WordBankEntry`, decoupled from the ORM
    so the scoring math can be unit tested without a database."""

    lemma: str
    familiarity: float
    last_reviewed_at: datetime
    review_interval_days: float


def _days_between(later: datetime, earlier: datetime) -> float:
    return max(0.0, (later - earlier).total_seconds() / 86400.0)


def effective_familiarity(word: WordState, now: datetime) -> float:
    """Decayed familiarity at `now`, using `review_interval_days` as the
    half-life: a word not reinforced for one interval has "lost" half of the
    familiarity it had at the last review."""

    days_since = _days_between(now, word.last_reviewed_at)
    half_life = max(word.review_interval_days, MIN_REVIEW_INTERVAL_DAYS)
    decay = 0.5 ** (days_since / half_life)
    return word.familiarity * decay


def _clamp_interval(days: float) -> float:
    return min(MAX_REVIEW_INTERVAL_DAYS, max(MIN_REVIEW_INTERVAL_DAYS, days))


def apply_passive_exposure(word: WordState, now: datetime) -> WordState:
    """Agent used the word, user didn't need to look it up."""
    current = effective_familiarity(word, now)
    new_familiarity = current + (1.0 - current) * PASSIVE_EXPOSURE_BOOST
    return WordState(
        lemma=word.lemma,
        familiarity=new_familiarity,
        last_reviewed_at=now,
        review_interval_days=_clamp_interval(word.review_interval_days * INTERVAL_GROWTH_ON_SUCCESS),
    )


def apply_hover_penalty(word: WordState, now: datetime) -> WordState:
    """User hovered/translated the word - they didn't know it."""
    current = effective_familiarity(word, now)
    new_familiarity = current * (1.0 - HOVER_PENALTY)
    return WordState(
        lemma=word.lemma,
        familiarity=new_familiarity,
        last_reviewed_at=now,
        review_interval_days=_clamp_interval(word.review_interval_days * INTERVAL_SHRINK_ON_FAILURE),
    )


def apply_active_recall(word: WordState, now: datetime) -> WordState:
    """User typed the word themselves, unprompted."""
    current = effective_familiarity(word, now)
    new_familiarity = current + (1.0 - current) * ACTIVE_RECALL_BOOST
    return WordState(
        lemma=word.lemma,
        familiarity=new_familiarity,
        last_reviewed_at=now,
        review_interval_days=_clamp_interval(word.review_interval_days * INTERVAL_GROWTH_ON_SUCCESS),
    )


