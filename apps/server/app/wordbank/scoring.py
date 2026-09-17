"""The word-bank weighting algorithm.

This is the "RL-ish" piece: every lemma has a `familiarity` score that decays
over time (a forgetting curve) and is nudged up or down by reward events tied
directly to the hover-translation feature:

  - the agent uses a word and the user does NOT hover it   -> small boost
    (passive recognition)
  - the user hovers/translates a word the agent used       -> penalty, and
    the word becomes due for review sooner
  - the user correctly types the word themselves            -> larger boost
    (active recall is stronger evidence of mastery)

On each turn we (a) pick a weighted-random sample of words that are "due" for
reinforcement, biased toward low-familiarity / overdue words, and (b) pick a
capped number of brand-new words to introduce. Both lists feed the prompt
builder; nothing here talks to the LLM or the DB directly, so it's cheap to
unit test in isolation.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from datetime import datetime, timezone

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


def _reinforce_weight(word: WordState, now: datetime) -> float:
    familiarity = effective_familiarity(word, now)
    days_since = _days_between(now, word.last_reviewed_at)
    overdue_ratio = days_since / max(word.review_interval_days, MIN_REVIEW_INTERVAL_DAYS)
    # +epsilon so a fully-mastered, not-yet-due word still has a small chance
    # of being sampled (keeps review from feeling purely mechanical).
    return max(overdue_ratio, 0.01) * (1.0 - familiarity + 0.05)


def weighted_sample_without_replacement(
    items: list[str], weights: list[float], k: int, rng: random.Random | None = None
) -> list[str]:
    """Efraimidis-Spirakis weighted sampling without replacement: draw a key
    key_i = U_i ** (1 / weight_i) for each item and take the top k by key.
    Equivalent to sequential weighted sampling but doesn't require mutating
    the candidate set."""

    rng = rng or random
    if k <= 0 or not items:
        return []
    keyed = []
    for item, weight in zip(items, weights):
        weight = max(weight, 1e-9)
        u = rng.random()
        key = u ** (1.0 / weight)
        keyed.append((key, item))
    keyed.sort(key=lambda pair: pair[0], reverse=True)
    return [item for _, item in keyed[:k]]


def select_reinforce_words(
    words: list[WordState], now: datetime, k: int, rng: random.Random | None = None
) -> list[str]:
    """Pick up to k lemmas to steer the agent toward this turn, biased
    toward low-familiarity / overdue words but not deterministic top-N."""

    if not words:
        return []
    weights = [_reinforce_weight(w, now) for w in words]
    lemmas = [w.lemma for w in words]
    return weighted_sample_without_replacement(lemmas, weights, k, rng=rng)


def select_new_words(
    frequency_ranked_candidates: list[str],
    known_lemmas: set[str],
    k: int,
    rng: random.Random | None = None,
    jitter_window: int = 8,
) -> list[str]:
    """Pick up to k brand-new lemmas to introduce, drawn from a small window
    at the front of the frequency-ranked candidate list (so introduction
    order roughly follows word frequency but isn't perfectly deterministic)."""

    rng = rng or random
    unseen = [w for w in frequency_ranked_candidates if w not in known_lemmas]
    if not unseen or k <= 0:
        return []
    window = unseen[: max(jitter_window, k)]
    chosen = window[:]
    rng.shuffle(chosen)
    return chosen[:k]
