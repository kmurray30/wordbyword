"""Picks the low-pressure "wildcard" review words - and the reward
bookkeeping that teaches it which words are worth offering.

Thompson sampling, with the SRS signal folded in: each candidate's draw is

    (WILDCARD_URGENCY_FLOOR + review_urgency) * Beta(alpha, beta).sample()

- review_urgency (wordbank/scoring.py) says how valuable reinforcing the
  word is right now: overdue and not yet known.
- Beta(alpha, beta) is the bandit's belief about how likely the word is to
  actually land when offered - get worked into the reply rather than
  ignored. Some words just don't fit most conversations; after a few
  misses they stop wasting slots, while sampling (rather than taking the
  mean) keeps giving uncertain words an occasional chance.

Both the relevance picks and the wildcards feed the same counts, so every
suggestion is a data point. Counts decay toward the prior on each update
(SUGGEST_BANDIT_DECAY), so a word that never fit last week's topics isn't
written off forever.

Pure functions only, like scoring.py - no DB, no LLM.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.config import SUGGEST_BANDIT_DECAY

# Keeps a not-yet-due word from being impossible to pick: "fuzzy" value,
# not a hard schedule.
WILDCARD_URGENCY_FLOOR = 0.1

# A suggested word the learner later types themselves counts this much
# extra on top of the reply having used it.
PICKED_UP_BONUS = 1.0


@dataclass
class BanditArm:
    lemma: str
    urgency: float
    alpha: float
    beta: float
    # Multiplies the draw - e.g. words the learner saved to their list.
    weight: float = 1.0


def thompson_pick(arms: list[BanditArm], k: int, rng: np.random.Generator) -> list[str]:
    if k <= 0 or not arms:
        return []
    draws = [
        arm.weight * (WILDCARD_URGENCY_FLOOR + arm.urgency) * rng.beta(max(arm.alpha, 1e-3), max(arm.beta, 1e-3))
        for arm in arms
    ]
    order = np.argsort(draws)[::-1][:k]
    return [arms[i].lemma for i in order]


def update_counts(alpha: float, beta: float, success: float) -> tuple[float, float]:
    """One reward observation in [0, 1] (or above, for a bonus), after
    decaying the old counts toward the Beta(1, 1) prior."""
    alpha = 1.0 + (alpha - 1.0) * SUGGEST_BANDIT_DECAY
    beta = 1.0 + (beta - 1.0) * SUGGEST_BANDIT_DECAY
    if success > 0:
        alpha += success
    else:
        beta += 1.0
    return alpha, beta
