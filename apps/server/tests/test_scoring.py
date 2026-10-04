from datetime import datetime, timedelta, timezone

from app.wordbank.scoring import (
    WordState,
    apply_active_recall,
    apply_hover_penalty,
    apply_passive_exposure,
    effective_familiarity,
)

NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


def word(familiarity=0.5, days_since_review=0.0, interval=1.0, lemma="hablar"):
    return WordState(
        lemma=lemma,
        familiarity=familiarity,
        last_reviewed_at=NOW - timedelta(days=days_since_review),
        review_interval_days=interval,
    )


def test_effective_familiarity_no_decay_at_review_time():
    w = word(familiarity=0.8, days_since_review=0.0, interval=2.0)
    assert effective_familiarity(w, NOW) == 0.8


def test_effective_familiarity_halves_after_one_interval():
    w = word(familiarity=0.8, days_since_review=2.0, interval=2.0)
    assert abs(effective_familiarity(w, NOW) - 0.4) < 1e-9


def test_effective_familiarity_decays_further_over_time():
    w = word(familiarity=0.8, days_since_review=4.0, interval=2.0)
    assert abs(effective_familiarity(w, NOW) - 0.2) < 1e-9


def test_passive_exposure_boosts_familiarity_and_interval():
    w = word(familiarity=0.5, days_since_review=0.0, interval=1.0)
    updated = apply_passive_exposure(w, NOW)
    assert updated.familiarity > 0.5
    assert updated.review_interval_days > 1.0
    assert updated.last_reviewed_at == NOW


def test_hover_penalty_reduces_familiarity_and_shrinks_interval():
    w = word(familiarity=0.5, days_since_review=0.0, interval=4.0)
    updated = apply_hover_penalty(w, NOW)
    assert updated.familiarity < 0.5
    assert updated.review_interval_days < 4.0


def test_active_recall_boosts_more_than_passive_exposure():
    passive = apply_passive_exposure(word(familiarity=0.5), NOW)
    active = apply_active_recall(word(familiarity=0.5), NOW)
    assert active.familiarity > passive.familiarity


def test_familiarity_never_exceeds_one():
    w = word(familiarity=0.99)
    for _ in range(20):
        w = apply_active_recall(w, NOW)
    assert w.familiarity <= 1.0


def test_familiarity_never_goes_negative():
    w = word(familiarity=0.05)
    for _ in range(20):
        w = apply_hover_penalty(w, NOW)
    assert w.familiarity >= 0.0
