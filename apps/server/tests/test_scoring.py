import random
from datetime import datetime, timedelta, timezone

from app.wordbank.scoring import (
    WordState,
    apply_active_recall,
    apply_hover_penalty,
    apply_passive_exposure,
    effective_familiarity,
    select_new_words,
    select_reinforce_words,
    weighted_sample_without_replacement,
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


def test_weighted_sample_prefers_higher_weight_items_over_many_trials():
    rng = random.Random(42)
    items = ["low", "high"]
    weights = [1.0, 20.0]
    high_count = 0
    trials = 500
    for _ in range(trials):
        picked = weighted_sample_without_replacement(items, weights, k=1, rng=rng)
        if picked == ["high"]:
            high_count += 1
    # "high" has 20x the weight, so it should win the large majority of draws.
    assert high_count / trials > 0.85


def test_weighted_sample_without_replacement_no_duplicates():
    rng = random.Random(7)
    items = ["a", "b", "c", "d"]
    weights = [1.0, 2.0, 3.0, 4.0]
    picked = weighted_sample_without_replacement(items, weights, k=4, rng=rng)
    assert sorted(picked) == items


def test_select_reinforce_words_prioritizes_overdue_low_familiarity():
    rng = random.Random(1)
    words = [
        word(familiarity=0.95, days_since_review=0.0, interval=10.0, lemma="mastered"),
        word(familiarity=0.1, days_since_review=10.0, interval=1.0, lemma="weak_and_overdue"),
    ]
    counts = {"mastered": 0, "weak_and_overdue": 0}
    for _ in range(200):
        picked = select_reinforce_words(words, NOW, k=1, rng=rng)
        counts[picked[0]] += 1
    assert counts["weak_and_overdue"] > counts["mastered"]


def test_select_reinforce_words_respects_k_and_empty_input():
    words = [word(lemma=f"w{i}") for i in range(5)]
    picked = select_reinforce_words(words, NOW, k=3)
    assert len(picked) == 3
    assert select_reinforce_words([], NOW, k=3) == []


def test_select_new_words_excludes_known_and_respects_cap():
    candidates = ["ser", "estar", "tener", "hacer", "poder"]
    known = {"ser", "estar"}
    chosen = select_new_words(candidates, known, k=2, rng=random.Random(3))
    assert len(chosen) == 2
    assert not (set(chosen) & known)
    assert set(chosen) <= {"tener", "hacer", "poder"}


def test_select_new_words_returns_empty_when_all_known():
    candidates = ["ser", "estar"]
    known = {"ser", "estar"}
    assert select_new_words(candidates, known, k=2) == []
