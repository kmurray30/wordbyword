from dataclasses import dataclass

from app.wordbank.store import _reinforce_candidates, pick_turn_vocabulary_if_enabled


@dataclass
class FakeEntry:
    """Stand-in for WordBankEntry - _reinforce_candidates only touches .pos,
    so a real DB session isn't needed to test the filtering."""

    lemma: str
    pos: str


def test_excludes_closed_class_entries():
    entries = [
        FakeEntry("hablar", "VERB"),
        FakeEntry("por", "ADP"),
        FakeEntry("casa", "NOUN"),
        FakeEntry("que", "SCONJ"),
        FakeEntry("el", "DET"),
        FakeEntry("importante", "ADJ"),
    ]
    kept = [e.lemma for e in _reinforce_candidates(entries)]
    assert kept == ["hablar", "casa", "importante"]


def test_keeps_entries_with_unknown_empty_pos():
    # Created via a reward event for a word the agent never produced, before
    # any POS tag was ever captured - should stay eligible, not be dropped.
    entries = [FakeEntry("desconocido", "")]
    assert _reinforce_candidates(entries) == entries


def test_empty_input():
    assert _reinforce_candidates([]) == []


class ExplodingSession:
    """Stands in for a DB session that must never be touched - proves
    pick_turn_vocabulary_if_enabled(disabled) really is a no-op short
    circuit, not just returning the right thing after doing real work."""

    def scalars(self, *args, **kwargs):
        raise AssertionError("session should not be queried when weighting is disabled")


def test_disabled_returns_empty_without_touching_the_session():
    result = pick_turn_vocabulary_if_enabled(ExplodingSession(), enabled=False)
    assert result == ([], [], {})
