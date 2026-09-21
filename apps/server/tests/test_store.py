from dataclasses import dataclass

from app.wordbank.store import _reinforce_candidates


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
