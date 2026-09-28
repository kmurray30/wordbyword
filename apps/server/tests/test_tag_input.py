from unittest.mock import patch

from app.routes.translate import tag_input
from app.schemas import TagInputRequest


def _run(text: str):
    with patch(
        "app.routes.translate.word_candidates",
        side_effect=lambda word, source, target: [{"translation": f"{word}->{target}", "description": ""}],
    ):
        return tag_input(TagInputRequest(text=text))


def _find(response, surface: str):
    return next(t for t in response.tokens if t.surface == surface)


def test_spanish_word_gets_single_english_column():
    tok = _find(_run("Quiero comer pan"), "comer")
    assert tok.is_spanish is True
    assert [c.language for c in tok.columns] == ["en"]


def test_english_word_gets_single_spanish_column():
    tok = _find(_run("I want pan"), "want")
    assert tok.is_spanish is False
    assert [c.language for c in tok.columns] == ["es"]


def test_cognate_gets_both_columns():
    tok = _find(_run("Vivo en un hotel"), "hotel")
    assert {c.language for c in tok.columns} == {"en", "es"}


def test_punctuation_gets_no_columns():
    tok = _find(_run("Hola,"), ",")
    assert tok.columns == []
