from unittest.mock import patch

from app.routes.translate import tag_input
from app.schemas import TagInputRequest


def _run(text: str, *, spanish_words: set[str] = frozenset(), english_words: set[str] = frozenset()):
    with (
        patch(
            "app.routes.translate.word_candidates",
            side_effect=lambda word, source, target: [{"translation": f"{word}->{target}", "description": ""}],
        ),
        patch("app.routes.translate.is_valid_spanish_word", side_effect=lambda w: w.lower() in spanish_words),
        patch("app.routes.translate.is_valid_english_word", side_effect=lambda w: w.lower() in english_words),
    ):
        return tag_input(TagInputRequest(text=text))


def _find(response, surface: str):
    return next(t for t in response.tokens if t.surface == surface)


def test_spanish_only_word_gets_unclickable_english_column():
    tok = _find(_run("Quiero comer pan", spanish_words={"comer"}), "comer")
    assert tok.is_spanish is True
    assert [(c.language, c.clickable) for c in tok.columns] == [("en", False)]


def test_english_only_word_gets_clickable_spanish_column():
    tok = _find(_run("I want pan", english_words={"want"}), "want")
    assert tok.is_spanish is False
    assert [(c.language, c.clickable) for c in tok.columns] == [("es", True)]


def test_word_valid_in_both_languages_gets_both_columns_independently():
    # "once" is Spanish for "eleven" and also an English word ("on one
    # occasion") - both readings are real, so both should show up, one
    # clickable (the English->Spanish swap) and one not (the Spanish
    # gloss, already correct as typed).
    tok = _find(_run("Nos vemos once veces", spanish_words={"once"}, english_words={"once"}), "once")
    assert {(c.language, c.clickable) for c in tok.columns} == {("en", False), ("es", True)}


def test_word_in_neither_dictionary_falls_back_to_single_morphological_guess():
    tok = _find(_run("Quiero comprar xyzzy"), "xyzzy")
    assert len(tok.columns) == 1


def test_punctuation_gets_no_columns():
    tok = _find(_run("Hola,"), ",")
    assert tok.columns == []
