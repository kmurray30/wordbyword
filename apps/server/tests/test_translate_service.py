from unittest.mock import patch

from app.translate.service import gloss, word_candidates


def test_dictionary_hit_returns_curated_candidates_with_descriptions():
    candidates = word_candidates("estar", source_lang="es", target_lang="en")
    assert any(c["translation"] == "to be" for c in candidates)
    assert all(c["description"] for c in candidates)


def test_dictionary_miss_falls_back_to_mt():
    with patch("app.translate.service.mt.translate_word", return_value="cheese") as mocked:
        candidates = word_candidates("queso", source_lang="es", target_lang="en")
    mocked.assert_called_once_with("queso", "es", "en")
    assert candidates == [{"translation": "cheese", "description": ""}]


def test_gloss_returns_first_candidate_translation():
    with patch("app.translate.service.mt.translate_word", return_value="cheese"):
        assert gloss("queso", "es", "en") == "cheese"


def test_mt_fallback_strips_sentence_capitalization_and_punctuation():
    # Argos's word lookup is its sentence-level MT run on a lone word, so it
    # can come back as "Hola." for "hello" - fine as a standalone gloss, but
    # wrong once spliced into the middle of an existing sentence (the input
    # box's word-hover replace feature does exactly that).
    with patch("app.translate.service.mt.translate_word", return_value="Hola."):
        candidates = word_candidates("hello", source_lang="en", target_lang="es")
    assert candidates == [{"translation": "hola", "description": ""}]


def test_mt_fallback_preserves_all_caps_words():
    with patch("app.translate.service.mt.translate_word", return_value="OK."):
        candidates = word_candidates("okay", source_lang="en", target_lang="es")
    assert candidates == [{"translation": "OK", "description": ""}]
