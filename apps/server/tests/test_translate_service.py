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
