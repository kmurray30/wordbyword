from unittest.mock import patch

from app.translate.service import gloss, word_candidates


def test_dictionary_hit_returns_curated_candidates_with_descriptions():
    candidates = word_candidates("estar", source_lang="es", target_lang="en")
    assert any(c["translation"] == "to be" for c in candidates)
    assert all(c["description"] for c in candidates)


def test_wiktionary_hit_returns_before_mt_es_to_en():
    wiktionary_entries = [{"translation": "cheese", "description": "dairy product"}]
    with (
        patch("app.translate.service.wiktionary_dict.lookup_es_to_en", return_value=wiktionary_entries),
        patch("app.translate.service.mt.translate_word") as mocked_mt,
    ):
        candidates = word_candidates("queso", source_lang="es", target_lang="en")
    mocked_mt.assert_not_called()
    assert candidates == wiktionary_entries


def test_wiktionary_hit_returns_before_mt_en_to_es():
    wiktionary_entries = [
        {"translation": "tío", "description": "(informal) close male friend"},
        {"translation": "hermano", "description": "brother, also used affectionately"},
    ]
    with (
        patch("app.translate.service.wiktionary_dict.lookup_en_to_es", return_value=wiktionary_entries),
        patch("app.translate.service.mt.translate_word") as mocked_mt,
    ):
        candidates = word_candidates("bro", source_lang="en", target_lang="es")
    mocked_mt.assert_not_called()
    assert candidates == wiktionary_entries


def test_dictionary_and_wiktionary_miss_falls_back_to_mt():
    with (
        patch("app.translate.service.wiktionary_dict.lookup_es_to_en", return_value=[]),
        patch("app.translate.service.mt.translate_word", return_value="cheese") as mocked,
    ):
        candidates = word_candidates("queso", source_lang="es", target_lang="en")
    mocked.assert_called_once_with("queso", "es", "en")
    assert candidates == [{"translation": "cheese", "description": ""}]


def test_gloss_returns_first_candidate_translation():
    with (
        patch("app.translate.service.wiktionary_dict.lookup_es_to_en", return_value=[]),
        patch("app.translate.service.mt.translate_word", return_value="cheese"),
    ):
        assert gloss("queso", "es", "en") == "cheese"


def test_mt_fallback_strips_sentence_capitalization_and_punctuation():
    # Argos's word lookup is its sentence-level MT run on a lone word, so it
    # can come back as "Hola." for "hello" - fine as a standalone gloss, but
    # wrong once spliced into the middle of an existing sentence (the input
    # box's word-hover replace feature does exactly that).
    with (
        patch("app.translate.service.wiktionary_dict.lookup_en_to_es", return_value=[]),
        patch("app.translate.service.mt.translate_word", return_value="Hola."),
    ):
        candidates = word_candidates("hello", source_lang="en", target_lang="es")
    assert candidates == [{"translation": "hola", "description": ""}]


def test_mt_fallback_preserves_all_caps_words():
    with (
        patch("app.translate.service.wiktionary_dict.lookup_en_to_es", return_value=[]),
        patch("app.translate.service.mt.translate_word", return_value="OK."),
    ):
        candidates = word_candidates("okay", source_lang="en", target_lang="es")
    assert candidates == [{"translation": "OK", "description": ""}]


def test_mt_pass_through_is_reported_as_no_translation_found():
    # Argos has no way to say "I don't know this word" - for an
    # out-of-vocabulary word (slang that never appeared in its formal
    # training corpus) it just echoes the input back unchanged, which would
    # otherwise look exactly like a real (wrong) translation.
    with (
        patch("app.translate.service.wiktionary_dict.lookup_en_to_es", return_value=[]),
        patch("app.translate.service.mt.translate_word", return_value="bro"),
    ):
        candidates = word_candidates("bro", source_lang="en", target_lang="es")
    assert candidates == [{"translation": "", "description": 'no translation found for "bro"'}]


def test_mt_pass_through_detection_is_case_insensitive():
    with (
        patch("app.translate.service.wiktionary_dict.lookup_es_to_en", return_value=[]),
        patch("app.translate.service.mt.translate_word", return_value="Partner."),
    ):
        candidates = word_candidates("partner", source_lang="es", target_lang="en")
    assert candidates == [{"translation": "", "description": 'no translation found for "partner"'}]
