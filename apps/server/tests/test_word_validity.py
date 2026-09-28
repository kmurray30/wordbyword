from app.translate.word_validity import is_valid_english_word, is_valid_spanish_word


def test_once_is_valid_in_both_languages():
    # Spanish "eleven" and English "on one occasion" - the case the dual-
    # column tag-input feature exists for.
    assert is_valid_spanish_word("once") is True
    assert is_valid_english_word("once") is True


def test_ordinary_spanish_word_not_valid_english():
    assert is_valid_spanish_word("perro") is True
    assert is_valid_english_word("perro") is False


def test_ordinary_english_word_not_valid_spanish():
    assert is_valid_english_word("dude") is True
    assert is_valid_spanish_word("dude") is False


def test_case_insensitive():
    assert is_valid_spanish_word("Hola") is True


def test_spanish_word_valid_even_with_accent_dropped():
    # The checker's own dictionary has "área" (accented) but not "area" -
    # a learner who drops the accent should still be recognized.
    assert is_valid_spanish_word("area") is True
