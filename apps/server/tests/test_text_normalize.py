from app.translate.text_normalize import strip_accents


def test_strips_acute_accents():
    assert strip_accents("cómo") == "como"
    assert strip_accents("está") == "esta"


def test_strips_diaeresis():
    assert strip_accents("pingüino") == "pinguino"


def test_preserves_enye_as_a_distinct_letter():
    # "ñ" is its own letter in Spanish, not an accented "n" - stripping it
    # would collide real word pairs that differ only by ñ/n, e.g. "año"
    # ("year") vs "ano" (a very different word).
    assert strip_accents("año") == "año"
    assert strip_accents("compañero") == "compañero"


def test_leaves_unaccented_text_unchanged():
    assert strip_accents("hello") == "hello"
