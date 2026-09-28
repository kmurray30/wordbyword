from app.translate.dictionary import lookup


def test_lookup_hit_returns_curated_candidates():
    candidates = lookup("estar")
    assert candidates
    assert candidates[0]["translation"] == "to be"


def test_lookup_is_accent_insensitive():
    # "como" is the curated dictionary's own key, already covering both the
    # accented ("cómo", question word) and unaccented ("como", comparison)
    # readings - querying with the accent present must still hit it.
    assert lookup("cómo") == lookup("como")
    assert lookup("cómo")


def test_lookup_is_case_insensitive():
    assert lookup("ESTAR") == lookup("estar")


def test_lookup_miss_returns_empty_list():
    assert lookup("xyzzy") == []
