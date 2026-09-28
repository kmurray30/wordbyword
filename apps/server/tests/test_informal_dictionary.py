from app.translate.informal_dictionary import lookup


def test_sup_has_multiple_real_candidates():
    candidates = lookup("sup")
    assert len(candidates) >= 2
    assert all(c["translation"] for c in candidates)
    assert all(c["description"] for c in candidates)


def test_bro_has_multiple_real_candidates_not_just_itself():
    candidates = lookup("bro")
    translations = [c["translation"] for c in candidates]
    assert len(translations) >= 2
    assert any(t != "bro" for t in translations)


def test_lookup_is_case_and_accent_insensitive():
    assert lookup("SUP") == lookup("sup")


def test_lookup_miss_returns_empty_list():
    assert lookup("hello") == []


def test_a_offers_both_gendered_articles():
    translations = [c["translation"] for c in lookup("a")]
    assert "un" in translations
    assert "una" in translations


def test_once_translates_to_one_time_not_the_number():
    candidates = lookup("once")
    assert candidates[0]["translation"] == "una vez"
