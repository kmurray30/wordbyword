from app.translate.lemmatizer import analyze


def test_conjugated_verb_collapses_to_infinitive():
    # es_core_news_sm's POS tagger occasionally mis-tags less common
    # conjugated forms (e.g. 1st-person preterite "comí" as ADJ rather than
    # VERB, which then skips lemmatization) - see README's "known
    # limitations" note. "comimos" is reliably tagged correctly, so we use
    # it here rather than asserting on a form the small model gets wrong.
    tokens = analyze("Nosotros comimos pan ayer.")
    by_surface = {t.surface.lower(): t.lemma for t in tokens}
    assert by_surface["comimos"] == "comer"


def test_known_spanish_word_flagged_as_spanish():
    tokens = analyze("El perro es grande")
    for tok in tokens:
        if tok.surface.lower() == "perro":
            assert tok.is_spanish is True


def test_english_word_dropped_into_spanish_sentence_flagged():
    tokens = analyze("Quiero comprar a house nueva")
    house = [t for t in tokens if t.surface.lower() == "house"]
    assert house, "expected 'house' to be tokenized"
    assert house[0].is_spanish is False


def test_punctuation_not_flagged_as_unknown():
    tokens = analyze("¿Hola, cómo estás?")
    for tok in tokens:
        if not tok.surface.isalpha():
            assert tok.is_spanish is True
