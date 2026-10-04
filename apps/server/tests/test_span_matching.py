from app.translate.span_matching import match_spans


def test_basic_single_word_match():
    matched = match_spans("Quiero comer pan", [{"surface": "comer", "gloss": "to eat"}])
    assert len(matched) == 1
    m = matched[0]
    assert (m.start, m.end) == (7, 12)
    assert m.surface == "comer"


def test_accent_insensitive_match():
    # Draft has the accent, span surface doesn't (or vice versa) - either
    # way it should still resolve to the right offsets.
    matched = match_spans("¿Cómo estás?", [{"surface": "como", "gloss": "how"}])
    assert len(matched) == 1
    assert matched[0].surface == "Cómo"


def test_case_insensitive_match():
    matched = match_spans("Hola Amigo", [{"surface": "amigo", "gloss": "friend"}])
    assert len(matched) == 1
    assert matched[0].surface == "Amigo"


def test_multi_word_group_spans_a_space():
    text = "voy a echar de menos esto"
    matched = match_spans(text, [{"surface": "echar de menos", "gloss": "to miss"}])
    assert len(matched) == 1
    assert text[matched[0].start : matched[0].end] == "echar de menos"


def test_literal_breakdown_passes_through_for_multi_word_span():
    text = "voy a echar de menos esto"
    matched = match_spans(
        text,
        [{"surface": "echar de menos", "gloss": "to miss", "literal": "echar (to throw) + de menos (of less)"}],
    )
    assert len(matched) == 1
    assert matched[0].literal == "echar (to throw) + de menos (of less)"


def test_literal_breakdown_defaults_to_empty():
    matched = match_spans("comer", [{"surface": "comer", "gloss": "to eat"}])
    assert matched[0].literal == ""


def test_repeated_surface_text_resolves_each_occurrence_separately():
    text = "the cat and the cat"
    matched = match_spans(text, [{"surface": "cat", "gloss": "a"}, {"surface": "cat", "gloss": "b"}])
    assert len(matched) == 2
    first, second = matched
    assert text[first.start : first.end] == "cat"
    assert text[second.start : second.end] == "cat"
    assert first.start < second.start
    assert first.gloss == "a"
    assert second.gloss == "b"


def test_unmatchable_span_is_dropped_and_matching_continues():
    text = "hola amigo"
    matched = match_spans(
        text,
        [
            {"surface": "this is not in the text", "gloss": "x"},
            {"surface": "amigo", "gloss": "friend"},
        ],
    )
    assert len(matched) == 1
    assert matched[0].surface == "amigo"


def test_out_of_order_span_still_resolves_via_fallback_search():
    # The model occasionally returns spans slightly out of textual order -
    # a span whose surface appears BEFORE the cursor should still resolve
    # via the fallback full-text search, as long as it doesn't collide with
    # an already-matched span.
    text = "amigo y gato"
    matched = match_spans(text, [{"surface": "gato", "gloss": "cat"}, {"surface": "amigo", "gloss": "friend"}])
    assert len(matched) == 1
    assert matched[0].surface == "gato"


def test_overlapping_out_of_order_span_is_dropped():
    text = "el gato negro"
    matched = match_spans(
        text,
        [{"surface": "gato negro", "gloss": "black cat"}, {"surface": "gato", "gloss": "cat"}],
    )
    assert len(matched) == 1
    assert matched[0].surface == "gato negro"


def test_empty_or_whitespace_surface_is_dropped_without_crashing():
    matched = match_spans("hola", [{"surface": "", "gloss": "x"}, {"surface": "   ", "gloss": "y"}])
    assert matched == []
