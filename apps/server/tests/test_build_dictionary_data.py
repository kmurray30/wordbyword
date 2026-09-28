from scripts.build_dictionary_data import (
    _augment_en_to_es_from_es_glosses,
    _augment_en_to_es_self_loanwords,
    _format_description,
    _process_english_line,
    _process_spanish_line,
    _short,
)


def _es_line(word, senses):
    import json

    return json.dumps({"word": word, "lang_code": "es", "senses": senses})


def _en_line(word, senses, translations):
    import json

    return json.dumps({"word": word, "lang_code": "en", "senses": senses, "translations": translations})


def test_short_truncates_with_ellipsis():
    assert _short("hi", 10) == "hi"
    assert _short("a" * 20, 10) == "a" * 9 + "…"


def test_format_description_combines_register_tag_and_example():
    desc = _format_description({"informal", "grammar-unrelated-tag"}, example="¡Qué tal, tío!")
    assert desc == "(informal) ¡Qué tal, tío!"


def test_process_spanish_line_includes_common_word():
    line = _es_line("banco", [{"glosses": ["bank (financial institution)"], "tags": [], "examples": []}])
    out = {}
    _process_spanish_line(line, {"banco"}, out)
    assert out["banco"] == [{"translation": "bank (financial institution)", "description": ""}]


def test_process_spanish_line_excludes_uncommon_non_informal_word():
    # Leaves an empty candidate list rather than no key at all - pruned by
    # _build_direction() before the dataset is written out.
    line = _es_line("esotericismo", [{"glosses": ["esotericism"], "tags": [], "examples": []}])
    out = {}
    _process_spanish_line(line, set(), out)
    assert out == {"esotericismo": []}


def test_process_spanish_line_rescues_informal_word_despite_low_frequency():
    line = _es_line("tío", [{"glosses": ["dude, guy"], "tags": ["informal"], "examples": [{"text": "¡Qué tal, tío!"}]}])
    out = {}
    _process_spanish_line(line, set(), out)
    assert out["tío"][0]["translation"] == "dude, guy"
    assert "informal" in out["tío"][0]["description"]


def test_process_spanish_line_caps_candidates_per_word():
    senses = [{"glosses": [f"sense {i}"], "tags": [], "examples": []} for i in range(10)]
    line = _es_line("muchosentidos", senses)
    out = {}
    _process_spanish_line(line, {"muchosentidos"}, out)
    assert len(out["muchosentidos"]) <= 4


def test_process_english_line_skips_word_with_no_spanish_translation():
    line = _en_line("gizmo", [{"glosses": ["a gadget"], "tags": [], "examples": []}], [])
    out = {}
    _process_english_line(line, {"gizmo"}, out)
    assert out == {}


def test_process_english_line_includes_common_word_with_translation():
    line = _en_line(
        "partner",
        [{"glosses": ["a person one is romantically involved with"], "tags": [], "examples": []}],
        [{"code": "es", "word": "pareja", "sense": "a person one is romantically involved with"}],
    )
    out = {}
    _process_english_line(line, {"partner"}, out)
    assert out["partner"] == [
        {"translation": "pareja", "description": "a person one is romantically involved with"}
    ]


def test_process_english_line_rescues_informal_slang_despite_low_frequency():
    line = _en_line(
        "sup",
        [{"glosses": ["informal greeting"], "tags": ["informal"], "examples": []}],
        [{"code": "es", "word": "qué tal", "sense": "informal greeting"}],
    )
    out = {}
    _process_english_line(line, set(), out)
    assert out["sup"][0]["translation"] == "qué tal"


def test_process_english_line_skips_self_translation_when_alternative_exists():
    line = _en_line(
        "bro",
        [{"glosses": ["informal term for a close friend"], "tags": ["informal"], "examples": []}],
        [
            {"code": "es", "word": "tío", "sense": "informal term for a close friend"},
            {"code": "es", "word": "bro", "sense": "informal term for a close friend"},
        ],
    )
    out = {}
    _process_english_line(line, set(), out)
    translations = [c["translation"] for c in out["bro"]]
    assert translations == ["tío"]


def test_process_english_line_keeps_self_translation_when_it_is_the_only_candidate():
    line = _en_line(
        "hotel",
        [{"glosses": ["a place offering lodging"], "tags": [], "examples": []}],
        [{"code": "es", "word": "hotel", "sense": "a place offering lodging"}],
    )
    out = {}
    _process_english_line(line, {"hotel"}, out)
    assert out["hotel"] == [{"translation": "hotel", "description": "a place offering lodging"}]


def test_augment_recovers_word_missing_from_translation_tables():
    # "bro" has no "es" entry in its own translation table (Wiktionary's
    # crowd-sourced tables are often incomplete for slang), but "tío" is
    # independently defined on the ES side with "bro" as its English gloss -
    # the augmentation should recover that reverse relationship.
    es_to_en = {"tío": [{"translation": "bro", "description": "(informal) close male friend"}]}
    en_to_es: dict = {}
    _augment_en_to_es_from_es_glosses(es_to_en, en_to_es, en_freq=set())
    assert en_to_es["bro"] == [{"translation": "tío", "description": "(informal) close male friend"}]


def test_augment_skips_multiword_glosses():
    es_to_en = {"banco": [{"translation": "bank (financial institution)", "description": ""}]}
    en_to_es: dict = {}
    _augment_en_to_es_from_es_glosses(es_to_en, en_to_es, en_freq=set())
    assert en_to_es == {}


def test_augment_skips_uncommon_non_informal_gloss():
    es_to_en = {"esotérico": [{"translation": "esoteric", "description": ""}]}
    en_to_es: dict = {}
    _augment_en_to_es_from_es_glosses(es_to_en, en_to_es, en_freq=set())
    assert en_to_es == {}


def test_augment_does_not_duplicate_existing_translations_table_entry():
    es_to_en = {"pareja": [{"translation": "partner", "description": ""}]}
    en_to_es = {"partner": [{"translation": "pareja", "description": "a person one is romantically involved with"}]}
    _augment_en_to_es_from_es_glosses(es_to_en, en_to_es, en_freq={"partner"})
    assert en_to_es["partner"] == [{"translation": "pareja", "description": "a person one is romantically involved with"}]


def test_augment_self_loanwords_recovers_word_with_its_own_es_entry():
    # Real data, confirmed live: Spanish Wiktionary independently defines
    # "bro" as a Spanish loanword ("bro (a male comrade or friend)",
    # tagged slang) - "bro" itself has NO entry in its own English
    # translation table, so _process_english_line alone never finds it.
    es_to_en = {
        "bro": [
            {"translation": "bro (a male comrade or friend)", "description": "(slang)"},
            {"translation": "bro (used to address a male)", "description": "(slang)"},
        ]
    }
    en_to_es: dict = {}
    _augment_en_to_es_self_loanwords(es_to_en, en_to_es, en_freq={"bro"})
    assert en_to_es["bro"] == [
        {"translation": "bro", "description": "(slang) bro (a male comrade or friend)"}
    ]


def test_augment_self_loanwords_requires_common_or_informal_english():
    es_to_en = {"esoterismo": [{"translation": "esotericism (a body of knowledge)", "description": ""}]}
    en_to_es: dict = {}
    _augment_en_to_es_self_loanwords(es_to_en, en_to_es, en_freq=set())
    assert en_to_es == {}


def test_augment_self_loanwords_does_not_duplicate_existing_entry():
    es_to_en = {"hotel": [{"translation": "a place offering lodging", "description": ""}]}
    en_to_es = {"hotel": [{"translation": "hotel", "description": "a place offering lodging"}]}
    _augment_en_to_es_self_loanwords(es_to_en, en_to_es, en_freq={"hotel"})
    assert en_to_es["hotel"] == [{"translation": "hotel", "description": "a place offering lodging"}]
