from unittest.mock import patch

from app.routes.translate import gloss_spans, tag_input
from app.schemas import GlossSpansRequest, TagInputRequest
from app.translate.llm_translate import TranslationUnavailableError


def test_tokens_are_populated_independent_of_llm_success():
    # tag_input is the cheap spaCy-only pass now - no LLM call at all.
    response = tag_input(TagInputRequest(text="Quiero comer pan"))
    surfaces = {t.surface for t in response.tokens}
    assert surfaces == {"Quiero", "comer", "pan"}


def _run_gloss(text: str, raw_spans: list[dict[str, str]], translation: str = ""):
    with patch("app.routes.translate.llm_translate.tag_draft", return_value=(translation, raw_spans)):
        return gloss_spans(GlossSpansRequest(text=text))


def test_clickable_span_when_llm_gives_a_replacement():
    response = _run_gloss(
        "I want pan",
        [{"surface": "want", "gloss": "deseo", "note": "", "translation": "quiero"}],
    )
    span = next(s for s in response.spans if s.surface == "want")
    assert span.clickable is True
    assert span.candidates[0].translation == "quiero"


def test_unclickable_span_when_already_correct_spanish():
    response = _run_gloss(
        "Quiero comer pan",
        [{"surface": "comer", "gloss": "to eat", "note": "", "translation": ""}],
    )
    span = next(s for s in response.spans if s.surface == "comer")
    assert span.clickable is False
    assert span.candidates[0].translation == "to eat"


def test_multi_word_group_span_matches_contiguous_text():
    response = _run_gloss(
        "voy a echar de menos esto",
        [{"surface": "echar de menos", "gloss": "to miss (someone/something)", "note": "", "translation": ""}],
    )
    span = next(s for s in response.spans if s.surface == "echar de menos")
    assert (span.start, span.end) == (len("voy a "), len("voy a echar de menos"))


def test_span_with_nothing_usable_is_dropped():
    response = _run_gloss("hola", [{"surface": "hola", "gloss": "", "note": "", "translation": ""}])
    assert response.spans == []


def test_unmatchable_span_is_dropped_not_crashed_on():
    response = _run_gloss(
        "hola amigo",
        [{"surface": "this text is not in the draft", "gloss": "x", "note": "", "translation": "y"}],
    )
    assert response.spans == []


def test_llm_failure_yields_no_spans():
    with patch("app.routes.translate.llm_translate.tag_draft", side_effect=TranslationUnavailableError("down")):
        response = gloss_spans(GlossSpansRequest(text="Quiero comer pan"))
    assert response.spans == []


def test_alternate_gloss_passed_through():
    response = _run_gloss(
        "once veces",
        [
            {
                "surface": "once",
                "gloss": "eleven",
                "note": "",
                "translation": "",
                "alternate_gloss": "once, as in 'at one time' - English word",
            }
        ],
    )
    span = next(s for s in response.spans if s.surface == "once")
    assert span.alternate_gloss == "once, as in 'at one time' - English word"
