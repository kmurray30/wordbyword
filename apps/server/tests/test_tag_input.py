from unittest.mock import patch

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db import Base
from app.routes.translate import tag_input
from app.schemas import TagInputRequest
from app.translate.llm_translate import TranslationUnavailableError


def _session():
    engine = create_engine("sqlite:///:memory:")
    from app import models  # noqa: F401  (register tables on Base)

    Base.metadata.create_all(bind=engine)
    return sessionmaker(bind=engine)()


def _run(text: str, raw_spans: list[dict[str, str]], translation: str = ""):
    with patch("app.routes.translate.llm_translate.tag_draft", return_value=(translation, raw_spans)):
        return tag_input(TagInputRequest(text=text), session=_session())


def test_tokens_are_populated_independent_of_llm_success():
    # tokens comes from the cheap spaCy pass, computed on every call
    # regardless of whether the LLM-backed spans succeed.
    response = _run("Quiero comer pan", [])
    surfaces = {t.surface for t in response.tokens}
    assert surfaces == {"Quiero", "comer", "pan"}


def test_clickable_span_when_llm_gives_a_replacement():
    response = _run(
        "I want pan",
        [{"surface": "want", "gloss": "deseo", "note": "", "translation": "quiero"}],
    )
    span = next(s for s in response.spans if s.surface == "want")
    assert span.clickable is True
    assert span.candidates[0].translation == "quiero"


def test_unclickable_span_when_already_correct_spanish():
    response = _run(
        "Quiero comer pan",
        [{"surface": "comer", "gloss": "to eat", "note": "", "translation": ""}],
    )
    span = next(s for s in response.spans if s.surface == "comer")
    assert span.clickable is False
    assert span.candidates[0].translation == "to eat"


def test_multi_word_group_span_matches_contiguous_text():
    response = _run(
        "voy a echar de menos esto",
        [{"surface": "echar de menos", "gloss": "to miss (someone/something)", "note": "", "translation": ""}],
    )
    span = next(s for s in response.spans if s.surface == "echar de menos")
    assert (span.start, span.end) == (len("voy a "), len("voy a echar de menos"))


def test_span_with_nothing_usable_is_dropped():
    response = _run("hola", [{"surface": "hola", "gloss": "", "note": "", "translation": ""}])
    assert response.spans == []


def test_unmatchable_span_is_dropped_not_crashed_on():
    response = _run(
        "hola amigo",
        [{"surface": "this text is not in the draft", "gloss": "x", "note": "", "translation": "y"}],
    )
    assert response.spans == []


def test_llm_failure_yields_no_spans_but_keeps_tokens():
    with patch("app.routes.translate.llm_translate.tag_draft", side_effect=TranslationUnavailableError("down")):
        response = tag_input(TagInputRequest(text="Quiero comer pan"), session=_session())
    assert response.spans == []
    assert len(response.tokens) == 3
