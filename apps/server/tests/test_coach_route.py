import asyncio
import json
from unittest.mock import patch

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db import Base
from app.routes.translate import coach_draft
from app.schemas import CoachDraftRequest
from app.translate.llm_translate import TranslationUnavailableError


def _session():
    engine = create_engine("sqlite:///:memory:")
    from app import models  # noqa: F401  (register tables on Base)

    Base.metadata.create_all(bind=engine)
    return sessionmaker(bind=engine)()


async def _collect(body_iterator):
    chunks = []
    async for chunk in body_iterator:
        chunks.append(chunk)
    return chunks


def _run(text: str, events: list[tuple]):
    # The route function only builds the StreamingResponse - its generator
    # is lazy and isn't actually iterated (and coach_draft_stream isn't
    # actually called) until body_iterator is consumed below, so the
    # patch has to stay active for that too, not just for the route call.
    with patch("app.routes.translate.llm_translate.coach_draft_stream", return_value=iter(events)):
        response = coach_draft(CoachDraftRequest(text=text, session_id="s1"), session=_session())
        raw = "".join(asyncio.run(_collect(response.body_iterator)))
    # Parse the SSE wire format ("event: <name>\ndata: <json>\n\n") back
    # into (name, parsed_json) pairs.
    parsed = []
    for block in raw.split("\n\n"):
        if not block.strip():
            continue
        lines = block.splitlines()
        name = next(line[len("event:") :].strip() for line in lines if line.startswith("event:"))
        data = next(line[len("data:") :].strip() for line in lines if line.startswith("data:"))
        parsed.append((name, json.loads(data)))
    return response, parsed


def test_clean_verdict_sse_sequence():
    events = [
        ("verdict", "clean"),
        ("meaning_chunk", "Hello "),
        ("meaning_chunk", "there"),
        ("meaning_complete", "Hello there"),
    ]
    response, parsed = _run("hola", events)

    assert response.media_type == "text/event-stream"
    assert [name for name, _ in parsed] == ["verdict", "meaning_chunk", "meaning_chunk", "meaning_complete"]
    assert parsed[0][1] == {"verdict": "clean"}
    assert parsed[1][1] == {"delta": "Hello "}
    assert parsed[3][1] == {"meaning": "Hello there"}


def test_minor_verdict_sse_sequence():
    events = [
        ("verdict", "minor"),
        ("meaning_chunk", "You mean hello"),
        ("meaning_complete", "You mean hello"),
        ("suggestion_chunk", "Hola, "),
        ("suggestion_chunk", "¿cómo estás?"),
        ("suggestion_complete", "Hola, ¿cómo estás?"),
    ]
    response, parsed = _run("hola como estas", events)

    assert [name for name, _ in parsed] == [
        "verdict",
        "meaning_chunk",
        "meaning_complete",
        "suggestion_chunk",
        "suggestion_chunk",
        "suggestion_complete",
    ]
    assert parsed[0][1] == {"verdict": "minor"}
    assert parsed[-1][1] == {"suggestion": "Hola, ¿cómo estás?"}


def test_fix_verdict_sse_sequence_is_interleaved_per_option():
    events = [
        ("verdict", "fix"),
        ("meaning_chunk", "I want to go to the beach"),
        ("meaning_complete", "I want to go to the beach"),
        ("option_spanish_chunk", 0, "Quiero ir a la playa."),
        ("option_spanish_complete", 0, "Quiero ir a la playa."),
        ("option_english_chunk", 0, "I want to go to the beach."),
        ("option_english_complete", 0, "I want to go to the beach."),
        ("option_spanish_chunk", 1, "Quiero ir a la playa, ¿va?"),
        ("option_spanish_complete", 1, "Quiero ir a la playa, ¿va?"),
        ("option_english_chunk", 1, "I want to go to the beach, right?"),
        ("option_english_complete", 1, "I want to go to the beach, right?"),
    ]
    response, parsed = _run("quiero ir playa", events)

    names = [name for name, _ in parsed]
    assert names == [
        "verdict",
        "meaning_chunk",
        "meaning_complete",
        "option_chunk",
        "option_complete",
        "option_chunk",
        "option_complete",
        "option_chunk",
        "option_complete",
        "option_chunk",
        "option_complete",
    ]
    # Interleaved order confirmed: option 0's spanish+english complete
    # before option 1's spanish even starts, not all-spanish-then-all-
    # english.
    option_complete_events = [data for name, data in parsed if name == "option_complete"]
    assert option_complete_events == [
        {"index": 0, "field": "spanish", "text": "Quiero ir a la playa."},
        {"index": 0, "field": "english", "text": "I want to go to the beach."},
        {"index": 1, "field": "spanish", "text": "Quiero ir a la playa, ¿va?"},
        {"index": 1, "field": "english", "text": "I want to go to the beach, right?"},
    ]


def test_coach_failure_yields_an_error_event():
    with patch(
        "app.routes.translate.llm_translate.coach_draft_stream",
        side_effect=TranslationUnavailableError("model is down"),
    ):
        response = coach_draft(CoachDraftRequest(text="hola", session_id="s1"), session=_session())
        raw = "".join(asyncio.run(_collect(response.body_iterator)))
    assert "event: error" in raw
    assert "model is down" in raw
