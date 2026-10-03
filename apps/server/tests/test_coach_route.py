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


def _run(text: str, events: list[tuple[str, object]]):
    # The route function only builds the StreamingResponse - its generator
    # is lazy and isn't actually iterated (and coach_draft_stream isn't
    # actually called) until body_iterator is consumed below, so the
    # patch has to stay active for that too, not just for the route call.
    with patch("app.routes.translate.llm_translate.coach_draft_stream", return_value=iter(events)):
        response = coach_draft(CoachDraftRequest(text=text, session_id="s1"), session=_session())
        raw = "".join(asyncio.run(_collect(response.body_iterator)))
    # Parse the SSE wire format ("event: <name>\ndata: <json>\n\n") back
    # into (name, parsed_json) pairs, same shape as the mocked events
    # list above, so assertions can compare structurally.
    parsed = []
    for block in raw.split("\n\n"):
        if not block.strip():
            continue
        lines = block.splitlines()
        name = next(line[len("event:") :].strip() for line in lines if line.startswith("event:"))
        data = next(line[len("data:") :].strip() for line in lines if line.startswith("data:"))
        parsed.append((name, json.loads(data)))
    return response, parsed


def test_core_event_is_streamed_with_options():
    core = {
        "meaning": "I want to go to the beach tomorrow.",
        "feedback": "Good attempt.",
        "options": [("neutral", "Quiero ir a la playa mañana.")],
    }
    response, parsed = _run("quiero playa manana ir", [("core", core)])

    assert response.media_type == "text/event-stream"
    assert len(parsed) == 1
    name, data = parsed[0]
    assert name == "core"
    assert data["meaning"] == "I want to go to the beach tomorrow."
    assert data["options"] == [{"formality": "neutral", "spanish": "Quiero ir a la playa mañana.", "english": "", "spans": []}]


def test_translations_event_matches_spans_against_each_options_own_text():
    core = {
        "meaning": "x",
        "feedback": "",
        "options": [("neutral", "Quiero ir a la playa.")],
    }
    translations = [{"english": "I want to go to the beach.", "spans": [{"surface": "Quiero", "gloss": "I want"}]}]
    response, parsed = _run("quiero ir playa", [("core", core), ("translations", translations)])

    assert [name for name, _ in parsed] == ["core", "translations"]
    option = parsed[1][1]["options"][0]
    assert option["english"] == "I want to go to the beach."
    assert option["spans"][0]["surface"] == "Quiero"
    assert option["spans"][0]["start"] == 0  # "Quiero" is the first word of its own option text


def test_translations_event_short_by_an_option_fills_in_empty():
    # The model's PART 2 can come back with fewer options than PART 1 (a
    # garbled/incomplete entry got dropped) - every PART-1 option must
    # still appear in the translations event, just with nothing attached.
    core = {
        "meaning": "x",
        "feedback": "",
        "options": [("neutral", "Uno."), ("casual", "Dos.")],
    }
    translations = [{"english": "One.", "spans": []}]
    response, parsed = _run("uno dos", [("core", core), ("translations", translations)])

    options = parsed[1][1]["options"]
    assert options[0]["english"] == "One."
    assert options[1]["english"] == ""
    assert options[1]["spans"] == []


def test_core_failure_yields_an_error_event():
    with patch(
        "app.routes.translate.llm_translate.coach_draft_stream",
        side_effect=TranslationUnavailableError("model is down"),
    ):
        response = coach_draft(CoachDraftRequest(text="hola", session_id="s1"), session=_session())
        raw = "".join(asyncio.run(_collect(response.body_iterator)))
    assert "event: error" in raw
    assert "model is down" in raw
