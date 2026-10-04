from unittest.mock import patch

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.chat.openai_client import ModelServerUnavailableError
from app.db import Base
from app.routes.chat import get_history, take_turn
from app.schemas import ChatTurnRequest


def _session():
    engine = create_engine("sqlite:///:memory:")
    from app import models  # noqa: F401  (register tables on Base)

    Base.metadata.create_all(bind=engine)
    return sessionmaker(bind=engine)()


def test_gloss_reply_failure_leaves_every_word_with_no_gloss():
    # No dictionary/MT fallback anymore (see app.routes.chat) - when the
    # LLM-based gloss_reply call fails outright (no model server in tests),
    # every word just gets an empty gloss rather than falling through to a
    # second translation mechanism.
    reply = "Hay una tormenta y el paraguas es mío."
    with (
        patch("app.routes.chat.model_chat", return_value=reply),
        patch("app.translate.llm_translate.model_chat", side_effect=ModelServerUnavailableError("down")),
    ):
        result = take_turn(ChatTurnRequest(session_id="t1", message="hola"), session=_session())

    by_surface = {t.surface.lower(): t for t in result.tokens}
    assert by_surface["tormenta"].gloss == ""
    assert by_surface["mío"].gloss == ""
    assert result.translation == ""


def test_punctuation_still_gets_no_gloss():
    with (
        patch("app.routes.chat.model_chat", return_value="Hola, ¿qué tal?"),
        patch("app.translate.llm_translate.model_chat", side_effect=ModelServerUnavailableError("down")),
    ):
        result = take_turn(ChatTurnRequest(session_id="t2", message="hola"), session=_session())

    by_surface = {t.surface: t for t in result.tokens}
    assert by_surface[","].gloss == ""
    assert by_surface["¿"].gloss == ""


def test_word_gloss_uses_llm_context():
    # gloss_reply sees the whole sentence, so it can gloss an ambiguous word
    # correctly for how it's actually used here - "banco" as "bank" (the
    # financial institution), not "bench" (its other common sense).
    reply = "El banco está cerrado."
    gloss_json = (
        '{"translation": "The bank is closed.", "spans": ['
        '{"surface": "El", "gloss": "the", "note": ""}, '
        '{"surface": "banco", "gloss": "bank", "note": "financial institution, not a park bench"}, '
        '{"surface": "está", "gloss": "is", "note": ""}, '
        '{"surface": "cerrado", "gloss": "closed", "note": ""}]}'
    )
    with (
        patch("app.routes.chat.model_chat", return_value=reply),
        patch("app.translate.llm_translate.model_chat", return_value=gloss_json),
    ):
        result = take_turn(ChatTurnRequest(session_id="t3", message="hola"), session=_session())

    by_surface = {t.surface.lower(): t for t in result.tokens}
    assert by_surface["banco"].gloss == "bank"
    assert by_surface["banco"].note == "financial institution, not a park bench"
    assert by_surface["cerrado"].gloss == "closed"
    assert by_surface["cerrado"].note == ""
    assert result.translation == "The bank is closed."


def test_punctuation_survives_a_history_reload():
    # Punctuation tokens used to only ever exist in the live ChatTurnResponse
    # (annotations) - never persisted as a MessageToken row - so GET
    # /chat/history (which rebuilds tokens purely from those rows) silently
    # dropped every punctuation mark on the very next reload.
    session = _session()
    with (
        patch("app.routes.chat.model_chat", return_value="Hola, ¿qué tal?"),
        patch("app.translate.llm_translate.model_chat", side_effect=ModelServerUnavailableError("down")),
    ):
        take_turn(ChatTurnRequest(session_id="t5", message="hola"), session=session)

    history = get_history(session_id="t5", session=session)
    assistant_message = next(m for m in history.messages if m.role == "assistant")
    surfaces = [t.surface for t in assistant_message.tokens]
    assert "," in surfaces
    assert "¿" in surfaces
    assert "?" in surfaces


def test_note_and_literal_survive_a_history_reload():
    # note/literal used to only ever exist in the live ChatTurnResponse
    # (annotations) - not persisted on the MessageToken row - so GET
    # /chat/history (which rebuilds tokens purely from those rows) silently
    # flattened a reloaded message's words back down to their bare gloss,
    # dropping any disambiguation note and idiom word-by-word breakdown.
    session = _session()
    reply = "Tenlo en cuenta."
    gloss_json = (
        '{"translation": "Keep it in mind.", "spans": ['
        '{"surface": "Tenlo en cuenta", "gloss": "keep it in mind", "note": "imperative form", '
        '"literal": "ten (have) + lo (it) + en cuenta (in mind)"}]}'
    )
    with (
        patch("app.routes.chat.model_chat", return_value=reply),
        patch("app.translate.llm_translate.model_chat", return_value=gloss_json),
    ):
        take_turn(ChatTurnRequest(session_id="t7", message="hola"), session=session)

    history = get_history(session_id="t7", session=session)
    assistant_message = next(m for m in history.messages if m.role == "assistant")
    by_surface = {t.surface.lower(): t for t in assistant_message.tokens}
    assert by_surface["tenlo"].note == "imperative form"
    assert by_surface["tenlo"].literal == "ten (have) + lo (it) + en cuenta (in mind)"
    assert by_surface["cuenta"].note == "imperative form"
    assert by_surface["cuenta"].literal == "ten (have) + lo (it) + en cuenta (in mind)"


def test_word_missing_from_llm_spans_gets_empty_gloss():
    reply = "El gato es grande."
    # The LLM's span list only covers one of the two content words - the
    # other should come back with an empty gloss, not fall through to a
    # dictionary/MT fallback (removed - see app.routes.chat).
    gloss_json = '{"translation": "The cat is big.", "spans": [{"surface": "grande", "gloss": "big", "note": ""}]}'
    with (
        patch("app.routes.chat.model_chat", return_value=reply),
        patch("app.translate.llm_translate.model_chat", return_value=gloss_json),
    ):
        result = take_turn(ChatTurnRequest(session_id="t4", message="hola"), session=_session())

    by_surface = {t.surface.lower(): t for t in result.tokens}
    assert by_surface["grande"].gloss == "big"
    assert by_surface["gato"].gloss == ""
    assert by_surface["gato"].note == ""


def test_multi_word_group_shares_one_gloss_across_both_tokens():
    # "el tuyo" ("yours") glossed as a single two-word span - every token
    # inside that span's offset range should get the SAME gloss, not just
    # the first word (the exact "y"/"mi"/"el tuyo" bug this round fixes).
    reply = "Ese bolso es el tuyo."
    gloss_json = (
        '{"translation": "That bag is yours.", "spans": ['
        '{"surface": "Ese", "gloss": "that", "note": ""}, '
        '{"surface": "bolso", "gloss": "bag", "note": ""}, '
        '{"surface": "es", "gloss": "is", "note": ""}, '
        '{"surface": "el tuyo", "gloss": "yours", "note": ""}]}'
    )
    with (
        patch("app.routes.chat.model_chat", return_value=reply),
        patch("app.translate.llm_translate.model_chat", return_value=gloss_json),
    ):
        result = take_turn(ChatTurnRequest(session_id="t6", message="hola"), session=_session())

    by_surface = {t.surface.lower(): t for t in result.tokens}
    assert by_surface["el"].gloss == "yours"
    assert by_surface["tuyo"].gloss == "yours"
