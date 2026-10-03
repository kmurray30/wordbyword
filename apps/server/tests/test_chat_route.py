from unittest.mock import patch

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.chat.llama_client import ModelServerUnavailableError
from app.db import Base
from app.routes.chat import take_turn
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
        '{"translation": "The bank is closed.", "words": {'
        '"banco": {"gloss": "bank", "note": "financial institution, not a park bench"}, '
        '"cerrado": {"gloss": "closed", "note": ""}}}'
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


def test_word_missing_from_llm_map_gets_empty_gloss():
    reply = "El gato es grande."
    # The LLM's word map only covers one of the two content words - the
    # other should come back with an empty gloss, not fall through to a
    # dictionary/MT fallback (removed - see app.routes.chat).
    gloss_json = '{"translation": "The cat is big.", "words": {"grande": {"gloss": "big", "note": ""}}}'
    with (
        patch("app.routes.chat.model_chat", return_value=reply),
        patch("app.translate.llm_translate.model_chat", return_value=gloss_json),
    ):
        result = take_turn(ChatTurnRequest(session_id="t4", message="hola"), session=_session())

    by_surface = {t.surface.lower(): t for t in result.tokens}
    assert by_surface["grande"].gloss == "big"
    assert by_surface["gato"].gloss == ""
    assert by_surface["gato"].note == ""
