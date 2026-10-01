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


def test_every_real_spanish_word_gets_a_gloss_not_just_frequent_ones():
    # "tormenta" (storm) and "mío" (mine) are ordinary Spanish words outside
    # the small ~300-word frequency list that app.translate.lemmatizer's
    # is_spanish heuristic was built for tagging the *learner's* input, not
    # gating what the agent's own (always-Spanish) reply gets glossed for.
    # Neither is in the curated dictionary, so glossing them falls through
    # to Argos MT - mocked here (as the rest of the suite does) rather than
    # depending on the real language package being installed. The LLM-based
    # gloss_reply call fails here (no model server in tests), exercising the
    # fallback to the older per-word mechanism.
    reply = "Hay una tormenta y el paraguas es mío."
    with (
        patch("app.routes.chat.model_chat", return_value=reply),
        patch("app.translate.llm_translate.model_chat", side_effect=ModelServerUnavailableError("down")),
        patch("app.translate.service.mt.translate_word", return_value="storm"),
    ):
        result = take_turn(ChatTurnRequest(session_id="t1", message="hola"), session=_session())

    by_surface = {t.surface.lower(): t for t in result.tokens}
    assert by_surface["tormenta"].gloss
    assert by_surface["mío"].gloss
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


def test_word_gloss_prefers_llm_context_over_dictionary_lookup():
    # gloss_reply sees the whole sentence, so it can gloss an ambiguous word
    # correctly for how it's actually used here - "banco" as "bank" (the
    # financial institution) - whereas the old dictionary/MT lookup has no
    # sentence context and could just as easily land on "bench". The LLM's
    # answer, and its note, should win when it's available.
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


def test_word_missing_from_llm_map_falls_back_to_dictionary():
    reply = "El gato es grande."
    # The LLM's word map only covers one of the two content words - the
    # other should still get a gloss via the old mechanism, not come back
    # empty.
    gloss_json = '{"translation": "The cat is big.", "words": {"grande": {"gloss": "big", "note": ""}}}'
    with (
        patch("app.routes.chat.model_chat", return_value=reply),
        patch("app.translate.llm_translate.model_chat", return_value=gloss_json),
    ):
        result = take_turn(ChatTurnRequest(session_id="t4", message="hola"), session=_session())

    by_surface = {t.surface.lower(): t for t in result.tokens}
    assert by_surface["grande"].gloss == "big"
    assert by_surface["gato"].gloss == "cat"  # from the curated dictionary
    assert by_surface["gato"].note == ""  # no LLM data available for it
