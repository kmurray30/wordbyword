from unittest.mock import patch

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

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
    # depending on the real language package being installed.
    reply = "Hay una tormenta y el paraguas es mío."
    with (
        patch("app.routes.chat.llama_chat", return_value=reply),
        patch("app.translate.service.mt.translate_word", return_value="storm"),
    ):
        result = take_turn(ChatTurnRequest(session_id="t1", message="hola"), session=_session())

    by_surface = {t.surface.lower(): t for t in result.tokens}
    assert by_surface["tormenta"].gloss
    assert by_surface["mío"].gloss


def test_punctuation_still_gets_no_gloss():
    with patch("app.routes.chat.llama_chat", return_value="Hola, ¿qué tal?"):
        result = take_turn(ChatTurnRequest(session_id="t2", message="hola"), session=_session())

    by_surface = {t.surface: t for t in result.tokens}
    assert by_surface[","].gloss == ""
    assert by_surface["¿"].gloss == ""
