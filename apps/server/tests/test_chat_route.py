from unittest.mock import patch

import pytest
from fastapi import HTTPException
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.chat.openai_client import ModelServerUnavailableError
from app.db import Base
from app.models import ChatMessage
from app.routes.chat import get_history, message_glosses, save_user_interpretation, take_turn
from app.schemas import ChatTurnRequest, MessageGlossesRequest, SaveUserInterpretationRequest


def _session():
    engine = create_engine("sqlite:///:memory:")
    from app import models  # noqa: F401  (register tables on Base)

    Base.metadata.create_all(bind=engine)
    return sessionmaker(bind=engine)()


def test_gloss_failure_leaves_every_word_with_no_gloss():
    # No dictionary/MT fallback anymore (see app.routes.chat) - when the
    # LLM-based gloss_message call fails outright (no model server in tests),
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


def test_empty_model_reply_raises_502_instead_of_a_blank_bubble():
    # Live bug: a reasoning-style model can burn its whole generation
    # budget on invisible reasoning tokens and return a 200 with literally
    # no content - not an exception. Previously this silently persisted a
    # vacant assistant turn (empty text, no tokens) with nothing telling
    # the learner anything went wrong.
    session = _session()
    with (
        patch("app.routes.chat.model_chat", return_value="   "),
        patch("app.translate.llm_translate.model_chat", side_effect=ModelServerUnavailableError("down")),
    ):
        with pytest.raises(HTTPException) as exc_info:
            take_turn(ChatTurnRequest(session_id="t-empty", message="hola"), session=session)

    assert exc_info.value.status_code == 502
    # Nothing persisted at all - not even the user's own turn.
    assert session.query(ChatMessage).count() == 0


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
    # gloss_message sees the whole sentence, so it can gloss an ambiguous word
    # correctly for how it's actually used here - "banco" as "bank" (the
    # financial institution), not "bench" (its other common sense).
    reply = "El banco está cerrado."
    gloss_json = (
        '{"translation": "The bank is closed.", "words": ['
        '{"surface": "El", "gloss": "the", "grammar": ""}, '
        '{"surface": "banco", "gloss": "bank", "grammar": ""}, '
        '{"surface": "está", "gloss": "is", "grammar": "estar (to be): he/she/it, present"}, '
        '{"surface": "cerrado", "gloss": "closed", "grammar": ""}], "phrases": []}'
    )
    with (
        patch("app.routes.chat.model_chat", return_value=reply),
        patch("app.translate.llm_translate.model_chat", return_value=gloss_json),
    ):
        result = take_turn(ChatTurnRequest(session_id="t3", message="hola"), session=_session())

    by_surface = {t.surface.lower(): t for t in result.tokens}
    assert by_surface["banco"].gloss == "bank"
    assert by_surface["está"].note == "estar (to be): he/she/it, present"
    assert by_surface["cerrado"].gloss == "closed"
    assert by_surface["cerrado"].note == ""
    assert result.translation == "The bank is closed."
    assert [(w.gloss, w.grammar) for w in result.glosses.words][1:3] == [
        ("bank", ""),
        ("is", "estar (to be): he/she/it, present"),
    ]


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


def test_word_and_phrase_glosses_survive_a_history_reload():
    session = _session()
    reply = "Qué bueno, comí."
    gloss_json = (
        '{"translation": "How wonderful, I ate.", "words": ['
        '{"surface": "Qué", "gloss": "what"}, {"surface": "bueno", "gloss": "good"}, '
        '{"surface": "comí", "gloss": "ate", "grammar": "comer (to eat): I, preterite"}], '
        '"phrases": [{"surface": "Qué bueno", "gloss": "how wonderful", "note": ""}]}'
    )
    with (
        patch("app.routes.chat.model_chat", return_value=reply),
        patch("app.translate.llm_translate.model_chat", return_value=gloss_json),
    ):
        take_turn(ChatTurnRequest(session_id="t7", message="hola"), session=session)

    history = get_history(session_id="t7", session=session)
    assistant_message = next(m for m in history.messages if m.role == "assistant")
    glosses = assistant_message.glosses
    assert [(reply[w.start : w.end], w.gloss) for w in glosses.words] == [("Qué", "what"), ("bueno", "good"), ("comí", "ate")]
    assert glosses.words[2].grammar == "comer (to eat): I, preterite"
    assert [(reply[p.start : p.end], p.gloss) for p in glosses.phrases] == [("Qué bueno", "how wonderful")]
    by_surface = {t.surface.lower(): t for t in assistant_message.tokens}
    assert by_surface["comí"].note == "comer (to eat): I, preterite"


def test_word_missing_from_llm_spans_gets_empty_gloss():
    reply = "El gato es grande."
    # The LLM's span list only covers one of the two content words - the
    # other should come back with an empty gloss, not fall through to a
    # dictionary/MT fallback (removed - see app.routes.chat).
    gloss_json = '{"translation": "The cat is big.", "words": [{"surface": "grande", "gloss": "big"}], "phrases": []}'
    with (
        patch("app.routes.chat.model_chat", return_value=reply),
        patch("app.translate.llm_translate.model_chat", return_value=gloss_json),
    ):
        result = take_turn(ChatTurnRequest(session_id="t4", message="hola"), session=_session())

    by_surface = {t.surface.lower(): t for t in result.tokens}
    assert by_surface["grande"].gloss == "big"
    assert by_surface["gato"].gloss == ""
    assert by_surface["gato"].note == ""


def test_turn_response_carries_the_user_messages_own_id():
    # The frontend fetches the user message's own native/target
    # interpretation live (POST /translate/interpret), independent of this
    # call, and needs this id to persist that result back (POST
    # /chat/message/interpretation) once both are known - see
    # save_user_interpretation below.
    session = _session()
    with (
        patch("app.routes.chat.model_chat", return_value="Hola"),
        patch("app.translate.llm_translate.model_chat", side_effect=ModelServerUnavailableError("down")),
    ):
        result = take_turn(ChatTurnRequest(session_id="t8", message="hola"), session=session)

    user_row = session.query(ChatMessage).filter(ChatMessage.role == "user").one()
    assert result.user_message_id == user_row.id


def test_save_user_interpretation_persists_and_survives_a_history_reload():
    session = _session()
    with (
        patch("app.routes.chat.model_chat", return_value="Hola"),
        patch("app.translate.llm_translate.model_chat", side_effect=ModelServerUnavailableError("down")),
    ):
        result = take_turn(ChatTurnRequest(session_id="t9", message="hola amigo"), session=session)

    save_user_interpretation(
        SaveUserInterpretationRequest(message_id=result.user_message_id, native="Hello, friend.", target="Hola, amigo."),
        session=session,
    )

    history = get_history(session_id="t9", session=session)
    user_message = next(m for m in history.messages if m.role == "user")
    assert user_message.native == "Hello, friend."
    assert user_message.target == "Hola, amigo."


def test_save_interpretation_backfills_an_assistant_replys_translation():
    session = _session()
    with (
        patch("app.routes.chat.model_chat", return_value="Hola"),
        patch("app.translate.llm_translate.model_chat", side_effect=ModelServerUnavailableError("down")),
    ):
        result = take_turn(ChatTurnRequest(session_id="t10", message="hola"), session=session)

    save_user_interpretation(
        SaveUserInterpretationRequest(message_id=result.message_id, native="Hello", target="ignored"),
        session=session,
    )
    history = get_history(session_id="t10", session=session)
    reply = next(m for m in history.messages if m.role == "assistant")
    assert reply.native == "Hello"
    assert reply.target == ""


def test_save_interpretation_404s_for_an_unknown_message():
    with pytest.raises(HTTPException) as exc_info:
        save_user_interpretation(
            SaveUserInterpretationRequest(message_id=9999, native="x", target="y"),
            session=_session(),
        )
    assert exc_info.value.status_code == 404


def test_assistant_reply_translation_is_saved_with_it():
    gloss_json = '{"translation": "The cat is big.", "words": [{"surface": "grande", "gloss": "big"}], "phrases": []}'
    session = _session()
    with (
        patch("app.routes.chat.model_chat", return_value="El gato es grande."),
        patch("app.translate.llm_translate.model_chat", return_value=gloss_json),
    ):
        take_turn(ChatTurnRequest(session_id="t11", message="hola"), session=session)

    history = get_history(session_id="t11", session=session)
    reply = next(m for m in history.messages if m.role == "assistant")
    assert reply.native == "The cat is big."


def test_message_glosses_are_made_once_for_the_learners_message_and_saved():
    session = _session()
    with (
        patch("app.routes.chat.model_chat", return_value="Hola"),
        patch("app.translate.llm_translate.model_chat", side_effect=ModelServerUnavailableError("down")),
    ):
        result = take_turn(ChatTurnRequest(session_id="t12", message="me gusta swimming"), session=session)

    gloss_json = (
        '{"words": [{"surface": "me", "gloss": "to me"}, {"surface": "gusta", "gloss": "pleases", '
        '"grammar": "gustar (to please): it, present"}, {"surface": "swimming", "gloss": "nadar"}], '
        '"phrases": [{"surface": "me gusta", "gloss": "I like"}]}'
    )
    with patch("app.translate.llm_translate.model_chat", return_value=gloss_json) as mock_chat:
        glosses = message_glosses(MessageGlossesRequest(message_id=result.user_message_id), session=session)
        again = message_glosses(MessageGlossesRequest(message_id=result.user_message_id), session=session)

    assert mock_chat.call_count == 1
    assert "learner wrote" in mock_chat.call_args.args[0][0]["content"]
    assert [w.gloss for w in glosses.words] == ["to me", "pleases", "nadar"]
    assert [(p.start, p.end, p.gloss) for p in glosses.phrases] == [(0, 8, "I like")]
    assert again == glosses
    history = get_history(session_id="t12", session=session)
    user_message = next(m for m in history.messages if m.role == "user")
    assert user_message.glosses == glosses


def test_message_glosses_503_when_the_model_is_down_and_404_for_unknown():
    session = _session()
    with (
        patch("app.routes.chat.model_chat", return_value="Hola"),
        patch("app.translate.llm_translate.model_chat", side_effect=ModelServerUnavailableError("down")),
    ):
        result = take_turn(ChatTurnRequest(session_id="t13", message="hola"), session=session)
        with pytest.raises(HTTPException) as down:
            message_glosses(MessageGlossesRequest(message_id=result.user_message_id), session=session)
    assert down.value.status_code == 503
    with pytest.raises(HTTPException) as missing:
        message_glosses(MessageGlossesRequest(message_id=9999), session=session)
    assert missing.value.status_code == 404