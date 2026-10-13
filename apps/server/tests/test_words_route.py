import pytest
from fastapi import HTTPException
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db import Base
from app.routes.words import list_words, remove_word, save_word
from app.schemas import SavedWordIn


def _session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


def test_save_list_and_remove_a_word():
    session = _session()
    saved = save_word(
        SavedWordIn(session_id="s", word="Comí", gloss="ate", grammar="comer (to eat): I, preterite", sentence="Ayer comí paella.", message_id=3),
        session=session,
    )
    assert (saved.word, saved.gloss, saved.message_id) == ("Comí", "ate", 3)

    # Saving it again (any case) keeps the one entry.
    again = save_word(SavedWordIn(session_id="s", word="comí"), session=session)
    assert again.id == saved.id
    assert [w.word for w in list_words(session_id="s", session=session).words] == ["Comí"]
    assert list_words(session_id="other", session=session).words == []

    assert remove_word(session_id="s", word="COMÍ", session=session).removed == 1
    assert list_words(session_id="s", session=session).words == []
    assert remove_word(session_id="s", word="comí", session=session).removed == 0


def test_saving_an_empty_word_is_rejected():
    with pytest.raises(HTTPException) as exc:
        save_word(SavedWordIn(session_id="s", word="  "), session=_session())
    assert exc.value.status_code == 422
