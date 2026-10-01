from unittest.mock import patch

import pytest
from fastapi import HTTPException
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app import settings_store
from app.db import Base
from app.routes.settings import read_settings, write_settings
from app.schemas import UpdateSettingsRequest


def _session():
    engine = create_engine("sqlite:///:memory:")
    from app import models  # noqa: F401  (register tables on Base)

    Base.metadata.create_all(bind=engine)
    return sessionmaker(bind=engine)()


def test_get_settings_seeds_a_default_row_from_config_on_first_read():
    with (
        patch("app.settings_store.MODEL_PROVIDER", "local"),
        patch("app.settings_store.WORD_WEIGHTING_ENABLED", False),
    ):
        row = settings_store.get_settings(_session())

    assert row.model_provider == "local"
    assert row.word_weighting_enabled is False


def test_get_settings_returns_the_same_row_on_a_second_read():
    session = _session()
    first = settings_store.get_settings(session)
    first.model_provider = "openai"
    session.commit()

    second = settings_store.get_settings(session)
    assert second.model_provider == "openai"


def test_update_settings_only_touches_fields_that_are_passed():
    session = _session()
    settings_store.update_settings(session, model_provider="openai", word_weighting_enabled=True)

    row = settings_store.update_settings(session, word_weighting_enabled=False)
    assert row.model_provider == "openai"  # untouched by the second call
    assert row.word_weighting_enabled is False


def test_weighting_active_requires_both_enabled_and_local_provider():
    session = _session()
    row = settings_store.update_settings(session, model_provider="local", word_weighting_enabled=True)
    assert settings_store.weighting_active(row) is True

    row = settings_store.update_settings(session, model_provider="openai")
    # logit_bias needs the local tokenizer - never active against a hosted
    # provider regardless of the stored toggle.
    assert settings_store.weighting_active(row) is False

    row = settings_store.update_settings(session, model_provider="local", word_weighting_enabled=False)
    assert settings_store.weighting_active(row) is False


def test_read_settings_route_reports_word_weighting_active():
    session = _session()
    settings_store.update_settings(session, model_provider="local", word_weighting_enabled=True)

    result = read_settings(session=session)
    assert result.model_provider == "local"
    assert result.word_weighting_enabled is True
    assert result.word_weighting_active is True


def test_write_settings_route_rejects_an_unknown_provider():
    with pytest.raises(HTTPException) as exc_info:
        write_settings(UpdateSettingsRequest(model_provider="anthropic"), session=_session())
    assert exc_info.value.status_code == 400


def test_write_settings_route_applies_a_valid_update():
    session = _session()
    result = write_settings(UpdateSettingsRequest(model_provider="openai"), session=session)
    assert result.model_provider == "openai"
    # Never active under a hosted provider, even though nothing here
    # explicitly disabled the stored toggle.
    assert result.word_weighting_active is False
