"""Backs the single live, UI-togglable settings row (app/models.py's
AppSettings) - which chat backend is active and whether word-bank
vocabulary steering is on. Separate from app/wordbank/store.py (which
reads/writes per-lemma WordBankEntry rows) since this is global app
configuration, not per-word data.
"""

from sqlalchemy.orm import Session

from app.config import MODEL_PROVIDER, WORD_WEIGHTING_ENABLED
from app.models import AppSettings

_SETTINGS_ROW_ID = 1


def get_settings(session: Session) -> AppSettings:
    row = session.get(AppSettings, _SETTINGS_ROW_ID)
    if row is None:
        # First-ever read: seed from the env-var BOOT defaults in
        # config.py, not the column defaults on the model (which are the
        # same values, captured at import time) - this way an operator's
        # env config still takes effect until someone actually flips the
        # UI toggle for the first time, and a restart with a changed env
        # var only matters before that first flip too.
        row = AppSettings(
            id=_SETTINGS_ROW_ID,
            model_provider=MODEL_PROVIDER,
            word_weighting_enabled=WORD_WEIGHTING_ENABLED,
        )
        session.add(row)
        session.commit()
        session.refresh(row)
    return row


def update_settings(
    session: Session,
    *,
    model_provider: str | None = None,
    word_weighting_enabled: bool | None = None,
) -> AppSettings:
    row = get_settings(session)
    if model_provider is not None:
        row.model_provider = model_provider
    if word_weighting_enabled is not None:
        row.word_weighting_enabled = word_weighting_enabled
    session.commit()
    session.refresh(row)
    return row


def weighting_active(row: AppSettings) -> bool:
    """Per-word weighting (logit_bias) is keyed to the local model's own
    tokenizer (see app/chat/logit_bias.py) - meaningless against a hosted
    model's black-box tokenizer, so it's never actually active against the
    "openai" provider regardless of the stored toggle value."""
    return row.word_weighting_enabled and row.model_provider == "local"
