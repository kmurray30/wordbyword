from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app import settings_store
from app.db import get_session
from app.models import AppSettings
from app.schemas import SettingsResponse, UpdateSettingsRequest

router = APIRouter(prefix="/settings", tags=["settings"])

_VALID_PROVIDERS = {"local", "openai"}


def _to_response(row: AppSettings) -> SettingsResponse:
    return SettingsResponse(
        model_provider=row.model_provider,
        word_weighting_enabled=row.word_weighting_enabled,
        word_weighting_active=settings_store.weighting_active(row),
    )


@router.get("", response_model=SettingsResponse)
def read_settings(session: Session = Depends(get_session)) -> SettingsResponse:
    return _to_response(settings_store.get_settings(session))


@router.put("", response_model=SettingsResponse)
def write_settings(req: UpdateSettingsRequest, session: Session = Depends(get_session)) -> SettingsResponse:
    if req.model_provider is not None and req.model_provider not in _VALID_PROVIDERS:
        raise HTTPException(status_code=400, detail=f"model_provider must be one of {sorted(_VALID_PROVIDERS)}")
    row = settings_store.update_settings(
        session, model_provider=req.model_provider, word_weighting_enabled=req.word_weighting_enabled
    )
    return _to_response(row)
