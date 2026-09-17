from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db import get_session
from app.schemas import RewardEventRequest, RewardEventResponse
from app.wordbank import store

router = APIRouter(prefix="/events", tags=["events"])


@router.post("/reward", response_model=RewardEventResponse)
def reward_event(req: RewardEventRequest, session: Session = Depends(get_session)) -> RewardEventResponse:
    try:
        entry = store.apply_reward_event(session, req.event_type, req.lemma)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if entry is None:
        raise HTTPException(status_code=404, detail=f"unknown lemma: {req.lemma}")

    session.commit()
    return RewardEventResponse(
        lemma=entry.lemma,
        familiarity=entry.familiarity,
        review_interval_days=entry.review_interval_days,
    )
