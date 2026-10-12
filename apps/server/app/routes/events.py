from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db import get_session
from app.schemas import RewardEventRequest, RewardEventResponse
from app.vocab.suggest import record_pick_up
from app.wordbank import store

router = APIRouter(prefix="/events", tags=["events"])


@router.post("/reward", response_model=RewardEventResponse)
def reward_event(req: RewardEventRequest, session: Session = Depends(get_session)) -> RewardEventResponse:
    try:
        entry = store.apply_reward_event(session, req.event_type, req.lemma)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if req.event_type == "userTypedSpanishWord":
        # A no-op unless contextual vocab recently offered this word.
        record_pick_up(session, entry.lemma)

    session.commit()
    return RewardEventResponse(
        lemma=entry.lemma,
        familiarity=entry.familiarity,
        review_interval_days=entry.review_interval_days,
    )
