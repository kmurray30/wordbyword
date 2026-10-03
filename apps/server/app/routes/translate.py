from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.orm import Session

from app import settings_store
from app.config import NATIVE_LANGUAGE, TARGET_LANGUAGE
from app.db import get_session
from app.models import ChatMessage
from app.schemas import (
    CoachDraftRequest,
    CoachDraftResponse,
    CoachOption,
    DraftSpan,
    DraftToken,
    InterpretInputRequest,
    InterpretInputResponse,
    TagInputRequest,
    TagInputResponse,
    TranslateCandidate,
    TranslateTextRequest,
    TranslateTextResponse,
)
from app.translate import llm_translate
from app.translate.lemmatizer import analyze
from app.translate.span_matching import match_spans

_COACH_HISTORY_TURNS = 8

router = APIRouter(prefix="/translate", tags=["translate"])


@router.post("/text", response_model=TranslateTextResponse)
def translate_text(req: TranslateTextRequest, session: Session = Depends(get_session)) -> TranslateTextResponse:
    provider = settings_store.get_settings(session).model_provider
    try:
        translation = llm_translate.translate_text(req.text, req.source_lang, req.target_lang, provider=provider)
    except llm_translate.TranslationUnavailableError as exc:
        return TranslateTextResponse(translation=f"(translation unavailable: {exc})")
    return TranslateTextResponse(translation=translation)


@router.post("/interpret", response_model=InterpretInputResponse)
def interpret_input(req: InterpretInputRequest, session: Session = Depends(get_session)) -> InterpretInputResponse:
    """For the learner's own message: infers what they meant across a
    possible mix of English/Spanish and grammar mistakes, returning a
    corrected English restatement alongside its Spanish translation - used
    to show both under the user's chat bubble rather than a single literal
    (and possibly nonsensical) pass."""
    provider = settings_store.get_settings(session).model_provider
    try:
        native, target = llm_translate.interpret_user_input(
            req.text, NATIVE_LANGUAGE, TARGET_LANGUAGE, provider=provider
        )
    except llm_translate.TranslationUnavailableError as exc:
        msg = f"(translation unavailable: {exc})"
        return InterpretInputResponse(native=msg, target=msg)
    return InterpretInputResponse(native=native, target=target)


@router.post("/coach", response_model=CoachDraftResponse)
def coach_draft(req: CoachDraftRequest, session: Session = Depends(get_session)) -> CoachDraftResponse:
    """For a message the learner is still drafting (not yet sent) - unlike
    /interpret above, which corrects a single message in isolation, this
    pulls the session's recent conversation for context so the suggested
    phrasing actually fits the tone of what's been said so far, and returns
    feedback on the attempt plus phrasing options at a few formality
    levels rather than a single flat correction."""
    provider = settings_store.get_settings(session).model_provider
    history_rows = session.scalars(
        select(ChatMessage)
        .where(ChatMessage.session_id == req.session_id)
        .order_by(ChatMessage.id.desc())
        .limit(_COACH_HISTORY_TURNS)
    ).all()
    history = [(row.role, row.text) for row in reversed(history_rows)]

    try:
        meaning, feedback, options = llm_translate.coach_draft(
            req.text, history, NATIVE_LANGUAGE, TARGET_LANGUAGE, provider=provider
        )
    except llm_translate.TranslationUnavailableError as exc:
        msg = f"(translation unavailable: {exc})"
        return CoachDraftResponse(meaning=msg, feedback="", options=[CoachOption(formality="neutral", spanish=msg)])
    return CoachDraftResponse(
        meaning=meaning,
        feedback=feedback,
        options=[CoachOption(formality=f, spanish=s) for f, s in options],
    )


@router.post("/tag-input", response_model=TagInputResponse)
def tag_input(req: TagInputRequest, session: Session = Depends(get_session)) -> TagInputResponse:
    """Two independent passes over the same draft text: `tokens` is a
    cheap, synchronous, LLM-free per-word classification (spaCy +
    lemmatizer.py's is_spanish heuristic) the frontend needs on every call
    for reward-event tracking and its phrase-selection direction vote;
    `spans` is the slower LLM-backed word/group glossing (app.translate.
    llm_translate.tag_draft) that drives the hover-to-translate UI,
    matched back to exact offsets via app.translate.span_matching. Kept in
    one response so the frontend only has one request to debounce, even
    though the two halves serve different purposes. On an LLM failure,
    `spans` comes back empty - no dictionary/MT fallback - but `tokens` is
    unaffected."""
    provider = settings_store.get_settings(session).model_provider

    tokens = [
        DraftToken(surface=tok.surface, lemma=tok.lemma, is_spanish=tok.is_spanish, start=tok.start, end=tok.end)
        for tok in analyze(req.text)
        if tok.surface.isalpha() and len(tok.surface) > 1
    ]

    try:
        _translation, raw_spans = llm_translate.tag_draft(req.text, NATIVE_LANGUAGE, TARGET_LANGUAGE, provider=provider)
    except llm_translate.TranslationUnavailableError:
        raw_spans = []

    spans: list[DraftSpan] = []
    for m in match_spans(req.text, raw_spans):
        candidate_text = m.translation.strip() or m.gloss.strip()
        if not candidate_text:
            continue  # nothing usable to show for this span - drop it
        spans.append(
            DraftSpan(
                surface=m.surface,
                start=m.start,
                end=m.end,
                clickable=bool(m.translation.strip()),
                candidates=[TranslateCandidate(translation=candidate_text, description=m.note)],
            )
        )

    return TagInputResponse(tokens=tokens, spans=spans)
