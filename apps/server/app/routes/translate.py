import logging

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import NATIVE_LANGUAGE, TARGET_LANGUAGE
from app.db import get_session
from app.models import ChatMessage
from app.schemas import (
    CoachCoreEvent,
    CoachDraftRequest,
    CoachErrorEvent,
    CoachOption,
    CoachTranslationsEvent,
    DraftSpan,
    DraftToken,
    GlossSpansRequest,
    GlossSpansResponse,
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

logger = logging.getLogger(__name__)

_COACH_HISTORY_TURNS = 8

router = APIRouter(prefix="/translate", tags=["translate"])


def _build_draft_spans(text: str, raw_spans: list[dict[str, str]]) -> list[DraftSpan]:
    """Shared between /gloss-spans and the coach's streaming "translations"
    phase below - matches raw surface/gloss/note/translation/
    alternate_gloss dicts back to exact offsets in `text` (app.translate.
    span_matching) and builds the DraftSpan shape the frontend's word-
    candidate popover already knows how to render, whichever endpoint the
    data came from."""
    matched = match_spans(text, raw_spans)
    spans: list[DraftSpan] = []
    for m in matched:
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
                alternate_gloss=m.alternate_gloss,
                literal=m.literal,
            )
        )
    return spans


def _sse(event: str, data: str) -> str:
    return f"event: {event}\ndata: {data}\n\n"


@router.post("/text", response_model=TranslateTextResponse)
def translate_text(req: TranslateTextRequest) -> TranslateTextResponse:
    try:
        translation = llm_translate.translate_text(req.text, req.source_lang, req.target_lang)
    except llm_translate.TranslationUnavailableError as exc:
        return TranslateTextResponse(translation=f"(translation unavailable: {exc})")
    return TranslateTextResponse(translation=translation)


@router.post("/interpret", response_model=InterpretInputResponse)
def interpret_input(req: InterpretInputRequest) -> InterpretInputResponse:
    """For the learner's own message: infers what they meant across a
    possible mix of English/Spanish and grammar mistakes, returning a
    corrected English restatement alongside its Spanish translation - used
    to show both under the user's chat bubble rather than a single literal
    (and possibly nonsensical) pass."""
    try:
        native, target = llm_translate.interpret_user_input(req.text, NATIVE_LANGUAGE, TARGET_LANGUAGE)
    except llm_translate.TranslationUnavailableError as exc:
        msg = f"(translation unavailable: {exc})"
        return InterpretInputResponse(native=msg, target=msg)
    return InterpretInputResponse(native=native, target=target)


@router.post("/coach")
def coach_draft(req: CoachDraftRequest, session: Session = Depends(get_session)) -> StreamingResponse:
    """For a message the learner is still drafting (not yet sent) - unlike
    /interpret above, which corrects a single message in isolation, this
    pulls the session's recent conversation for context so the suggested
    phrasing actually fits the tone of what's been said so far, and returns
    feedback on the attempt plus phrasing options at a few formality
    levels rather than a single flat correction.

    Streamed as Server-Sent Events, one event per phase of the SAME
    underlying LLM call (app.translate.llm_translate.coach_draft_stream):
    a "core" event the moment the feedback + options themselves are ready
    (usable immediately - the frontend shows these without waiting on
    anything else), then a "translations" event once each option's own
    English translation + word-by-word breakdown finishes streaming
    afterward (for pre-populating the chat input's gloss cache the instant
    an option is picked - see CoachOption's fields). An "error" event
    means the core phase itself never produced anything usable; a stream
    that ends after "core" with no "translations" just means that part
    wasn't ready - that's not itself an error (see coach_draft_stream's
    docstring)."""
    history_rows = session.scalars(
        select(ChatMessage)
        .where(ChatMessage.session_id == req.session_id)
        .order_by(ChatMessage.id.desc())
        .limit(_COACH_HISTORY_TURNS)
    ).all()
    history = [(row.role, row.text) for row in reversed(history_rows)]

    def events():
        core_options: list[tuple[str, str]] = []
        try:
            for kind, payload in llm_translate.coach_draft_stream(req.text, history, NATIVE_LANGUAGE, TARGET_LANGUAGE):
                if kind == "core":
                    core_options = payload["options"]
                    core_event = CoachCoreEvent(
                        meaning=payload["meaning"],
                        feedback=payload["feedback"],
                        options=[CoachOption(formality=f, spanish=s) for f, s in core_options],
                    )
                    yield _sse("core", core_event.model_dump_json())
                elif kind == "translations":
                    options = []
                    for i, (formality, spanish) in enumerate(core_options):
                        per_option = payload[i] if i < len(payload) else {"english": "", "spans": []}
                        spans = _build_draft_spans(spanish, per_option["spans"])
                        options.append(
                            CoachOption(formality=formality, spanish=spanish, english=per_option["english"], spans=spans)
                        )
                    translations_event = CoachTranslationsEvent(options=options)
                    yield _sse("translations", translations_event.model_dump_json())
        except llm_translate.TranslationUnavailableError as exc:
            yield _sse("error", CoachErrorEvent(message=str(exc)).model_dump_json())

    return StreamingResponse(events(), media_type="text/event-stream")


@router.post("/tag-input", response_model=TagInputResponse)
def tag_input(req: TagInputRequest) -> TagInputResponse:
    """Cheap, synchronous, LLM-free per-word classification (spaCy +
    lemmatizer.py's is_spanish heuristic) - fast enough to call on every
    keystroke (debounced). Needed for reward-event tracking
    (userTypedSpanishWord) and the phrase-selection direction vote. The
    slower LLM-backed word/group glossing that drives the hover-to-
    translate UI lives separately in /translate/gloss-spans below, fetched
    lazily on interaction rather than on every edit."""
    tokens = [
        DraftToken(surface=tok.surface, lemma=tok.lemma, is_spanish=tok.is_spanish, start=tok.start, end=tok.end)
        for tok in analyze(req.text)
        if tok.surface.isalpha() and len(tok.surface) > 1
    ]
    return TagInputResponse(tokens=tokens)


@router.post("/gloss-spans", response_model=GlossSpansResponse)
def gloss_spans(req: GlossSpansRequest) -> GlossSpansResponse:
    """The slower, LLM-backed half of what used to be /translate/tag-input:
    app.translate.llm_translate.tag_draft's word/group glossing, matched
    back to exact offsets via app.translate.span_matching. Fetched lazily
    (on hover/click/tap of a word), not on every keystroke, since this is a
    real LLM call. On failure, returns an empty span list - no dictionary/
    MT fallback."""
    try:
        _translation, raw_spans = llm_translate.tag_draft(req.text, NATIVE_LANGUAGE, TARGET_LANGUAGE)
    except llm_translate.TranslationUnavailableError as exc:
        logger.warning("tag_draft unavailable for %r: %s", req.text, exc)
        raw_spans = []

    spans = _build_draft_spans(req.text, raw_spans)
    if raw_spans and not spans:
        # The call succeeded, but every span's surface text failed to
        # locate in the draft (a hallucinated/paraphrased span, or a
        # normalization mismatch span_matching doesn't handle) - visible
        # here since the symptom (empty `spans` in the response) is
        # otherwise indistinguishable from the LLM call failing outright.
        logger.warning("tag_draft returned %d span(s) but none matched %r: %r", len(raw_spans), req.text, raw_spans)

    return GlossSpansResponse(spans=spans)
