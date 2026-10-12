import logging

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import NATIVE_LANGUAGE, TARGET_LANGUAGE
from app.db import get_session
from app.chat.threads import append_entries, entry_out
from app.models import ChatMessage
from app.schemas import (
    CoachAnswerChunkEvent,
    CoachAnswerCompleteEvent,
    CoachDraftRequest,
    CoachErrorEvent,
    CoachMeaningChunkEvent,
    CoachMeaningCompleteEvent,
    CoachOptionChunkEvent,
    CoachOptionCompleteEvent,
    CoachSuggestionChunkEvent,
    CoachSuggestionCompleteEvent,
    CoachThreadAnchorEvent,
    CoachVerdictEvent,
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
    ThreadEntryIn,
)
from app.sse_streaming import KEEPALIVE_LINE, iter_with_keepalive
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


# See app/routes/chat.py's own _SSE_HEADERS for why: a reverse proxy
# between the browser and this server (Railway's edge, in production) can
# buffer a whole SSE response before forwarding it rather than relaying
# each chunk as it's yielded - these are the standard headers that hint a
# proxy to not do that.
_SSE_HEADERS = {"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"}


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

    Streamed as Server-Sent Events, one event per field of the SAME
    underlying LLM call (app.translate.llm_translate.coach_draft_stream -
    see its own docstring for the full field-by-field sequence). First, a
    "verdict" event ("clean"/"minor"/"fix") - usable INSTANTLY, before
    anything else has streamed, to decide whether the draft is already
    good to go (see ChatInput.tsx's coachReady). Then "meaning_chunk"/
    "meaning_complete" carry the restatement live, delta by delta - the
    same token-by-token feel /chat/turn/stream's own reply already has.
    What follows depends on the verdict: nothing further for "clean";
    "suggestion_chunk"/"suggestion_complete" (one lightweight corrected-
    text suggestion) for "minor"; or, for "fix", three formality-ranked
    options streamed via "option_chunk"/"option_complete" events (each
    carrying `index` 0/1/2 and `field` "spanish"/"english"), each option's
    own English back-translation streaming in right after its own Spanish
    text - not batched together at the end. An "error" event means
    nothing at all was usable (not even the verdict); a stream that ends
    partway through the "fix" options just means the rest weren't ready -
    whatever DID arrive stays fully usable (see coach_draft_stream's own
    docstring on this graceful-degradation philosophy)."""
    history_rows = list(
        reversed(
            session.scalars(
                select(ChatMessage)
                .where(ChatMessage.session_id == req.session_id)
                .order_by(ChatMessage.id.desc())
                .limit(_COACH_HISTORY_TURNS)
            ).all()
        )
    )
    history = [(row.role, row.text) for row in history_rows]

    # The message a question would be about by default: the thread open on
    # screen (if it's really one of this session's messages), else the
    # partner's latest message.
    by_id = {row.id: row for row in history_rows}
    anchor = by_id.get(req.thread_message_id) if req.thread_message_id is not None else None
    if anchor is None and req.thread_message_id is not None:
        candidate = session.get(ChatMessage, req.thread_message_id)
        if candidate is not None and candidate.session_id == req.session_id:
            anchor = candidate
    open_thread_anchor = anchor
    if anchor is None:
        anchor = next((row for row in reversed(history_rows) if row.role == "assistant"), None)

    def thread_turns(message: ChatMessage | None) -> list[tuple[str, str]]:
        if message is None:
            return []
        return [
            ("learner" if e.kind == "question" else "tutor", e.text)
            for e in message.thread
            if e.kind in ("question", "answer")
        ]

    open_thread = (
        (open_thread_anchor.text, thread_turns(open_thread_anchor)) if open_thread_anchor is not None else None
    )
    numbered = [
        {
            "number": i + 1,
            "role": row.role,
            "text": row.text,
            "glosses": "; ".join(f"{t.surface} = {t.gloss}" for t in row.tokens if t.gloss)[:600],
        }
        for i, row in enumerate(history_rows)
    ]
    number_to_row = {i + 1: row for i, row in enumerate(history_rows)}
    default_number = next((n for n, row in number_to_row.items() if anchor is not None and row.id == anchor.id), None)
    default_turns = thread_turns(anchor)
    draft_turns = [(t.role, t.text) for t in req.draft_thread if t.text.strip()]

    def tutor_events():
        about_row = anchor
        answer = ""
        for item in iter_with_keepalive(
            llm_translate.tutor_answer_stream(
                req.text,
                numbered,
                default_number,
                default_turns,
                NATIVE_LANGUAGE,
                TARGET_LANGUAGE,
                draft_turns=draft_turns,
            )
        ):
            if item is None:
                yield KEEPALIVE_LINE
                continue
            kind, *rest = item
            if kind == "about":
                # None = about the learner's NEXT message (or nothing in
                # particular): answered in the next-message log on the
                # client, which saves it with that message once it's sent.
                about_row = number_to_row.get(rest[0], anchor) if rest[0] is not None else None
                yield _sse(
                    "thread_anchor",
                    CoachThreadAnchorEvent(
                        message_id=about_row.id if about_row is not None else None, question=req.text
                    ).model_dump_json(),
                )
            elif kind == "answer_chunk":
                yield _sse("answer_chunk", CoachAnswerChunkEvent(delta=rest[0]).model_dump_json())
            elif kind == "answer_complete":
                answer = rest[0]
        # Persisted only now, after every model call - a short write, so the
        # SQLite write lock is never held while the model streams (see
        # routes/chat.py's _gloss_and_persist_reply).
        entries = []
        if about_row is not None and answer:
            rows = append_entries(
                session,
                about_row.id,
                [
                    ThreadEntryIn(role="learner", kind="question", text=req.text),
                    ThreadEntryIn(role="tutor", kind="answer", text=answer),
                ],
            )
            session.commit()
            entries = [entry_out(r) for r in rows]
        yield _sse("answer_complete", CoachAnswerCompleteEvent(answer=answer, entries=entries).model_dump_json())

    def events():
        try:
            if req.force == "ask":
                yield _sse("verdict", CoachVerdictEvent(verdict="ask").model_dump_json())
                yield from tutor_events()
                return
            for item in iter_with_keepalive(
                llm_translate.coach_draft_stream(
                    req.text,
                    history,
                    NATIVE_LANGUAGE,
                    TARGET_LANGUAGE,
                    open_thread=open_thread,
                    allow_ask=req.force != "attempt",
                    draft_turns=draft_turns,
                )
            ):
                if item is None:
                    yield KEEPALIVE_LINE
                    continue
                kind, *rest = item
                if kind == "verdict" and rest[0] == "ask":
                    yield _sse("verdict", CoachVerdictEvent(verdict="ask").model_dump_json())
                    yield from tutor_events()
                    return
                if kind == "verdict":
                    yield _sse("verdict", CoachVerdictEvent(verdict=rest[0]).model_dump_json())
                elif kind == "meaning_chunk":
                    yield _sse("meaning_chunk", CoachMeaningChunkEvent(delta=rest[0]).model_dump_json())
                elif kind == "meaning_complete":
                    yield _sse("meaning_complete", CoachMeaningCompleteEvent(meaning=rest[0]).model_dump_json())
                elif kind == "suggestion_chunk":
                    yield _sse("suggestion_chunk", CoachSuggestionChunkEvent(delta=rest[0]).model_dump_json())
                elif kind == "suggestion_complete":
                    yield _sse(
                        "suggestion_complete", CoachSuggestionCompleteEvent(suggestion=rest[0]).model_dump_json()
                    )
                elif kind == "option_spanish_chunk":
                    yield _sse(
                        "option_chunk", CoachOptionChunkEvent(index=rest[0], field="spanish", delta=rest[1]).model_dump_json()
                    )
                elif kind == "option_spanish_complete":
                    yield _sse(
                        "option_complete",
                        CoachOptionCompleteEvent(index=rest[0], field="spanish", text=rest[1]).model_dump_json(),
                    )
                elif kind == "option_english_chunk":
                    yield _sse(
                        "option_chunk", CoachOptionChunkEvent(index=rest[0], field="english", delta=rest[1]).model_dump_json()
                    )
                elif kind == "option_english_complete":
                    yield _sse(
                        "option_complete",
                        CoachOptionCompleteEvent(index=rest[0], field="english", text=rest[1]).model_dump_json(),
                    )
        except llm_translate.TranslationUnavailableError as exc:
            yield _sse("error", CoachErrorEvent(message=str(exc)).model_dump_json())

    return StreamingResponse(events(), media_type="text/event-stream", headers=_SSE_HEADERS)


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
