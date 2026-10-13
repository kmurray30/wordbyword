import logging

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db import get_session
from app.languages import NATIVE_LANGUAGE, get_language
from app.chat.threads import append_entries, entry_out
from app.models import ChatMessage
from app.schemas import (
    CoachAboutEvent,
    CoachDraftRequest,
    CoachErrorEvent,
    CoachReadyEvent,
    CoachReplyChunkEvent,
    CoachReplyCompleteEvent,
    DraftSpan,
    DraftToken,
    GlossSpansRequest,
    GlossSpansResponse,
    InterpretInputRequest,
    InterpretInputResponse,
    ReadingOut,
    ReadingsRequest,
    ReadingsResponse,
    TagInputRequest,
    TagInputResponse,
    ThreadEntryIn,
    TranslateCandidate,
    TranslateTextRequest,
    TranslateTextResponse,
    TextReadings,
)
from app.sse_streaming import KEEPALIVE_LINE, iter_with_keepalive
from app.translate import llm_translate
from app.translate.lemmatizer import analyze
from app.translate.readings import readings_for
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
    possible mix of English/the target language and grammar mistakes,
    returning a corrected English restatement alongside its target-language
    translation - used
    to show both under the user's chat bubble rather than a single literal
    (and possibly nonsensical) pass."""
    try:
        native, target = llm_translate.interpret_user_input(req.text, NATIVE_LANGUAGE, req.language)
    except llm_translate.TranslationUnavailableError as exc:
        msg = f"(translation unavailable: {exc})"
        return InterpretInputResponse(native=msg, target=msg)
    return InterpretInputResponse(native=native, target=target)


@router.post("/coach")
def coach_draft(req: CoachDraftRequest, session: Session = Depends(get_session)) -> StreamingResponse:
    """The coach's reply, streamed as SSE. With `thread_message_id` it's
    the side conversation about that message (opened from its 💬) and is
    saved in its thread; without it, it's about the learner's next message
    (the coach box at the bottom; with `check`, the exact message they're
    about to send) and lives in the coach panel until it's sent. Events:
    "about" (that message's id, or null), "ready" (their own target-language text is
    fine to send as written), "reply_chunk"s and "reply_complete"; "error"
    if nothing came back. See app.translate.llm_translate.coach_reply_stream."""
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

    # The message this conversation is about, if it's really one of this
    # session's (a check is always about the next message).
    anchor = None
    if req.thread_message_id is not None and not req.check:
        candidate = session.get(ChatMessage, req.thread_message_id)
        if candidate is not None and candidate.session_id == req.session_id:
            anchor = candidate
    if anchor is not None and anchor.id not in {row.id for row in history_rows}:
        history_rows = [anchor, *history_rows]

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
    about_number = next((n for n, row in number_to_row.items() if anchor is not None and row.id == anchor.id), None)
    thread_turns = (
        [
            ("learner" if e.kind == "question" else "coach", e.text)
            for e in anchor.thread
            if e.kind in ("question", "answer")
        ]
        if anchor is not None
        else []
    )
    draft_turns = [(t.role, t.text) for t in req.draft_thread if t.text.strip()]

    def events():
        about_row = anchor
        reply = ""
        yield _sse("about", CoachAboutEvent(message_id=anchor.id if anchor else None).model_dump_json())
        try:
            for item in iter_with_keepalive(
                llm_translate.coach_reply_stream(
                    req.text,
                    numbered,
                    about_number,
                    thread_turns,
                    draft_turns,
                    NATIVE_LANGUAGE,
                    req.language,
                    check=req.check,
                    message_draft="" if req.check else req.message_draft,
                )
            ):
                if item is None:
                    yield KEEPALIVE_LINE
                    continue
                kind, *rest = item
                if kind == "ready":
                    yield _sse("ready", CoachReadyEvent(ready=rest[0]).model_dump_json())
                elif kind == "reply_chunk":
                    yield _sse("reply_chunk", CoachReplyChunkEvent(delta=rest[0]).model_dump_json())
                elif kind == "reply_complete":
                    reply = rest[0]
        except llm_translate.TranslationUnavailableError as exc:
            yield _sse("error", CoachErrorEvent(message=str(exc)).model_dump_json())
            return
        # Saved only now, after every model call - a short write, so the
        # SQLite write lock is never held while the model streams (see
        # routes/chat.py's _gloss_and_persist_reply).
        entries = []
        if about_row is not None and reply:
            rows = append_entries(
                session,
                about_row.id,
                [
                    ThreadEntryIn(role="learner", kind="question", text=req.text),
                    ThreadEntryIn(role="coach", kind="answer", text=reply),
                ],
            )
            session.commit()
            entries = [entry_out(r) for r in rows]
        yield _sse("reply_complete", CoachReplyCompleteEvent(reply=reply, entries=entries).model_dump_json())

    return StreamingResponse(events(), media_type="text/event-stream", headers=_SSE_HEADERS)


@router.post("/tag-input", response_model=TagInputResponse)
def tag_input(req: TagInputRequest) -> TagInputResponse:
    """Cheap, synchronous, LLM-free per-word classification (see
    app.translate.lemmatizer) - fast enough to call on every keystroke
    (debounced). Needed for reward-event tracking (userTypedTargetWord) and
    the phrase-selection direction vote. The
    slower LLM-backed word/group glossing that drives the hover-to-
    translate UI lives separately in /translate/gloss-spans below, fetched
    lazily on interaction rather than on every edit."""
    spaced = get_language(req.language).spaced
    tokens = [
        DraftToken(surface=tok.surface, lemma=tok.lemma, is_target=tok.is_target, start=tok.start, end=tok.end)
        for tok in analyze(req.text, req.language)
        # A one-letter "word" is noise in a spaced language; a single
        # character is a real word in Japanese/Chinese.
        if any(c.isalpha() for c in tok.surface) and (len(tok.surface) > 1 or not spaced)
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
        _translation, raw_spans = llm_translate.tag_draft(req.text, NATIVE_LANGUAGE, req.language)
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


@router.post("/readings", response_model=ReadingsResponse)
def readings(req: ReadingsRequest) -> ReadingsResponse:
    """How to read each text: romanization for every word, kana over kanji,
    and Mandarin in the learner's characters (see app.translate.readings).
    Deterministic and fast - no LLM. Empty readings for a language written
    in the Latin alphabet."""
    items = []
    for text in req.texts[:100]:
        shown, found = readings_for(text, req.language) if len(text) <= 4000 else (text, [])
        items.append(
            TextReadings(
                text=shown,
                readings=[ReadingOut(start=r.start, end=r.end, roman=r.roman, kana=r.kana, cont=r.cont) for r in found],
            )
        )
    return ReadingsResponse(items=items)
