from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from app.chat.openai_client import ModelServerUnavailableError, chat as model_chat, chat_stream as model_chat_stream
from app.chat.prompt_builder import build_messages, build_vocab_block
from app.chat.glosses import build_glosses, load_glosses, reply_tokens, save_glosses, token_glosses
from app.chat.threads import append_entries, entry_out
from app.config import CONTEXTUAL_VOCAB_ENABLED
from app.db import get_session
from app.languages import NATIVE_LANGUAGE, get_language
from app.models import ChatMessage, MessageToken
from app.schemas import (
    ChatHistoryMessage,
    ChatHistoryResponse,
    ChatTurnChunkEvent,
    ChatTurnErrorEvent,
    ChatTurnRequest,
    ChatTurnResponse,
    ClearHistoryResponse,
    MessageGlosses,
    MessageGlossesRequest,
    SaveUserInterpretationRequest,
    SaveUserInterpretationResponse,
    ThreadEntryIn,
    TokenAnnotation,
    VocabDebug,
)
from app.sse_streaming import KEEPALIVE_LINE, iter_with_keepalive
from app.translate import llm_translate
from app.vocab import suggest as vocab_suggest
from app.vocab.summary import clear_summary
from app.wordbank import store

router = APIRouter(prefix="/chat", tags=["chat"])


def _sse(event: str, data: str) -> str:
    return f"event: {event}\ndata: {data}\n\n"


# Reported live: the reply arrived all at once instead of streaming in -
# consistent with a reverse proxy between the browser and this server
# (Railway's edge, in production) buffering the whole SSE response before
# forwarding it, rather than this server failing to flush each chunk as
# it's yielded (uvicorn does that correctly on its own - confirmed via a
# direct local stream). Cache-Control/Connection are standard for SSE;
# X-Accel-Buffering is nginx's own directive to disable response
# buffering, but several other reverse proxies honor it by the same
# convention since it's become a de facto standard for exactly this
# problem - cheap to send even if Railway's edge doesn't recognize it.
_SSE_HEADERS = {"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"}

HISTORY_TURNS = 10


def _recent_history(session: Session, session_id: str) -> list[tuple[str, str]]:
    rows = session.scalars(
        select(ChatMessage)
        .where(ChatMessage.session_id == session_id)
        .order_by(ChatMessage.id.desc())
        .limit(HISTORY_TURNS)
    ).all()
    rows = list(reversed(rows))
    return [(row.role, row.text) for row in rows]


def _prepare_turn(
    session: Session, req: ChatTurnRequest
) -> tuple[list[dict[str, str]], "vocab_suggest.TurnSuggestions | None"]:
    """The model's input messages for this turn, plus - when contextual
    vocab is on - what its system prompt offered (None when off, which also
    tells _gloss_and_persist_reply to skip the vocab bookkeeping)."""
    history = _recent_history(session, req.session_id)
    if not CONTEXTUAL_VOCAB_ENABLED:
        return build_messages(history, req.message, req.language), None
    suggestions = vocab_suggest.suggest_for_turn(session, req.session_id, req.language, history, req.message)
    vocab_block = build_vocab_block(
        [(w.lemma, w.translation) for w in suggestions.relevant],
        [(w.lemma, w.translation) for w in suggestions.wildcard],
    )
    return build_messages(history, req.message, req.language, vocab_block), suggestions


def _load_vocab_debug(message: ChatMessage) -> VocabDebug | None:
    """The saved debug payload of an assistant reply, if it has one."""
    if not message.vocab_debug:
        return None
    try:
        return VocabDebug.model_validate_json(message.vocab_debug)
    except ValueError:
        return None


@router.get("/history", response_model=ChatHistoryResponse)
def get_history(session_id: str, session: Session = Depends(get_session)) -> ChatHistoryResponse:
    rows = session.scalars(
        select(ChatMessage).where(ChatMessage.session_id == session_id).order_by(ChatMessage.id)
    ).all()
    messages = [
        ChatHistoryMessage(
            id=row.id,
            role=row.role,
            text=row.text,
            tokens=[
                TokenAnnotation(
                    surface=t.surface,
                    lemma=t.lemma,
                    pos=t.pos,
                    gloss=t.gloss,
                    is_new=t.is_new,
                    note=t.note,
                    literal=t.literal,
                )
                for t in row.tokens
            ],
            native=row.native_text,
            target=row.target_text,
            thread=[entry_out(e) for e in row.thread],
            glosses=load_glosses(row),
            vocab_debug=_load_vocab_debug(row),
        )
        for row in rows
    ]
    return ChatHistoryResponse(messages=messages)


@router.post("/message/interpretation", response_model=SaveUserInterpretationResponse)
def save_user_interpretation(
    req: SaveUserInterpretationRequest, session: Session = Depends(get_session)
) -> SaveUserInterpretationResponse:
    """The frontend fetches a fresh user message's native/target
    interpretation live (POST /translate/interpret, fired the moment the
    message is sent - see ChatMessage.tsx's fetchUserInterpretation) rather
    than this server computing it itself as part of /chat/turn (which would
    add a second, serial LLM call to every turn's latency). This just
    persists that already-fetched result against the message it belongs to,
    so GET /chat/history can hydrate it directly on a later reload instead
    of leaving the learner looking at "Translating…" forever for an old
    message if that on-demand fetch is ever slow or fails. Also used for an
    assistant reply whose translation wasn't saved with it (`native` only)."""
    message = session.get(ChatMessage, req.message_id)
    if message is None:
        raise HTTPException(status_code=404, detail="message not found")
    message.native_text = req.native
    # An assistant reply only has a translation (native); keep target empty.
    message.target_text = req.target if message.role == "user" else ""
    session.commit()
    return SaveUserInterpretationResponse(ok=True)


@router.post("/message/glosses", response_model=MessageGlosses)
def message_glosses(req: MessageGlossesRequest, session: Session = Depends(get_session)) -> MessageGlosses:
    """The word/phrase glosses for a message, made and saved the first time
    they're asked for (the learner's own messages, and replies from before
    glosses were saved) - after that, GET /chat/history carries them.
    503 if they can't be made right now."""
    message = session.get(ChatMessage, req.message_id)
    if message is None:
        raise HTTPException(status_code=404, detail="message not found")
    saved = load_glosses(message)
    if saved is not None:
        return saved
    text = message.text
    learner = message.role == "user"
    language = message.language or "es"
    session.rollback()  # no transaction open while the model runs (see _gloss_and_persist_reply)
    try:
        _translation, words, phrases = llm_translate.gloss_message(text, NATIVE_LANGUAGE, language, learner=learner)
    except llm_translate.TranslationUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    glosses = build_glosses(text, words, phrases, language)
    message = session.get(ChatMessage, req.message_id)
    if message is not None and glosses.words:
        save_glosses(message, glosses)
        session.commit()
    return glosses


@router.post("/history/clear", response_model=ClearHistoryResponse)
def clear_history(session_id: str, session: Session = Depends(get_session)) -> ClearHistoryResponse:
    rows = session.scalars(select(ChatMessage).where(ChatMessage.session_id == session_id)).all()
    for row in rows:
        session.delete(row)
    clear_summary(session, session_id)
    session.commit()
    return ClearHistoryResponse(cleared=len(rows))


def _gloss_and_persist_reply(
    session: Session,
    session_id: str,
    reply_text: str,
    user_message: ChatMessage,
    user_thread: list[ThreadEntryIn] | None = None,
    language: str = "es",
    suggestions: "vocab_suggest.TurnSuggestions | None" = None,
) -> ChatTurnResponse:
    """Shared by both /chat/turn and /chat/turn/stream - everything that
    happens once the model's full reply text is known: one LLM call glosses
    every word/group in the reply using its meaning IN CONTEXT, and returns
    the whole-sentence translation from that exact same pass - one
    consistent source for both. No dictionary/MT fallback for a word it
    missed or a failed call - that word just gets no gloss until a later
    turn supplies one (see store.record_exposure below).

    `user_message` is the learner's own just-sent turn, NOT yet added to
    the session: it's added (and flushed, which is what assigns its id)
    only here, after the gloss call below - the FIRST write of the whole
    turn. SQLite has one writer at a time, and a flush takes that write
    lock until commit: flushing the user message up front, before the
    model call (as both routes used to), held the lock for the entire
    reply generation (10s+), so any other write landing in that window -
    the reward events the frontend fires at the moment of sending, in
    particular - timed out with "database is locked" (seen live). Now the
    lock is held only for the few milliseconds between this flush and the
    commit at the end.

    `suggestions` (contextual vocab on) are rewarded against the reply's
    words in that same transaction, kept as the reply's debug payload, and
    the slower follow-up work (running summary, embedding new words) is
    handed to a background thread once it commits."""
    base = get_language(language).code
    try:
        translation, words, phrases = llm_translate.gloss_message(reply_text, NATIVE_LANGUAGE, language)
    except llm_translate.TranslationUnavailableError:
        translation, words, phrases = "", [], []
    glosses = build_glosses(reply_text, words, phrases, base)

    session.add(user_message)
    session.flush()  # assigns user_message.id without committing yet - needed below
    if user_thread:
        # The coach checks behind this message, kept as its thread.
        append_entries(session, user_message.id, user_thread)

    tokens = reply_tokens(reply_text, base, glosses)
    per_token = token_glosses(tokens, glosses)
    annotations: list[TokenAnnotation] = []
    token_rows: list[MessageToken] = []

    for position, (tok, (word_gloss, word_note)) in enumerate(zip(tokens, per_token)):
        # tok.is_target is a heuristic built for the learner's own possibly-
        # English-mixed input (see lemmatizer.py); it's gated on a small
        # ~300-word frequency list for Spanish, so applying it here to the
        # agent's own reply - which is always in the target language -
        # silently dropped the gloss (and word-bank tracking) for any real
        # word outside that list, e.g. "tormenta". Every word in the agent's
        # reply is in the target language by construction; only
        # punctuation has nothing to gloss.
        if not any(c.isalpha() for c in tok.surface):
            annotations.append(TokenAnnotation(surface=tok.surface, lemma=tok.lemma, pos=tok.pos, gloss="", is_new=False))
            # Persisted too (not just returned live) - a MessageToken row
            # with no gloss, same as any other token with nothing to show.
            # Skipping this used to silently drop every punctuation mark
            # from a message's token array on the very next history
            # reload (GET /chat/history reconstructs tokens purely from
            # these rows), which joinTokens.ts then rendered as words
            # mashed directly together with no sign anything was missing.
            token_rows.append(MessageToken(position=position, surface=tok.surface, lemma=tok.lemma, pos=tok.pos))
            continue

        word_note = word_note if word_gloss else ""
        entry = store.record_exposure(session, tok.lemma, pos=tok.pos, translation=word_gloss, language=base)
        is_new = entry.exposure_count == 1

        annotations.append(
            TokenAnnotation(
                surface=tok.surface,
                lemma=tok.lemma,
                pos=tok.pos,
                gloss=word_gloss,
                is_new=is_new,
                note=word_note,
            )
        )
        token_rows.append(
            MessageToken(
                position=position,
                surface=tok.surface,
                lemma=tok.lemma,
                pos=tok.pos,
                gloss=word_gloss,
                is_new=is_new,
                note=word_note,
            )
        )

    assistant_message = ChatMessage(
        session_id=session_id,
        language=base,
        role="assistant",
        text=reply_text,
        tokens=token_rows,
        native_text=translation,
    )
    if glosses.words:
        save_glosses(assistant_message, glosses)
    session.add(assistant_message)
    vocab_debug: VocabDebug | None = None
    if suggestions is not None:
        session.flush()  # assigns assistant_message.id for the suggestion log
        # Lemmas and surface forms both - a saved word is stored as the
        # learner saw it, which needn't be its lemma.
        reply_words = {tok.lemma for tok in tokens} | {tok.surface.lower() for tok in tokens}
        vocab_suggest.record_outcome(session, session_id, base, assistant_message.id, suggestions, reply_words)
        vocab_debug = VocabDebug.model_validate(suggestions.debug_dict(reply_words))
        assistant_message.vocab_debug = vocab_debug.model_dump_json()
    session.commit()
    session.refresh(assistant_message)

    if suggestions is not None:
        vocab_suggest.schedule_post_turn(
            sessionmaker(bind=session.get_bind()), session_id, base, user_message.text, reply_text
        )

    return ChatTurnResponse(
        message_id=assistant_message.id,
        text=reply_text,
        tokens=annotations,
        translation=translation,
        user_message_id=user_message.id,
        glosses=glosses if glosses.words else None,
        vocab_debug=vocab_debug,
    )


@router.post("/turn", response_model=ChatTurnResponse)
def take_turn(req: ChatTurnRequest, session: Session = Depends(get_session)) -> ChatTurnResponse:
    messages, suggestions = _prepare_turn(session, req)

    try:
        reply_text = model_chat(messages)
    except ModelServerUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    if not reply_text.strip():
        # The model returned a 200 with literally no content - seen live
        # against a reasoning-style model whose internal reasoning tokens
        # can exhaust the whole generation budget before any visible
        # output, on a harder turn (see MAX_REPLY_TOKENS's comment in
        # config.py). Surfacing this as a real error rather than silently
        # persisting a blank assistant turn - a vacant chat bubble the
        # learner can't tell apart from an actual (empty) reply, sitting
        # in history forever once saved.
        raise HTTPException(status_code=502, detail="Model returned an empty reply")

    user_message = ChatMessage(
        session_id=req.session_id, language=get_language(req.language).code, role="user", text=req.message
    )
    return _gloss_and_persist_reply(
        session, req.session_id, reply_text, user_message, req.thread, req.language, suggestions
    )


@router.post("/turn/stream")
def take_turn_stream(req: ChatTurnRequest, session: Session = Depends(get_session)) -> StreamingResponse:
    """Same turn as POST /chat/turn, but the model's reply streams to the
    client as it's generated instead of waiting for the whole thing - for a
    live, typing-style display. Streamed as Server-Sent Events: a "chunk"
    event per token/delta of the reply's own text, then (once generation
    finishes) the exact same gloss/translate/persist work /chat/turn does,
    sent as one final "done" event carrying the exact same ChatTurnResponse
    shape /chat/turn returns - so the frontend ends up with one consistent
    payload either way, just delivered progressively here."""
    messages, suggestions = _prepare_turn(session, req)
    # Not added to the session until the reply is in (see
    # _gloss_and_persist_reply) - no DB write, and so no SQLite write lock,
    # is held while the model streams.
    user_message = ChatMessage(
        session_id=req.session_id, language=get_language(req.language).code, role="user", text=req.message
    )

    def events():
        # A leading SSE comment line (any line starting with ":" - part of
        # the spec, ignored by every conforming client, including this
        # app's own readServerSentEvents) padded well past a typical
        # reverse-proxy buffering threshold (commonly ~1-4KB). Some proxies
        # hold back ANY response below that size regardless of the
        # no-buffering headers above, which only ask nicely - this instead
        # forces an actual flush right at the start, before the real
        # (small, one-token-at-a-time) chunks below would ever cross that
        # threshold on their own.
        yield ": " + ("padding " * 512) + "\n\n"

        chunks: list[str] = []
        try:
            for delta in iter_with_keepalive(model_chat_stream(messages)):
                if delta is None:
                    yield KEEPALIVE_LINE
                    continue
                chunks.append(delta)
                yield _sse("chunk", ChatTurnChunkEvent(delta=delta).model_dump_json())
        except ModelServerUnavailableError as exc:
            yield _sse("error", ChatTurnErrorEvent(message=str(exc)).model_dump_json())
            return

        reply_text = "".join(chunks).strip()
        if not reply_text:
            # Same empty-reply case take_turn guards against (see its own
            # comment) - here it shows up as a stream with zero "chunk"
            # events at all (the frontend's "…" placeholder never fills
            # in), so surface it the same way: an "error" event instead of
            # silently persisting (and then "done"-ing with) a blank
            # assistant turn. The user message is never added in this
            # branch, so it doesn't leave a dangling user-only turn in
            # history either.
            yield _sse("error", ChatTurnErrorEvent(message="Model returned an empty reply").model_dump_json())
            return
        result = _gloss_and_persist_reply(
            session, req.session_id, reply_text, user_message, req.thread, req.language, suggestions
        )
        yield _sse("done", result.model_dump_json())

    return StreamingResponse(events(), media_type="text/event-stream", headers=_SSE_HEADERS)
