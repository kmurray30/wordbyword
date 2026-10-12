from pydantic import BaseModel


class TokenAnnotation(BaseModel):
    surface: str
    lemma: str
    pos: str
    gloss: str
    is_new: bool
    # From the same LLM call as ChatTurnResponse.translation - a short note
    # on why this word's sense applies here, when it could otherwise be
    # confused with a different one. Empty for punctuation, when the LLM
    # call missed this word or failed outright (gloss is then also empty -
    # no dictionary/MT fallback), or when the word didn't need a note.
    note: str = ""
    # From the same call - non-empty only when this token is part of a
    # multi-word group (an idiom/phrasal verb/fixed expression, e.g. "tener
    # en cuenta"): a short word-by-word breakdown of how the group's
    # individual words combine into its overall gloss, shared across every
    # token the group covers (see app.routes.chat's _match_token_glosses).
    # Empty for a standalone word - gloss already says everything literal
    # would repeat.
    literal: str = ""


class ChatTurnRequest(BaseModel):
    message: str
    session_id: str


class ChatTurnResponse(BaseModel):
    message_id: int
    text: str
    tokens: list[TokenAnnotation]
    # From the same LLM call that produced `tokens`' per-word glosses (see
    # app.translate.llm_translate.gloss_reply) - guaranteed to agree with
    # them on word sense, unlike a translation fetched separately. Empty
    # string if that call failed (rare; the per-word glosses are then also
    # empty - no dictionary/MT fallback).
    translation: str = ""
    # The id this same turn's own user-sent message was persisted under -
    # the frontend already fetches that message's native/target
    # interpretation live (via /translate/interpret, fired the moment it's
    # sent, independent of this call) and needs this id to persist that
    # result back (POST /chat/message/interpretation) so it survives a
    # reload. Not computed here directly - doing so would add a second,
    # serial LLM call to every turn's latency for no benefit, since the
    # frontend already has its own copy on the way.
    user_message_id: int = 0
    # Whether sending this turn closed out an open brainstorm log (now
    # attached to user_message_id) - the frontend then asks
    # POST /brainstorm/fold to turn it into notes.
    has_brainstorm: bool = False


class SaveUserInterpretationRequest(BaseModel):
    message_id: int
    native: str
    target: str


class SaveUserInterpretationResponse(BaseModel):
    ok: bool


# POST /chat/turn/stream sends these as SSE events rather than a single JSON
# response (see app/routes/chat.py's take_turn_stream) - "chunk" events
# stream the reply's own text as the model generates it, for a live
# typing-style display; once generation finishes, gloss_reply (which needs
# the complete text) runs and the full, final payload - the exact same
# shape the non-streaming POST /chat/turn returns - is sent as one "done"
# event.
class ChatTurnChunkEvent(BaseModel):
    delta: str


class ChatTurnErrorEvent(BaseModel):
    message: str


class BrainstormLogTurn(BaseModel):
    id: int
    role: str  # "learner" | "coach"
    text: str
    draft: str = ""


class ChatHistoryMessage(BaseModel):
    id: int
    role: str
    text: str
    tokens: list[TokenAnnotation] = []
    # Only meaningful for role == "user" (see ChatMessage.native_text/
    # target_text) - persisted so the frontend can hydrate its two
    # translation rows straight from history instead of re-fetching
    # /translate/interpret (and showing "Translating…" indefinitely while
    # that fetch is stuck/slow) on every reload. Empty for an assistant row,
    # or if the interpret call failed when this message was first sent.
    native: str = ""
    target: str = ""
    # Folded study notes shown in this bubble's Notes dropdown (either role).
    notes: list[str] = []
    # role == "user" only: the raw brainstorm log that led up to this
    # message, and whether it still needs folding into notes.
    brainstorm: list[BrainstormLogTurn] = []
    notes_pending: bool = False


class ChatHistoryResponse(BaseModel):
    messages: list[ChatHistoryMessage]


class ClearHistoryResponse(BaseModel):
    cleared: int


class TranslateCandidate(BaseModel):
    translation: str
    description: str = ""


class TranslateTextRequest(BaseModel):
    text: str
    source_lang: str = "es"
    target_lang: str = "en"


class TranslateTextResponse(BaseModel):
    translation: str


class InterpretInputRequest(BaseModel):
    text: str


class InterpretInputResponse(BaseModel):
    native: str
    target: str


class TagInputRequest(BaseModel):
    text: str


class DraftToken(BaseModel):
    # Cheap, LLM-free per-word classification (spaCy tokenize/lemmatize +
    # app.translate.lemmatizer's existing is_spanish heuristic) from
    # /translate/tag-input - fast enough to keep calling on every keystroke
    # (debounced), unlike the LLM-backed per-word/group glossing
    # (GlossSpansResponse.spans below), which is fetched lazily instead.
    # Needed for reward-event tracking (userTypedSpanishWord) and the
    # phrase-selection direction vote.
    surface: str
    lemma: str
    is_spanish: bool
    start: int
    end: int


class DraftSpan(BaseModel):
    # One word, or an LLM-chosen multi-word group (idiom/phrasal verb/fixed
    # expression) the learner is still typing - from app.translate.
    # llm_translate.tag_draft, matched back to exact offsets by
    # app.translate.span_matching.
    surface: str
    start: int
    end: int
    # False for a span that's already natural, correct Spanish as typed -
    # nothing to replace, candidates[0] is just its in-context gloss. True
    # for a span offering a Spanish replacement to swap in. Always the
    # EN->ES direction - the default/primary reading.
    clickable: bool
    candidates: list[TranslateCandidate]
    # Non-empty only for a standalone word that's ALSO a legitimate,
    # different word in the other language (a true cross-language cognate,
    # e.g. "once" - Spanish for "eleven", also an English word) - its
    # meaning read as that other word, in English. The frontend shows a
    # low-profile toggle between the primary (candidates/clickable) and
    # this alternate reading only when this is non-empty. Empty for the
    # overwhelming majority of spans, which have no such ambiguity.
    alternate_gloss: str = ""
    # Non-empty only for a multi-word span (an idiom/phrasal verb/fixed
    # expression, e.g. "tener en cuenta") - a short word-by-word breakdown
    # of how the group's individual words combine into candidates[0]'s
    # overall gloss/translation, so the learner can see how the phrase is
    # built rather than just what it means as a whole. Empty for a single-
    # word span, where it would just repeat the gloss.
    literal: str = ""


class LlmHealthResponse(BaseModel):
    ready: bool


class TagInputResponse(BaseModel):
    tokens: list[DraftToken]


class GlossSpansRequest(BaseModel):
    text: str


class GlossSpansResponse(BaseModel):
    spans: list[DraftSpan]


# The Brainstorm field's coach (POST /brainstorm/turn, streamed as SSE - see
# app/routes/brainstorm.py and app.translate.llm_translate.brainstorm_stream).
class VoiceWord(BaseModel):
    word: str
    confidence: float


class BrainstormTurnRequest(BaseModel):
    session_id: str
    # What the learner just typed or said into the Brainstorm field (may be
    # mixed Spanish/English; if spoken, may contain mis-heard words).
    text: str
    # Per-word recognizer confidence for spoken input, so the coach can
    # call out likely mis-hearings. Empty for typed input.
    words: list[VoiceWord] = []
    # What's in the Reply box right now, so "is this right?" can be about it.
    reply_draft: str = ""
    source: str = "text"  # "text" | "voice"


class BrainstormIntentEvent(BaseModel):
    # attempt | question | clarify | confirm | pick | cancel | translate_last
    intent: str


class BrainstormVerdictEvent(BaseModel):
    verdict: str  # "ready" | "revise" | "none"


class BrainstormDraftEvent(BaseModel):
    # The single best Spanish sentence for the Reply box ("" if none).
    draft: str


class BrainstormSayChunkEvent(BaseModel):
    delta: str


class BrainstormSayCompleteEvent(BaseModel):
    say: str


class BrainstormSavedEvent(BaseModel):
    learner_turn_id: int
    coach_turn_id: int


class BrainstormErrorEvent(BaseModel):
    message: str


class BrainstormOpenResponse(BaseModel):
    turns: list[BrainstormLogTurn]


class FoldBrainstormRequest(BaseModel):
    session_id: str
    user_message_id: int


class MessageNotesOut(BaseModel):
    message_id: int
    notes: list[str]


class FoldBrainstormResponse(BaseModel):
    # One entry per bubble that has notes from this fold (the learner's
    # reply, and/or the partner message before it). Empty if there was
    # nothing worth noting.
    messages: list[MessageNotesOut]


class RewardEventRequest(BaseModel):
    event_type: str  # "wordSeenNoHover" | "wordHovered" | "userTypedSpanishWord"
    lemma: str


class RewardEventResponse(BaseModel):
    lemma: str
    familiarity: float
    review_interval_days: float


class TTSRequest(BaseModel):
    text: str
    language: str = "es"
    voice: str | None = None
    # "partner" (default): the conversation partner's Kokoro voice, picked
    # by `language`/`voice` as before. "coach": voice mode's coach voice
    # (ElevenLabs multilingual - English with embedded Spanish phrases);
    # `language`/`voice` are ignored for it.
    role: str = "partner"


class TTSVoicesResponse(BaseModel):
    voices: list[str]
    default: str


class WordBankEntryOut(BaseModel):
    lemma: str
    pos: str
    primary_translation: str
    familiarity: float
    exposure_count: int
    review_interval_days: float

    model_config = {"from_attributes": True}
