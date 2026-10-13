from datetime import datetime

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


class ThreadEntryIn(BaseModel):
    role: str
    kind: str
    text: str = ""
    # JSON-encoded structured detail (a check's verdict/options/pick).
    data: str = ""


class ThreadEntryOut(ThreadEntryIn):
    id: int


class ChatTurnRequest(BaseModel):
    message: str
    session_id: str
    # The language being learned: "es" | "fr" | "ja" | "ko" | "zh", or for
    # Mandarin "zh-Hans"/"zh-Hant" (the learner's characters) - see
    # app/languages.py.
    language: str = "es"
    # The coach checks the learner ran before sending this message - stored
    # as the sent message's thread so the deliberation stays viewable under
    # it instead of cluttering the main conversation.
    thread: list[ThreadEntryIn] = []


class WordGloss(BaseModel):
    # Character offsets into the message text.
    start: int
    end: int
    # Just that one word's meaning ("good", "to eat").
    gloss: str
    # Optional detail about its form - a verb's infinitive, person, tense.
    grammar: str = ""


class PhraseGloss(BaseModel):
    start: int
    end: int
    # What the whole expression means ("how wonderful").
    gloss: str
    note: str = ""


class MessageGlosses(BaseModel):
    """What a tapped word in a chat message shows: its own meaning (to the
    side) and, if it's part of an expression, the expression's (below).
    See app.translate.llm_translate.gloss_message."""

    words: list[WordGloss] = []
    phrases: list[PhraseGloss] = []


class MessageGlossesRequest(BaseModel):
    message_id: int


class ChatTurnResponse(BaseModel):
    message_id: int
    text: str
    tokens: list[TokenAnnotation]
    # From the same LLM call that produced `tokens`' per-word glosses (see
    # app.translate.llm_translate.gloss_message) - guaranteed to agree with
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
    glosses: MessageGlosses | None = None


class SaveUserInterpretationRequest(BaseModel):
    message_id: int
    native: str
    target: str


class SaveUserInterpretationResponse(BaseModel):
    ok: bool


# POST /chat/turn/stream sends these as SSE events rather than a single JSON
# response (see app/routes/chat.py's take_turn_stream) - "chunk" events
# stream the reply's own text as the model generates it, for a live
# typing-style display; once generation finishes, gloss_message (which needs
# the complete text) runs and the full, final payload - the exact same
# shape the non-streaming POST /chat/turn returns - is sent as one "done"
# event.
class ChatTurnChunkEvent(BaseModel):
    delta: str


class ChatTurnErrorEvent(BaseModel):
    message: str


class ChatHistoryMessage(BaseModel):
    id: int
    role: str
    text: str
    tokens: list[TokenAnnotation] = []
    # The saved translation (see ChatMessage.native_text/target_text): for a
    # user row its native/target interpretation, for an assistant row
    # `native` is the reply's translation. Empty if not known yet.
    native: str = ""
    target: str = ""
    thread: list[ThreadEntryOut] = []
    # Saved word/phrase glosses; null if not made yet (POST
    # /chat/message/glosses makes and saves them).
    glosses: MessageGlosses | None = None


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
    # The language being learned: "es" | "fr" | "ja" | "ko" | "zh", or for
    # Mandarin "zh-Hans"/"zh-Hant" (the learner's characters) - see
    # app/languages.py.
    language: str = "es"


class InterpretInputResponse(BaseModel):
    native: str
    target: str


class TagInputRequest(BaseModel):
    text: str
    # The language being learned: "es" | "fr" | "ja" | "ko" | "zh", or for
    # Mandarin "zh-Hans"/"zh-Hant" (the learner's characters) - see
    # app/languages.py.
    language: str = "es"


class DraftToken(BaseModel):
    # Cheap, LLM-free per-word classification (app.translate.lemmatizer:
    # spaCy for Spanish, a plain tokenizer otherwise) from
    # /translate/tag-input - fast enough to keep calling on every keystroke
    # (debounced), unlike the LLM-backed per-word/group glossing
    # (GlossSpansResponse.spans below), which is fetched lazily instead.
    # Needed for reward-event tracking (userTypedTargetWord) and the
    # phrase-selection direction vote.
    surface: str
    lemma: str
    # In the language being learned, not an English word dropped in.
    is_target: bool
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
    # False for a span that's already natural, correct target language as
    # typed - nothing to replace, candidates[0] is just its in-context
    # gloss. True for a span offering a replacement to swap in. Always the
    # English -> target direction - the default/primary reading.
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
    # The language being learned: "es" | "fr" | "ja" | "ko" | "zh", or for
    # Mandarin "zh-Hans"/"zh-Hant" (the learner's characters) - see
    # app/languages.py.
    language: str = "es"


class GlossSpansResponse(BaseModel):
    spans: list[DraftSpan]


class DraftTurn(BaseModel):
    role: str
    text: str


class CoachDraftRequest(BaseModel):
    text: str
    session_id: str
    # The language being learned: "es" | "fr" | "ja" | "ko" | "zh", or for
    # Mandarin "zh-Hans"/"zh-Hant" (the learner's characters) - see
    # app/languages.py.
    language: str = "es"
    # The message this conversation is about (opened from its 💬), if any;
    # otherwise it's about the learner's next message.
    thread_message_id: int | None = None
    # Whatever is in the message box right now (typed, not sent) - so "is
    # that right?" can be about it.
    message_draft: str = ""
    # The learner's discussion with the coach about the message they're
    # about to send (not yet saved anywhere), as (role, text) turns with
    # role "learner" | "coach".
    draft_thread: list[DraftTurn] = []
    # True when `text` is the exact message in the message box (💡 or a
    # finished dictation there), rather than talk in the coach panel.
    check: bool = False


# /translate/coach streams these as SSE events (see
# app.translate.llm_translate.coach_reply_stream): which message the reply
# is about, whether the learner's own target-language text is ready to send, then the
# coach's reply text (suggestions inside it as [[target language | English]]).
class CoachAboutEvent(BaseModel):
    message_id: int | None  # None = the learner's next message


class CoachReadyEvent(BaseModel):
    ready: bool


class CoachReplyChunkEvent(BaseModel):
    delta: str


class CoachReplyCompleteEvent(BaseModel):
    reply: str
    # The question/reply entries saved on the message it was about (empty
    # when it's about the learner's next message).
    entries: list[ThreadEntryOut] = []


class CoachErrorEvent(BaseModel):
    message: str


class RewardEventRequest(BaseModel):
    event_type: str  # "wordSeenNoHover" | "wordHovered" | "userTypedTargetWord"
    lemma: str
    # The language being learned: "es" | "fr" | "ja" | "ko" | "zh", or for
    # Mandarin "zh-Hans"/"zh-Hant" (the learner's characters) - see
    # app/languages.py.
    language: str = "es"


class RewardEventResponse(BaseModel):
    lemma: str
    familiarity: float
    review_interval_days: float


class TTSRequest(BaseModel):
    text: str
    # The text's language. ElevenLabs' multilingual voices read anything;
    # Kokoro's are one language each, so it picks the Kokoro voice.
    language: str = "es"
    # A voice id from /tts/voices (for the same provider). None: the role's
    # voice.
    voice: str | None = None
    # "partner" (default): the conversation partner's voice (`voice`, else
    # the partner default). "coach": the coach's own voice; `voice` is
    # ignored for it.
    role: str = "partner"
    # "elevenlabs" | "kokoro"; None: the server's TTS_PROVIDER.
    provider: str | None = None


class TTSVoice(BaseModel):
    id: str
    name: str


class TTSVoicesResponse(BaseModel):
    provider: str
    voices: list[TTSVoice]
    default: str


class WordBankEntryOut(BaseModel):
    lemma: str
    pos: str
    primary_translation: str
    familiarity: float
    exposure_count: int
    review_interval_days: float

    model_config = {"from_attributes": True}


class SavedWordIn(BaseModel):
    session_id: str
    word: str
    gloss: str = ""
    grammar: str = ""
    sentence: str = ""
    message_id: int | None = None


class SavedWordOut(BaseModel):
    id: int
    word: str
    gloss: str
    grammar: str
    sentence: str
    message_id: int | None
    created_at: datetime


class SavedWordsResponse(BaseModel):
    words: list[SavedWordOut]


class RemoveSavedWordResponse(BaseModel):
    removed: int


class ReadingsRequest(BaseModel):
    texts: list[str]
    # As on every request (see ChatTurnRequest.language) - for Mandarin,
    # "zh-Hant" also returns the text in Traditional characters.
    language: str = "es"


class ReadingOut(BaseModel):
    start: int
    end: int
    # Romanization (romaji, pinyin, Korean romanization).
    roman: str
    # Japanese: the kana over a word written in kanji (furigana).
    kana: str = ""
    # Same word as the reading before it - romanized without a space.
    cont: bool = False


class TextReadings(BaseModel):
    # The text as it should be shown - differs from what was sent only for
    # Mandarin in the other script (same length, same offsets).
    text: str
    readings: list[ReadingOut]


class ReadingsResponse(BaseModel):
    items: list[TextReadings]
