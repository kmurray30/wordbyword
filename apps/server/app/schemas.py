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


class ChatHistoryMessage(BaseModel):
    id: int
    role: str
    text: str
    tokens: list[TokenAnnotation] = []


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


class CoachDraftRequest(BaseModel):
    text: str
    session_id: str


class CoachOption(BaseModel):
    formality: str  # "neutral" | "casual" | "formal"
    spanish: str


class CoachDraftResponse(BaseModel):
    # Best-guess English meaning of what the learner is trying to say -
    # empty if the draft was empty.
    meaning: str
    # One short, encouraging note on how apt/correct the attempt was -
    # empty if there's nothing worth flagging.
    feedback: str
    options: list[CoachOption]


class TagInputRequest(BaseModel):
    text: str


class DraftToken(BaseModel):
    # Cheap, LLM-free per-word classification (spaCy tokenize/lemmatize +
    # app.translate.lemmatizer's existing is_spanish heuristic), computed
    # on every /translate/tag-input call regardless of whether the slower
    # LLM-backed `spans` below succeeded. Needed for reward-event tracking
    # (userTypedSpanishWord) and the phrase-selection direction vote.
    surface: str
    lemma: str
    is_spanish: bool
    start: int
    end: int


class DraftSpan(BaseModel):
    # One word, or an LLM-chosen multi-word group (idiom/phrasal verb/fixed
    # expression) the learner is still typing - from app.translate.
    # llm_translate.tag_draft, matched back to exact offsets by
    # app.translate.span_matching. No dual Spanish/English-reading toggle
    # (unlike the old InputTokenAnnotation/TranslationColumn this replaces)
    # - the LLM already has full sentence context, so it picks the one
    # sense that applies here instead of two independent dictionary
    # lookups needing reconciliation in the UI.
    surface: str
    start: int
    end: int
    # False for a span that's already natural, correct Spanish as typed -
    # nothing to replace, candidates[0] is just its in-context gloss. True
    # for a span offering a Spanish replacement to swap in.
    clickable: bool
    candidates: list[TranslateCandidate]


class LlmHealthResponse(BaseModel):
    ready: bool


class TagInputResponse(BaseModel):
    tokens: list[DraftToken]
    spans: list[DraftSpan]


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


class TTSVoicesResponse(BaseModel):
    voices: list[str]
    default: str


class SettingsResponse(BaseModel):
    model_provider: str  # "local" | "openai"
    word_weighting_enabled: bool
    # Whether word_weighting_enabled actually does anything right now -
    # false whenever model_provider isn't "local", regardless of the
    # stored toggle value (see app.settings_store.weighting_active) -
    # logit_bias needs the local model's own tokenizer.
    word_weighting_active: bool


class UpdateSettingsRequest(BaseModel):
    model_provider: str | None = None
    word_weighting_enabled: bool | None = None


class WordBankEntryOut(BaseModel):
    lemma: str
    pos: str
    primary_translation: str
    familiarity: float
    exposure_count: int
    review_interval_days: float

    model_config = {"from_attributes": True}
