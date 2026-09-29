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
    # call failed (gloss then falls back to the older dictionary/MT lookup,
    # which has no such note), or when the word didn't need one.
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
    # string if that call failed (rare; the per-word glosses themselves
    # still fall back to the older dictionary/MT lookup in that case).
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


class TranslateWordRequest(BaseModel):
    word: str
    source_lang: str = "es"


class TranslateCandidate(BaseModel):
    translation: str
    description: str = ""


class TranslateWordResponse(BaseModel):
    word: str
    lemma: str
    candidates: list[TranslateCandidate]


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


class TranslationColumn(BaseModel):
    language: str  # the language these candidates translate INTO ("en" or "es")
    candidates: list[TranslateCandidate]
    # False for a column showing what a word the learner typed already means
    # in Spanish - there's nothing to replace, it's already correct. True for
    # a column offering a Spanish translation to swap in for a word read as
    # English.
    clickable: bool = True


class InputTokenAnnotation(BaseModel):
    surface: str
    lemma: str
    is_spanish: bool
    start: int
    end: int
    # One column per language the word is independently a valid word in (an
    # unclickable Spanish-meaning gloss, a clickable English->Spanish
    # translation, or both for a word valid in both, e.g. "once" - Spanish
    # for "eleven" and English "on one occasion"); a word valid in neither
    # dictionary falls back to a single best-guess column.
    columns: list[TranslationColumn] = []


class LlmHealthResponse(BaseModel):
    ready: bool


class TagInputResponse(BaseModel):
    tokens: list[InputTokenAnnotation]


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


class WordBankEntryOut(BaseModel):
    lemma: str
    pos: str
    primary_translation: str
    familiarity: float
    exposure_count: int
    review_interval_days: float

    model_config = {"from_attributes": True}
