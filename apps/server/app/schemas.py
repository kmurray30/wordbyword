from pydantic import BaseModel


class TokenAnnotation(BaseModel):
    surface: str
    lemma: str
    pos: str
    gloss: str
    is_new: bool


class ChatTurnRequest(BaseModel):
    message: str
    session_id: str


class ChatTurnResponse(BaseModel):
    message_id: int
    text: str
    tokens: list[TokenAnnotation]


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


class InputTokenAnnotation(BaseModel):
    surface: str
    lemma: str
    is_spanish: bool
    start: int
    end: int
    # Usually one column (the direction implied by is_spanish); two when the
    # word is a known cross-language cognate (see app/translate/cognates.py).
    columns: list[TranslationColumn] = []


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
