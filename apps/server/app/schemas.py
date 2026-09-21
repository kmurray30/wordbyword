from pydantic import BaseModel


class TokenAnnotation(BaseModel):
    surface: str
    lemma: str
    pos: str
    gloss: str
    is_new: bool


class ChatTurnRequest(BaseModel):
    message: str


class ChatTurnResponse(BaseModel):
    message_id: int
    text: str
    tokens: list[TokenAnnotation]


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


class TagInputRequest(BaseModel):
    text: str


class InputTokenAnnotation(BaseModel):
    surface: str
    lemma: str
    is_spanish: bool
    start: int
    end: int
    candidates: list[TranslateCandidate] = []


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


class WordBankEntryOut(BaseModel):
    lemma: str
    pos: str
    primary_translation: str
    familiarity: float
    exposure_count: int
    review_interval_days: float

    model_config = {"from_attributes": True}
