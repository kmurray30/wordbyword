from fastapi import APIRouter

from app.config import NATIVE_LANGUAGE, TARGET_LANGUAGE
from app.schemas import (
    TagInputRequest,
    TagInputResponse,
    TranslateCandidate,
    TranslateTextRequest,
    TranslateTextResponse,
    TranslateWordRequest,
    TranslateWordResponse,
    InputTokenAnnotation,
)
from app.translate import mt
from app.translate.lemmatizer import analyze
from app.translate.service import word_candidates

router = APIRouter(prefix="/translate", tags=["translate"])


def _word_candidates(lemma: str, source_lang: str, target_lang: str) -> list[TranslateCandidate]:
    return [TranslateCandidate(**c) for c in word_candidates(lemma, source_lang, target_lang)]


@router.post("/word", response_model=TranslateWordResponse)
def translate_word(req: TranslateWordRequest) -> TranslateWordResponse:
    target_lang = NATIVE_LANGUAGE if req.source_lang == TARGET_LANGUAGE else TARGET_LANGUAGE
    tokens = analyze(req.word)
    lemma = tokens[0].lemma if tokens else req.word.lower()
    candidates = _word_candidates(lemma, req.source_lang, target_lang)
    return TranslateWordResponse(word=req.word, lemma=lemma, candidates=candidates)


@router.post("/text", response_model=TranslateTextResponse)
def translate_text(req: TranslateTextRequest) -> TranslateTextResponse:
    try:
        translation = mt.translate_text(req.text, req.source_lang, req.target_lang)
    except mt.TranslationUnavailableError as exc:
        return TranslateTextResponse(translation=f"(translation unavailable: {exc})")
    return TranslateTextResponse(translation=translation)


@router.post("/tag-input", response_model=TagInputResponse)
def tag_input(req: TagInputRequest) -> TagInputResponse:
    tokens = analyze(req.text)
    out: list[InputTokenAnnotation] = []
    for tok in tokens:
        candidates: list[TranslateCandidate] = []
        if not tok.is_spanish:
            candidates = _word_candidates(tok.surface.lower(), NATIVE_LANGUAGE, TARGET_LANGUAGE)
        out.append(
            InputTokenAnnotation(
                surface=tok.surface,
                lemma=tok.lemma,
                is_spanish=tok.is_spanish,
                start=tok.start,
                end=tok.end,
                candidates=candidates,
            )
        )
    return TagInputResponse(tokens=out)
