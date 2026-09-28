from fastapi import APIRouter

from app.config import NATIVE_LANGUAGE, TARGET_LANGUAGE
from app.schemas import (
    InterpretInputRequest,
    InterpretInputResponse,
    TagInputRequest,
    TagInputResponse,
    TranslateCandidate,
    TranslateTextRequest,
    TranslateTextResponse,
    TranslateWordRequest,
    TranslateWordResponse,
    InputTokenAnnotation,
    TranslationColumn,
)
from app.translate import llm_translate
from app.translate.cognates import COMMON_ES_EN_COGNATES
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


@router.post("/tag-input", response_model=TagInputResponse)
def tag_input(req: TagInputRequest) -> TagInputResponse:
    """Every real word gets at least one translation column, in whichever
    direction spaCy's is_spanish flag implies; a word in COMMON_ES_EN_COGNATES
    gets both directions, since it's a real word in either language and the
    flag can only pick one."""
    tokens = analyze(req.text)
    out: list[InputTokenAnnotation] = []
    for tok in tokens:
        is_word = tok.surface.isalpha() and len(tok.surface) > 1
        is_cognate = is_word and tok.surface.lower() in COMMON_ES_EN_COGNATES

        columns: list[TranslationColumn] = []
        if is_word and (tok.is_spanish or is_cognate):
            columns.append(
                TranslationColumn(
                    language=NATIVE_LANGUAGE,
                    candidates=_word_candidates(tok.lemma, TARGET_LANGUAGE, NATIVE_LANGUAGE),
                )
            )
        if is_word and (not tok.is_spanish or is_cognate):
            columns.append(
                TranslationColumn(
                    language=TARGET_LANGUAGE,
                    candidates=_word_candidates(tok.surface.lower(), NATIVE_LANGUAGE, TARGET_LANGUAGE),
                )
            )

        out.append(
            InputTokenAnnotation(
                surface=tok.surface,
                lemma=tok.lemma,
                is_spanish=tok.is_spanish,
                start=tok.start,
                end=tok.end,
                columns=columns,
            )
        )
    return TagInputResponse(tokens=out)
