from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.orm import Session

from app import settings_store
from app.config import NATIVE_LANGUAGE, TARGET_LANGUAGE
from app.db import get_session
from app.models import ChatMessage
from app.schemas import (
    CoachDraftRequest,
    CoachDraftResponse,
    CoachOption,
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
from app.translate.lemmatizer import analyze
from app.translate.service import word_candidates
from app.translate.word_validity import is_valid_english_word, is_valid_spanish_word

_COACH_HISTORY_TURNS = 8

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
def translate_text(req: TranslateTextRequest, session: Session = Depends(get_session)) -> TranslateTextResponse:
    provider = settings_store.get_settings(session).model_provider
    try:
        translation = llm_translate.translate_text(req.text, req.source_lang, req.target_lang, provider=provider)
    except llm_translate.TranslationUnavailableError as exc:
        return TranslateTextResponse(translation=f"(translation unavailable: {exc})")
    return TranslateTextResponse(translation=translation)


@router.post("/interpret", response_model=InterpretInputResponse)
def interpret_input(req: InterpretInputRequest, session: Session = Depends(get_session)) -> InterpretInputResponse:
    """For the learner's own message: infers what they meant across a
    possible mix of English/Spanish and grammar mistakes, returning a
    corrected English restatement alongside its Spanish translation - used
    to show both under the user's chat bubble rather than a single literal
    (and possibly nonsensical) pass."""
    provider = settings_store.get_settings(session).model_provider
    try:
        native, target = llm_translate.interpret_user_input(
            req.text, NATIVE_LANGUAGE, TARGET_LANGUAGE, provider=provider
        )
    except llm_translate.TranslationUnavailableError as exc:
        msg = f"(translation unavailable: {exc})"
        return InterpretInputResponse(native=msg, target=msg)
    return InterpretInputResponse(native=native, target=target)


@router.post("/coach", response_model=CoachDraftResponse)
def coach_draft(req: CoachDraftRequest, session: Session = Depends(get_session)) -> CoachDraftResponse:
    """For a message the learner is still drafting (not yet sent) - unlike
    /interpret above, which corrects a single message in isolation, this
    pulls the session's recent conversation for context so the suggested
    phrasing actually fits the tone of what's been said so far, and returns
    feedback on the attempt plus phrasing options at a few formality
    levels rather than a single flat correction."""
    provider = settings_store.get_settings(session).model_provider
    history_rows = session.scalars(
        select(ChatMessage)
        .where(ChatMessage.session_id == req.session_id)
        .order_by(ChatMessage.id.desc())
        .limit(_COACH_HISTORY_TURNS)
    ).all()
    history = [(row.role, row.text) for row in reversed(history_rows)]

    try:
        meaning, feedback, options = llm_translate.coach_draft(
            req.text, history, NATIVE_LANGUAGE, TARGET_LANGUAGE, provider=provider
        )
    except llm_translate.TranslationUnavailableError as exc:
        msg = f"(translation unavailable: {exc})"
        return CoachDraftResponse(meaning=msg, feedback="", options=[CoachOption(formality="neutral", spanish=msg)])
    return CoachDraftResponse(
        meaning=meaning,
        feedback=feedback,
        options=[CoachOption(formality=f, spanish=s) for f, s in options],
    )


@router.post("/tag-input", response_model=TagInputResponse)
def tag_input(req: TagInputRequest) -> TagInputResponse:
    """The learner is assumed to be writing Spanish by default: every real
    word is checked independently against both languages' dictionaries
    (word_validity), not classified into a single Spanish-or-English bucket.
    A word already valid Spanish gets an unclickable EN gloss (it's correct
    as typed, nothing to replace); a word valid English gets a clickable ES
    translation (swaps it in place); a word valid in both - e.g. "once",
    Spanish for "eleven" and also an English word - gets both, independently.
    A word in neither dictionary (typo, name, slang) falls back to the
    morphological is_spanish guess for a single best-effort column."""
    tokens = analyze(req.text)
    out: list[InputTokenAnnotation] = []
    for tok in tokens:
        is_word = tok.surface.isalpha() and len(tok.surface) > 1
        columns: list[TranslationColumn] = []
        spanish_valid = False

        if is_word:
            spanish_valid = is_valid_spanish_word(tok.surface)
            english_valid = is_valid_english_word(tok.surface)
            if not spanish_valid and not english_valid:
                # Neither dictionary recognizes it - fall back to the
                # morphological guess rather than showing nothing.
                spanish_valid = tok.is_spanish
                english_valid = not tok.is_spanish

            if spanish_valid:
                columns.append(
                    TranslationColumn(
                        language=NATIVE_LANGUAGE,
                        clickable=False,
                        candidates=_word_candidates(tok.lemma, TARGET_LANGUAGE, NATIVE_LANGUAGE),
                    )
                )
            if english_valid:
                columns.append(
                    TranslationColumn(
                        language=TARGET_LANGUAGE,
                        clickable=True,
                        candidates=_word_candidates(tok.surface.lower(), NATIVE_LANGUAGE, TARGET_LANGUAGE),
                    )
                )

        out.append(
            InputTokenAnnotation(
                surface=tok.surface,
                lemma=tok.lemma,
                is_spanish=is_word and spanish_valid,
                start=tok.start,
                end=tok.end,
                columns=columns,
            )
        )
    return TagInputResponse(tokens=out)
