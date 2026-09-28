from app.config import TARGET_LANGUAGE
from app.translate import dictionary, mt


def _normalize_word_translation(text: str) -> str:
    """Argos's word-level lookup is actually its sentence-level MT run on a
    single word, so it can come back capitalized and/or with trailing
    sentence punctuation (e.g. "hello" -> "Hola."). Harmless for a gloss
    shown on its own, but wrong when a caller splices this straight into the
    middle of an existing sentence - which the input's word-hover replace
    feature does. Curated dictionary entries don't need this; they're
    already clean, hand-written lowercase words/phrases."""
    text = text.strip().rstrip(".!?,;:")
    if text and text[0].isupper() and not text[1:].isupper():
        text = text[0].lower() + text[1:]
    return text


def word_candidates(lemma: str, source_lang: str, target_lang: str) -> list[dict[str, str]]:
    """Dictionary-first, Argos-MT-fallback lookup shared by the /translate
    routes and the chat turn pipeline (which glosses every word the agent
    produces)."""
    if source_lang == TARGET_LANGUAGE:
        entries = dictionary.lookup(lemma)
        if entries:
            return entries
    try:
        translation = mt.translate_word(lemma, source_lang, target_lang)
    except mt.TranslationUnavailableError as exc:
        return [{"translation": "", "description": str(exc)}]
    return [{"translation": _normalize_word_translation(translation), "description": ""}]


def gloss(lemma: str, source_lang: str, target_lang: str) -> str:
    """Single best-guess gloss (first candidate) - used where we just need
    one string, e.g. per-token annotations on a chat message."""
    candidates = word_candidates(lemma, source_lang, target_lang)
    return candidates[0]["translation"] if candidates else ""
