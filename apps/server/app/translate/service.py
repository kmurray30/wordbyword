from app.config import TARGET_LANGUAGE
from app.translate import dictionary, mt


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
    return [{"translation": translation, "description": ""}]


def gloss(lemma: str, source_lang: str, target_lang: str) -> str:
    """Single best-guess gloss (first candidate) - used where we just need
    one string, e.g. per-token annotations on a chat message."""
    candidates = word_candidates(lemma, source_lang, target_lang)
    return candidates[0]["translation"] if candidates else ""
