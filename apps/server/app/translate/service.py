from app.config import TARGET_LANGUAGE
from app.translate import dictionary, mt, wiktionary_dict


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
    """Dictionary-first, Argos-MT-last-resort lookup shared by the
    /translate routes and the chat turn pipeline (which glosses every word
    the agent produces).

    Three tiers, in order:
    1. The small hand-curated dictionary (dictionary.py) - only for ES->EN,
       only ~30 words, but with hand-written sense descriptions.
    2. The bundled Wiktionary dataset (wiktionary_dict.py) - built at Docker
       build time from real dictionary data (definitions, usage examples,
       colloquial/slang register tags), covering both directions and,
       unlike Argos, offering genuinely multiple candidates per word.
    3. Argos Translate, as a last resort for words neither dictionary
       covers. Argos has no way to say "I don't know this word" - for an
       out-of-vocabulary word (slang like "bro"/"sup" that never appeared in
       its formal training corpus) it just echoes the input back unchanged,
       which would otherwise look like a real (wrong) translation. Detected
       and reported as "no translation found" instead of silently shown.
    """
    if source_lang == TARGET_LANGUAGE:
        entries = dictionary.lookup(lemma)
        if entries:
            return entries
        entries = wiktionary_dict.lookup_es_to_en(lemma)
        if entries:
            return entries
    else:
        entries = wiktionary_dict.lookup_en_to_es(lemma)
        if entries:
            return entries

    try:
        translation = mt.translate_word(lemma, source_lang, target_lang)
    except mt.TranslationUnavailableError as exc:
        return [{"translation": "", "description": str(exc)}]
    normalized = _normalize_word_translation(translation)
    if normalized.lower() == lemma.lower():
        # Argos's "I don't recognize this word" behavior is to echo it back
        # unchanged rather than error - without this check that looks
        # exactly like a (wrong) real translation instead of a miss.
        return [{"translation": "", "description": f'no translation found for "{lemma}"'}]
    return [{"translation": normalized, "description": ""}]


def gloss(lemma: str, source_lang: str, target_lang: str) -> str:
    """Single best-guess gloss (first candidate) - used where we just need
    one string, e.g. per-token annotations on a chat message."""
    candidates = word_candidates(lemma, source_lang, target_lang)
    return candidates[0]["translation"] if candidates else ""
