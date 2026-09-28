"""Small curated EN->ES dictionary for common informal English words and
greetings that Wiktionary's crowd-sourced translation tables don't reliably
cover - Wiktionary's data is thorough for formal vocabulary but thin for
slang (its own translation tables can be entirely empty for a word, or
carry only one weak candidate where several better ones exist). Checked
before the bundled Wiktionary dataset, same spirit as dictionary.py's
ES->EN override for common words worth hand-tuning."""

from functools import lru_cache

from app.translate.text_normalize import strip_accents

DICTIONARY_EN_ES: dict[str, list[dict[str, str]]] = {
    "sup": [
        {"translation": "qué tal", "description": "informal greeting, \"how's it going\""},
        {"translation": "qué onda", "description": "informal greeting, common in Latin America"},
        {"translation": "qué pasa", "description": "informal greeting, \"what's up\""},
    ],
    "bro": [
        {"translation": "bro", "description": "also used as-is, borrowed directly into Spanish slang"},
        {"translation": "tío", "description": "informal for \"dude\"/\"guy\", mainly used in Spain"},
        {"translation": "hermano", "description": "literally \"brother\", used affectionately for a close friend"},
        {"translation": "compa", "description": "informal, short for \"compañero\" (\"buddy\")"},
    ],
}


@lru_cache(maxsize=1)
def _normalized_index() -> dict[str, list[dict[str, str]]]:
    return {strip_accents(k): v for k, v in DICTIONARY_EN_ES.items()}


def lookup(lemma: str) -> list[dict[str, str]]:
    return _normalized_index().get(strip_accents(lemma.lower()), [])
