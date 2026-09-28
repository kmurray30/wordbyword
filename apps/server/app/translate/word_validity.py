"""Independent "is this a real word in language X" checks, backed by
pyspellchecker's bundled frequency dictionaries (offline, no network at
runtime). Used by /translate/tag-input to decide, per word the learner
typed, which reading(s) - Spanish, English, or both - are worth showing a
translation for. Deliberately independent per language rather than a single
binary classifier: a word like "once" is valid in both (Spanish "eleven",
English "on one occasion"), and neither check should have to know about the
other to get that right.
"""

from functools import lru_cache


@lru_cache(maxsize=1)
def _es_checker():
    from spellchecker import SpellChecker

    return SpellChecker(language="es")


@lru_cache(maxsize=1)
def _en_checker():
    from spellchecker import SpellChecker

    return SpellChecker(language="en")


def is_valid_spanish_word(word: str) -> bool:
    return word.lower() in _es_checker()


def is_valid_english_word(word: str) -> bool:
    return word.lower() in _en_checker()
