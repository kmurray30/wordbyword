"""Offline machine translation via Argos Translate, used for whole-sentence
translation and as the fallback for single words the curated dictionary
doesn't cover.

Requires the en<->es language packages to be installed once - see
scripts/install_translate_models.py.
"""

from __future__ import annotations

from functools import lru_cache


class TranslationUnavailableError(RuntimeError):
    pass


@lru_cache(maxsize=1)
def _installed_languages():
    import argostranslate.translate as argos_translate

    return argos_translate.get_installed_languages()


def _get_translation(source_lang: str, target_lang: str):
    languages = _installed_languages()
    from_lang = next((l for l in languages if l.code == source_lang), None)
    to_lang = next((l for l in languages if l.code == target_lang), None)
    if from_lang is None or to_lang is None:
        raise TranslationUnavailableError(
            f"Argos Translate language package for {source_lang}->{target_lang} is not "
            "installed. Run: python scripts/install_translate_models.py"
        )
    translation = from_lang.get_translation(to_lang)
    if translation is None:
        raise TranslationUnavailableError(f"No installed Argos package translates {source_lang}->{target_lang}")
    return translation


def translate_text(text: str, source_lang: str, target_lang: str) -> str:
    if not text.strip():
        return ""
    translation = _get_translation(source_lang, target_lang)
    return translation.translate(text)


def translate_word(word: str, source_lang: str, target_lang: str) -> str:
    return translate_text(word, source_lang, target_lang)
