"""Splits text into words, with a base form for the word bank and whether
each word is in the language being learned (vs an English word the learner
dropped in).

Spanish goes through spaCy (`es_core_news_sm`, downloaded once:
`python -m spacy download es_core_news_sm`), which also lemmatizes - the
word bank then tracks "comer", not "comimos". Every other language is split
by script with a plain tokenizer: Japanese and Chinese have no spaces, so a
"word" there is a run of one script, which is rough - the real word
boundaries come from the LLM's glosses wherever they exist (see
app.routes.chat), this is only the fallback and the fast per-keystroke pass.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

from app.languages import HAN_RE, HANGUL_RE, KANA_RE, get_language
from app.translate.frequency_es_set import KNOWN_SPANISH_LEMMAS
from app.translate.known_words_fr import KNOWN_FRENCH_WORDS
from app.translate.text_normalize import strip_accents


@dataclass
class Token:
    surface: str
    lemma: str
    pos: str
    # In the language being learned (punctuation and numbers count as yes:
    # nothing to flag).
    is_target: bool
    start: int
    end: int


@lru_cache(maxsize=1)
def _nlp():
    import spacy

    try:
        return spacy.load("es_core_news_sm")
    except OSError as exc:  # pragma: no cover - environment-dependent
        raise RuntimeError(
            "spaCy model 'es_core_news_sm' is not installed. Run: "
            "python -m spacy download es_core_news_sm"
        ) from exc


def _is_spanish(surface: str, lemma: str) -> bool:
    if not surface.isalpha():
        return True  # punctuation/numbers: don't flag as "unknown English"
    if lemma.lower() in KNOWN_SPANISH_LEMMAS:
        return True
    if lemma.lower() != surface.lower():
        # spaCy successfully derived a different base form (conjugation,
        # pluralization, etc.) - strong evidence this is Spanish morphology.
        return True
    return False


def _analyze_spanish(text: str) -> list[Token]:
    doc = _nlp()(text)
    tokens: list[Token] = []
    for tok in doc:
        if tok.is_space:
            continue
        lemma = tok.lemma_ or tok.text
        tokens.append(
            Token(
                surface=tok.text,
                lemma=lemma.lower(),
                pos=tok.pos_,
                is_target=_is_spanish(tok.text, lemma),
                start=tok.idx,
                end=tok.idx + len(tok.text),
            )
        )
    return tokens


def _script_class(ch: str) -> str:
    if HAN_RE.match(ch):
        return "han"
    if KANA_RE.match(ch):
        # Hiragana and katakana words are told apart (ー belongs to katakana).
        return "katakana" if "゠" <= ch <= "ヿ" or ch == "ー" else "hiragana"
    if HANGUL_RE.match(ch):
        return "hangul"
    if ch.isalpha() or ch.isdigit():
        return "alnum"
    if ch.isspace():
        return "space"
    return "punct"


# Letters only French uses (an English word never has them).
_FRENCH_ONLY = set("àâçèêëîïôûùœæÿ")


def _is_target_word(surface: str, script: str, code: str) -> bool:
    if not any(c.isalpha() for c in surface):
        return True
    lang = get_language(code)
    if lang.script == "japanese":
        return script in ("han", "hiragana", "katakana")
    if lang.script == "chinese":
        return script == "han"
    if lang.script == "korean":
        return script in ("hangul", "han")
    # French
    lower = surface.lower()
    return any(c in _FRENCH_ONLY for c in lower) or strip_accents(lower) in KNOWN_FRENCH_WORDS


def _analyze_generic(text: str, code: str) -> list[Token]:
    tokens: list[Token] = []
    i = 0
    n = len(text)
    while i < n:
        cls = _script_class(text[i])
        j = i + 1
        if cls == "punct":
            pass  # one punctuation mark per token
        else:
            while j < n and _script_class(text[j]) == cls:
                j += 1
        if cls != "space":
            surface = text[i:j]
            tokens.append(
                Token(
                    surface=surface,
                    lemma=surface.lower(),
                    pos="PUNCT" if cls == "punct" else "",
                    is_target=_is_target_word(surface, cls, code),
                    start=i,
                    end=j,
                )
            )
        i = j
    return tokens


def analyze(text: str, language: str = "es") -> list[Token]:
    if get_language(language).code == "es":
        return _analyze_spanish(text)
    return _analyze_generic(text, language)


def looks_like_target(text: str, language: str) -> bool:
    """Most of `text`'s words are in the language being learned (rather
    than English)."""
    words = [t for t in analyze(text, language) if any(c.isalpha() for c in t.surface)]
    if get_language(language).script == "latin":
        words = [t for t in words if len(t.surface) > 1]
    if not words:
        return False
    return sum(1 for t in words if t.is_target) > len(words) / 2
