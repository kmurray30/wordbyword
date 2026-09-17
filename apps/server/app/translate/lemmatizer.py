"""Spanish lemmatization/POS-tagging via spaCy. The word bank tracks lemmas
(base dictionary forms), not raw inflected surface forms, so every agent
message and every user keystroke gets run through this before touching the
word bank.

Requires the `es_core_news_sm` spaCy model to be downloaded once:
    python -m spacy download es_core_news_sm
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from app.translate.frequency_es_set import KNOWN_SPANISH_LEMMAS


@dataclass
class Token:
    surface: str
    lemma: str
    pos: str
    is_spanish: bool
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


def analyze(text: str) -> list[Token]:
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
                is_spanish=_is_spanish(tok.text, lemma),
                start=tok.idx,
                end=tok.idx + len(tok.text),
            )
        )
    return tokens
