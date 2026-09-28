"""Offline EN<->ES word dictionary built from Wiktionary (via kaikki.org's
pre-parsed exports), pre-processed at Docker build time by
scripts/build_dictionary_data.py into a single filtered JSON file baked into
the image - real definitions, short usage notes, and colloquial/slang
register tags, entirely offline (no network, no LLM at request time). This
is what makes "bro"/"sup"/"partner"-style words actually resolve to a real
translation instead of Argos MT's silent word-for-word echo, and what gives
the input box's candidate cycler real alternatives to cycle through instead
of just one Argos guess.

Falls back to an empty dictionary (not an error) if the data file isn't
present - e.g. in local dev before running the build script, or if the
build-time download failed - so callers should always chain to the Argos MT
fallback in app.translate.service rather than assume a hit.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from app.translate.text_normalize import strip_accents

DATA_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "wiktionary_dictionary.json"


@lru_cache(maxsize=1)
def _data() -> dict[str, dict[str, list[dict[str, str]]]]:
    if not DATA_PATH.exists():
        return {"es_to_en": {}, "en_to_es": {}}
    with DATA_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def lookup_es_to_en(lemma: str) -> list[dict[str, str]]:
    # Keys are stored accent-stripped (see build_dictionary_data.py) so a
    # learner typing "como" still finds "cómo"'s entry.
    return _data()["es_to_en"].get(strip_accents(lemma.lower()), [])


def lookup_en_to_es(lemma: str) -> list[dict[str, str]]:
    return _data()["en_to_es"].get(strip_accents(lemma.lower()), [])
