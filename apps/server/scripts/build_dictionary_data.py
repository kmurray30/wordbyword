"""Build-time step: downloads Wiktionary's EN and ES entries (pre-parsed by
kaikki.org from the English Wiktionary dump - real definitions, usage
examples, and colloquial/slang register tags, not just bare word-lists) and
filters them down into a single compact JSON file bundled into the Docker
image at app/translate/wiktionary_dict.py's DATA_PATH. Run once per build,
same pattern as install_translate_models.py.

Why: Argos Translate's word-level lookup (a single word run through its
sentence MT model with no context) has no way to offer more than one
candidate, and silently echoes words it doesn't recognize (slang like "bro"
or "sup") back unchanged instead of failing loudly - see app/translate/
service.py's word_candidates() for how this file's output is consulted
first, with Argos only as a last-resort fallback.

Deliberately non-fatal: if the download or parse fails for any reason (a
stale URL, a network hiccup, a schema change upstream), this prints a
warning and writes an empty dataset rather than failing the whole Docker
build - degrading to Argos-only lookups is fine, a broken deploy isn't.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import httpx

SPANISH_URL = "https://kaikki.org/dictionary/Spanish/kaikki.org-dictionary-Spanish.jsonl"
ENGLISH_URL = "https://kaikki.org/dictionary/English/kaikki.org-dictionary-English.jsonl"

OUT_PATH = Path(__file__).resolve().parent.parent / "data" / "wiktionary_dictionary.json"

# A word outside the frequency cutoff below still gets included if any of its
# senses carries one of these register tags - this is specifically what
# rescues informal/slang vocabulary ("bro", "sup") that a formal-corpus
# frequency list would otherwise exclude. Deliberately excludes vulgar/
# offensive tags - this is a general-audience learning app, not a slang
# dictionary, so casual register is in scope but crude language isn't sought
# out (an entry only shows up if it's ALSO common enough or independently
# tagged one of the tags below).
REGISTER_TAGS = {"informal", "colloquial", "slang"}

# How many words deep into each language's frequency list still counts as
# "common enough to bundle unconditionally" - keeps the final dataset a
# manageable size while covering the vocabulary a learner will actually run
# into, on top of the register-tag net above for informal words that a
# formal-corpus frequency list underrepresents.
FREQUENCY_CUTOFF = 15000

# Per word, how many distinct sense/translation candidates to keep - matches
# the frontend's CandidateCycler, which is meant for "a few alternatives to
# cycle through", not an exhaustive dictionary entry.
MAX_CANDIDATES_PER_WORD = 4

MAX_DESCRIPTION_LEN = 100
MAX_TRANSLATION_LEN = 60

_WORD_RE = re.compile(r"^[a-zA-ZñÑáéíóúüÁÉÍÓÚÜ]+$")

# TEMPORARY: dumps the raw kaikki.org entry for these words to the build log
# so a live deploy's build log can show exactly what Wiktionary has for them
# (translation tables, tags) - remove once bro/sup/partner coverage is
# confirmed working end to end.
_DEBUG_WORDS = {"bro", "sup", "partner"}


def _short(text: str, limit: int) -> str:
    text = text.strip()
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _first_gloss(sense: dict) -> str:
    glosses = sense.get("glosses") or []
    return glosses[0].strip() if glosses else ""


def _first_example(sense: dict) -> str:
    for ex in sense.get("examples") or []:
        text = (ex.get("text") or "").strip()
        if text:
            return text
    return ""


def _format_description(tags: set[str], gloss: str = "", example: str = "") -> str:
    tag_list = sorted(t for t in tags if t in REGISTER_TAGS)
    prefix = f"({', '.join(tag_list)}) " if tag_list else ""
    body = example or gloss
    return _short(f"{prefix}{body}".strip(), MAX_DESCRIPTION_LEN) if (prefix or body) else ""


def _stream_lines(url: str):
    with httpx.stream("GET", url, timeout=180.0, follow_redirects=True) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if line.strip():
                yield line


def _process_spanish_line(line: str, es_freq: set[str], out: dict[str, list[dict[str, str]]]) -> None:
    entry = json.loads(line)
    if entry.get("lang_code") != "es":
        return
    word = entry.get("word", "")
    if word.lower() in _DEBUG_WORDS:
        print(f"DEBUG raw Spanish entry for {word!r}: {line[:3000]}", file=sys.stderr)
    if not _WORD_RE.match(word):
        return
    lemma = word.lower()
    candidates = out.setdefault(lemma, [])
    if len(candidates) >= MAX_CANDIDATES_PER_WORD:
        return
    is_common = lemma in es_freq
    seen = {c["translation"].lower() for c in candidates}
    for sense in entry.get("senses") or []:
        if len(candidates) >= MAX_CANDIDATES_PER_WORD:
            break
        tags = set(sense.get("tags") or [])
        if not is_common and not (tags & REGISTER_TAGS):
            continue
        gloss = _first_gloss(sense)
        if not gloss:
            continue
        translation = _short(gloss, MAX_TRANSLATION_LEN)
        if translation.lower() in seen:
            continue
        candidates.append(
            {"translation": translation, "description": _format_description(tags, example=_first_example(sense))}
        )
        seen.add(translation.lower())


def _match_sense(senses: list[dict], sense_text: str) -> dict | None:
    if sense_text:
        for s in senses:
            gloss = _first_gloss(s).lower()
            if gloss and (sense_text in gloss or gloss in sense_text):
                return s
    return senses[0] if senses else None


def _process_english_line(line: str, en_freq: set[str], out: dict[str, list[dict[str, str]]]) -> None:
    entry = json.loads(line)
    if entry.get("lang_code") != "en":
        return
    word = entry.get("word", "")
    if word.lower() in _DEBUG_WORDS:
        print(f"DEBUG raw English entry for {word!r}: {line[:3000]}", file=sys.stderr)
    if not _WORD_RE.match(word):
        return
    es_translations = [t for t in (entry.get("translations") or []) if t.get("code") == "es" and t.get("word")]
    if not es_translations:
        return
    senses = entry.get("senses") or []
    lemma = word.lower()
    is_common = lemma in en_freq
    any_informal = any(set(s.get("tags") or []) & REGISTER_TAGS for s in senses)
    if not is_common and not any_informal:
        return

    candidates = out.setdefault(lemma, [])
    seen = {c["translation"].lower() for c in candidates}
    for t in es_translations:
        if len(candidates) >= MAX_CANDIDATES_PER_WORD:
            break
        es_word = (t.get("word") or "").strip()
        if not es_word or es_word.lower() in seen:
            continue
        if es_word.lower() == lemma and candidates:
            # Wiktionary sometimes lists a word as its own Spanish loanword
            # translation (e.g. "bro" -> "bro") - fine as the ONLY candidate
            # (true cognates like "hotel" are genuinely identical), but once
            # a real alternative already exists, a same-spelled entry is just
            # noise, not a second useful option to cycle to.
            continue
        matched = _match_sense(senses, (t.get("sense") or "").lower())
        tags = set((matched or {}).get("tags") or [])
        gloss = _first_gloss(matched) if matched else ""
        example = _first_example(matched) if matched else ""
        candidates.append({"translation": es_word, "description": _format_description(tags, gloss=gloss, example=example)})
        seen.add(es_word.lower())


def _augment_en_to_es_from_es_glosses(
    es_to_en: dict[str, list[dict[str, str]]], en_to_es: dict[str, list[dict[str, str]]], en_freq: set[str]
) -> None:
    """Wiktionary's per-word translation tables (what _process_english_line
    reads) are crowd-sourced and often incomplete for informal words -
    "bro" may have no "es" entry there at all even though a Spanish word is
    independently defined (on the ES side) with "bro" as its English gloss.
    Mine that reverse relationship from the ES dataset we already
    downloaded - no second pass over the wire, and it only adds candidates,
    never removes any from the translations-table pass above."""
    for es_word, candidates in es_to_en.items():
        for cand in candidates:
            gloss = cand["translation"].lower().strip()
            if not re.match(r"^[a-z]+$", gloss):
                continue  # only single bare words - "bank (financial institution)" stays out
            # _format_description() only ever prepends a "(...)" prefix from
            # a REGISTER_TAGS-filtered tag list, so this is exactly "did any
            # register tag apply to this sense" without re-parsing which one.
            is_informal = cand["description"].startswith("(")
            if gloss not in en_freq and not is_informal:
                continue
            out_list = en_to_es.setdefault(gloss, [])
            if len(out_list) >= MAX_CANDIDATES_PER_WORD:
                continue
            if any(c["translation"].lower() == es_word for c in out_list):
                continue
            out_list.append({"translation": es_word, "description": cand["description"] or gloss})


def _load_frequency_lists() -> tuple[set[str], set[str]]:
    try:
        from wordfreq import top_n_list

        return set(top_n_list("es", FREQUENCY_CUTOFF)), set(top_n_list("en", FREQUENCY_CUTOFF))
    except Exception as exc:  # pragma: no cover - defensive, see module docstring
        print(f"WARNING: could not load wordfreq lists ({exc}); frequency filtering disabled, "
              "dataset limited to register-tagged (informal/slang) words only")
        return set(), set()


def _build_direction(label: str, url: str, freq: set[str], process_line) -> dict[str, list[dict[str, str]]]:
    out: dict[str, list[dict[str, str]]] = {}
    try:
        print(f"Downloading and processing {url} ...")
        count = 0
        for line in _stream_lines(url):
            count += 1
            try:
                process_line(line, freq, out)
            except (json.JSONDecodeError, KeyError, TypeError):
                continue
        out = {k: v for k, v in out.items() if v}
        print(f"  {label}: processed {count} lines, kept {len(out)} headwords")
    except Exception as exc:
        print(f"WARNING: failed to build {label} dictionary from Wiktionary ({exc}); "
              "those word-level lookups will fall back to Argos MT only", file=sys.stderr)
    return out


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    es_freq, en_freq = _load_frequency_lists()

    es_to_en = _build_direction("es->en", SPANISH_URL, es_freq, _process_spanish_line)
    en_to_es = _build_direction("en->es", ENGLISH_URL, en_freq, _process_english_line)

    before = len(en_to_es)
    _augment_en_to_es_from_es_glosses(es_to_en, en_to_es, en_freq)
    print(f"  en->es: reverse-index augmentation added {len(en_to_es) - before} more headwords "
          f"(now {len(en_to_es)})")

    with OUT_PATH.open("w", encoding="utf-8") as f:
        json.dump({"es_to_en": es_to_en, "en_to_es": en_to_es}, f, ensure_ascii=False, separators=(",", ":"))
    size_mb = OUT_PATH.stat().st_size / 1_000_000
    print(f"Wrote {OUT_PATH} ({size_mb:.1f} MB): {len(es_to_en)} ES entries, {len(en_to_es)} EN entries")


if __name__ == "__main__":
    main()
