"""Maps the surface text app.translate.llm_translate.tag_draft returns for
each span back to exact character offsets in the real draft it was run on.
The model is asked to copy each span's surface text exactly, but small
models can't reliably report character offsets themselves - this does a
sequential, case/accent-insensitive scan instead, with the same tolerant
philosophy as gloss_reply's word-map lookups: a span that can't be located
(hallucinated text, or copied with some other normalization-insensitive
difference) is dropped rather than crashing or misplacing it."""

from __future__ import annotations

from dataclasses import dataclass

from app.translate.text_normalize import strip_accents


@dataclass
class MatchedSpan:
    surface: str
    start: int
    end: int
    gloss: str
    note: str
    translation: str


def _normalize(text: str) -> str:
    # strip_accents and str.lower() are both one-character-in/one-character-
    # out for every character this app deals with (Spanish/English) - the
    # normalized string stays perfectly index-aligned with the original, so
    # matched offsets can be used directly with no separate offset-mapping
    # table.
    return strip_accents(text).lower()


def match_spans(text: str, spans: list[dict[str, str]]) -> list[MatchedSpan]:
    """Spans are matched IN ORDER, each one searched for starting at the end
    of the previous match - this both disambiguates repeated surface text
    (two occurrences of "cat" each get their own span) and guarantees the
    result is non-overlapping, which the frontend's segment-building relies
    on. A span whose surface text can't be found ahead of the cursor gets
    one fallback full-text search (the model occasionally returns spans
    slightly out of textual order), but is still dropped if that would
    overlap an already-matched span, or isn't found at all."""
    normalized = _normalize(text)
    cursor = 0
    matched: list[MatchedSpan] = []
    for raw in spans:
        surface = raw.get("surface", "")
        needle = _normalize(surface)
        if not needle.strip():
            continue

        idx = normalized.find(needle, cursor)
        if idx == -1:
            idx = normalized.find(needle)
            if idx == -1 or idx < cursor:
                continue  # can't locate it, or it collides with an
                # already-matched span - drop rather than guess.

        start, end = idx, idx + len(needle)
        matched.append(
            MatchedSpan(
                surface=text[start:end],
                start=start,
                end=end,
                gloss=raw.get("gloss", ""),
                note=raw.get("note", ""),
                translation=raw.get("translation", ""),
            )
        )
        cursor = end
    return matched
