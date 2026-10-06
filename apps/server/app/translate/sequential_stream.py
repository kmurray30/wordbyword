"""Generalizes the marker-buffer-scan-with-withheld-tail streaming technique
coach_draft_stream already used for its single `meaning_chunk` field across
an ORDERED LIST of N marker-delimited fields - needed now that
coach_draft_stream's protocol has grown from 2 sections (a meaning, then one
big JSON blob) to up to 8 sequential plain-text fields (a verdict, a
meaning, then up to 6 per-option fields).

Pulled into its own module (rather than inlined in llm_translate.py,
already 900+ lines) since it's a pure, codec-like function with no LLM-
specific knowledge at all - testable in complete isolation from any model
mocking, same reasoning app.translate.span_matching got its own file/test
module for.
"""

from collections.abc import Iterator
from dataclasses import dataclass


@dataclass
class Field:
    """One ordered field in a marker-delimited stream. `marker` is the
    literal string that TERMINATES this field - everything in the buffer
    up to (not including) the marker belongs to this field; the marker
    itself is consumed (not re-emitted) once found. `extra` is arbitrary
    positional metadata the caller wants echoed back in every tuple this
    field yields (e.g. an option's index) - stream_sequential_fields itself
    has no opinion on what it means, it just threads it through."""

    name: str
    marker: str
    extra: tuple = ()


def stream_sequential_fields(
    chunks: Iterator[str],
    fields: list[Field],
    initial_buffer: str = "",
) -> Iterator[tuple]:
    """Walks `fields` in order against text pulled from `chunks` (typically
    app.chat.openai_client.chat_stream's own delta iterator), yielding, for
    each field in turn:
      (f"{field.name}_chunk", *field.extra, delta) - as text safely arrives
      (f"{field.name}_complete", *field.extra, full_text) - once that
        field's own marker is found

    "Safely arrives" withholds a trailing tail of the buffer sized to the
    LONGEST marker among ALL of `fields` (not just the current field's
    own) - a chunk boundary landing mid-marker must never leak a fragment
    of ANY upcoming marker into a `_chunk` delta, and with several
    distinct marker strings now in play (unlike the single-marker case
    this generalizes), a tail sized only for the current field's own
    marker could still leak a differently-shaped LATER marker's prefix
    once this function advances to it.

    `initial_buffer` lets a caller resume mid-stream against the SAME
    underlying `chunks` iterator (iterators are stateful/one-shot, so
    passing the same one back in continues exactly where a PREVIOUS call
    left off) - see coach_draft_stream, which calls this twice: once for
    just a leading verdict field (whose value decides which fields come
    next), then again - seeded with the first call's leftover buffer via
    this parameter - for the meaning field plus whichever branch-specific
    fields the verdict implies.

    Stops once every field in `fields` has completed, OR once `chunks` is
    exhausted first (some field's marker never arrived) - the caller tells
    these apart by how many `_complete` events it actually received, same
    as the single-field version this replaces did via its own `stage`
    variable. Returns (via a generator `return`, i.e. visible to the
    caller as `StopIteration.value`) whatever of the buffer was left over
    unconsumed once this function stops - either genuine leftover content
    after the last field's marker (the `initial_buffer` case above) or
    just whatever of an incomplete final field streamed before `chunks`
    ran out."""
    if not fields:
        return initial_buffer

    tail = max(len(f.marker) for f in fields) - 1
    buffer = initial_buffer
    field_idx = 0
    sent_len = 0

    def drain() -> Iterator[tuple]:
        nonlocal buffer, field_idx, sent_len
        while field_idx < len(fields):
            current = fields[field_idx]
            idx = buffer.find(current.marker)
            safe_end = idx if idx != -1 else max(sent_len, len(buffer) - tail)
            if safe_end > sent_len:
                delta = buffer[sent_len:safe_end]
                sent_len = safe_end
                if delta:
                    yield (f"{current.name}_chunk", *current.extra, delta)
            if idx == -1:
                return
            full_text = buffer[:idx]
            yield (f"{current.name}_complete", *current.extra, full_text)
            buffer = buffer[idx + len(current.marker) :]
            sent_len = 0
            field_idx += 1

    yield from drain()
    if field_idx >= len(fields):
        return buffer

    for chunk in chunks:
        buffer += chunk
        yield from drain()
        if field_idx >= len(fields):
            return buffer

    return buffer
