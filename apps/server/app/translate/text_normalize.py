"""Shared accent-insensitive normalization for dictionary lookups - a
learner typing "como" should still hit "cómo"'s entry (or "cafe" "café"), and vice versa,
rather than silently missing because of one accent mark.

Deliberately an explicit table, not a blanket Unicode decompose-and-strip:
"ñ" decomposes to "n" + a combining tilde, but ñ is its own letter in
Spanish, not an accented "n" - stripping it would collide real word pairs
that differ only by ñ/n (e.g. "año" "year" vs "ano" - not a hypothetical,
an actual embarrassing bug this specifically avoids)."""

# Spanish and French accents (one character in, one out - callers rely on
# offsets staying aligned).
_ACCENT_MAP = str.maketrans(
    "áéíóúÁÉÍÓÚüÜàâäçèêëîïôöùûÿÀÂÄÇÈÊËÎÏÔÖÙÛŸ",
    "aeiouAEIOUuUaaaceeeiioouuyAAACEEEIIOOUUY",
)


def strip_accents(text: str) -> str:
    return text.translate(_ACCENT_MAP)
