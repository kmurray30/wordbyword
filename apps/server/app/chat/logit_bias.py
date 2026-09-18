"""Maps word-bank lemmas to token-level logit_bias entries for llama-server's
/completion endpoint - the actual per-word weighting mechanism. See
llama_client.py's docstring for why this exists instead of relying on Ollama
(which has no logit_bias support at all).

Bias is computed only from each lemma's dictionary/base surface form's FIRST
token - deliberately not attempting to also cover inflected forms (hablo,
hablando, habló, ...). Each inflected form is treated as its own vocabulary
item to learn, not a variant of the lemma - a deliberate design choice, not a
shortcut (see the plan notes for the reasoning).

Bias is positive-only: we nudge the model toward words that are due for
review or newly introduced. We don't attempt to suppress the rest of the
vocabulary (that would mean enumerating and biasing every token NOT in the
target list, which isn't practical), so this is additive to - not a
replacement for - the system-prompt instruction, which still carries the
semantic/grammatical framing.
"""

from __future__ import annotations

from functools import lru_cache

from app.config import TOKENIZER_NAME

MIN_REINFORCE_BIAS = 1.5
MAX_REINFORCE_BIAS = 5.0
NEW_WORD_BIAS = 2.5


@lru_cache(maxsize=1)
def _tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(TOKENIZER_NAME)


@lru_cache(maxsize=4096)
def _first_token_id(surface_form: str) -> int | None:
    """Token id of the first token produced for this surface form. A leading
    space is included because BPE tokenizers (Qwen3 included) encode a word
    differently depending on whether it follows whitespace - the word-initial
    variant is what actually matters when biasing mid-sentence generation."""
    ids = _tokenizer().encode(" " + surface_form, add_special_tokens=False)
    return ids[0] if ids else None


def build_logit_bias(
    reinforce_lemmas: list[str],
    new_lemmas: list[str],
    reinforce_urgency: dict[str, float],
) -> dict[int, float]:
    bias: dict[int, float] = {}

    for lemma in reinforce_lemmas:
        token_id = _first_token_id(lemma)
        if token_id is None:
            continue
        urgency = reinforce_urgency.get(lemma, 1.0)
        value = MIN_REINFORCE_BIAS + urgency * (MAX_REINFORCE_BIAS - MIN_REINFORCE_BIAS)
        bias[token_id] = max(bias.get(token_id, 0.0), value)

    for lemma in new_lemmas:
        token_id = _first_token_id(lemma)
        if token_id is None:
            continue
        bias[token_id] = max(bias.get(token_id, 0.0), NEW_WORD_BIAS)

    return bias
