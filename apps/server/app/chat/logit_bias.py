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

# Lowered from 1.5/5.0/2.5, then again from 1.0/3.0/1.5: an A/B test against
# WORD_WEIGHTING_ENABLED=false showed even the 1.0-3.0 range was strong
# enough to make the 1.7B model echo the question back before answering it,
# twice out of six sampled replies. Paired with REINFORCE_WORDS_PER_TURN=1 /
# NEW_WORDS_PER_TURN=0 in config.py - a single, gently-nudged word per turn
# instead of forcing several at once.
MIN_REINFORCE_BIAS = 0.5
MAX_REINFORCE_BIAS = 1.0
NEW_WORD_BIAS = 0.5


@lru_cache(maxsize=1)
def _tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(TOKENIZER_NAME)


def warm_up() -> None:
    """Forces the tokenizer to load once, synchronously, at application
    startup rather than lazily on the first /chat/turn request. Observed
    live: an intermittent 500 ("ImportError: cannot import name
    'AutoTokenizer' from 'transformers'") on an early request - transformers'
    lazy-module loading isn't guaranteed safe against the first load racing
    concurrent request handling. Loading it once before any traffic arrives
    sidesteps that regardless of the exact mechanism, and as a side effect
    means the first real chat turn after a cold start doesn't also pay the
    tokenizer's own load latency inline.

    Deliberately swallows any failure (e.g. no network reachable to
    Hugging Face - true of local dev in a network-locked sandbox, and a
    real possibility in production too) rather than raising: this is a
    best-effort optimization, not a hard requirement - the whole app
    failing to start because of it would be a far worse outcome than
    falling back to the original lazy load on first real use."""
    try:
        _tokenizer()
    except Exception as exc:  # noqa: BLE001 - deliberately broad, see docstring
        print(f"WARNING: tokenizer warm-up failed ({exc}); will retry lazily on first /chat/turn")


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
