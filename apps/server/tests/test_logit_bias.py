from unittest.mock import patch

from app.chat.logit_bias import (
    MAX_REINFORCE_BIAS,
    MIN_REINFORCE_BIAS,
    NEW_WORD_BIAS,
    _first_token_id,
    build_logit_bias,
    warm_up,
)


class FakeTokenizer:
    """Deterministic stand-in for the real Qwen3 tokenizer, so tests don't
    need network access to Hugging Face."""

    VOCAB = {" hablar": [101], " comer": [202], " vivir": [101]}  # note: shares id with "hablar"

    def encode(self, text, add_special_tokens=False):
        return self.VOCAB.get(text, [999])


def setup_function():
    _first_token_id.cache_clear()


def test_build_logit_bias_scales_by_urgency():
    with patch("app.chat.logit_bias._tokenizer", return_value=FakeTokenizer()):
        bias = build_logit_bias(
            reinforce_lemmas=["hablar"],
            new_lemmas=[],
            reinforce_urgency={"hablar": 1.0},
        )
    assert bias == {101: MAX_REINFORCE_BIAS}


def test_build_logit_bias_min_urgency_gives_min_bias():
    with patch("app.chat.logit_bias._tokenizer", return_value=FakeTokenizer()):
        bias = build_logit_bias(
            reinforce_lemmas=["hablar"],
            new_lemmas=[],
            reinforce_urgency={"hablar": 0.0},
        )
    assert bias == {101: MIN_REINFORCE_BIAS}


def test_build_logit_bias_new_words_get_flat_bias():
    with patch("app.chat.logit_bias._tokenizer", return_value=FakeTokenizer()):
        bias = build_logit_bias(reinforce_lemmas=[], new_lemmas=["comer"], reinforce_urgency={})
    assert bias == {202: NEW_WORD_BIAS}


def test_build_logit_bias_dedupes_shared_first_token_by_taking_max():
    with patch("app.chat.logit_bias._tokenizer", return_value=FakeTokenizer()):
        bias = build_logit_bias(
            reinforce_lemmas=["hablar"],
            new_lemmas=["vivir"],  # shares token id 101 with "hablar"
            reinforce_urgency={"hablar": 0.0},  # -> MIN_REINFORCE_BIAS, lower than NEW_WORD_BIAS
        )
    assert bias == {101: max(MIN_REINFORCE_BIAS, NEW_WORD_BIAS)}


def test_build_logit_bias_empty_input():
    with patch("app.chat.logit_bias._tokenizer", return_value=FakeTokenizer()):
        assert build_logit_bias([], [], {}) == {}


def test_warm_up_loads_the_tokenizer():
    # Loading eagerly at startup, rather than lazily on the first request,
    # is what sidesteps the intermittent import race observed live - just
    # confirms warm_up() actually forces the (cached) load to happen.
    with patch("app.chat.logit_bias._tokenizer") as mocked:
        warm_up()
    mocked.assert_called_once()
