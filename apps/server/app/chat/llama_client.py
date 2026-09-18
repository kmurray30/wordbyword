"""Talks to llama-server (llama.cpp's own server), not Ollama.

Why: the word bank needs to actually *weight* individual words during
generation, not just ask the model nicely via the system prompt. That
requires logit_bias - directly boosting specific tokens' sampling
probability. Ollama has no logit_bias support at all (ollama/ollama#3795 is
an open, unresolved feature request as of this writing) - it can only read
token probabilities (logprobs), not bias them. llama-server's
/v1/chat/completions endpoint, by contrast, has a stable, documented
logit_bias parameter and applies the model's own chat template to the
messages array automatically, so we don't have to hand-format prompts for
Qwen3 ourselves.
"""

import httpx

from app.config import MODEL_SERVER_BASE_URL


class ModelServerUnavailableError(RuntimeError):
    pass


def chat(messages: list[dict[str, str]], logit_bias: dict[int, float], timeout: float = 60.0) -> str:
    """Calls llama-server's OpenAI-compatible /v1/chat/completions endpoint
    with a single, non-streamed request, applying `logit_bias` (token id ->
    bias value) so specific word-bank words are genuinely more likely to be
    sampled, not just suggested in the prompt text."""

    payload = {
        "messages": messages,
        "stream": False,
        "logit_bias": {str(token_id): bias for token_id, bias in logit_bias.items()},
    }

    try:
        response = httpx.post(
            f"{MODEL_SERVER_BASE_URL}/v1/chat/completions",
            json=payload,
            timeout=timeout,
        )
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise ModelServerUnavailableError(
            f"Could not reach the model server at {MODEL_SERVER_BASE_URL}. "
            "Make sure llama-server is running (see README)."
        ) from exc

    data = response.json()
    choices = data.get("choices") or []
    if not choices:
        return ""
    return choices[0].get("message", {}).get("content", "").strip()
