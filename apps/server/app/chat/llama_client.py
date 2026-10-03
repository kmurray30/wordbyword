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

import json
import time
from collections.abc import Iterator

import httpx

from app.config import MAX_REPLY_TOKENS, MODEL_SERVER_BASE_URL


class ModelServerUnavailableError(RuntimeError):
    pass


# model-server runs with Railway's Serverless mode (sleeps after ~5-10min
# idle, wakes on the next request) - a 502/503 or connection failure right
# after a quiet period is that documented wake-up window, not a real outage,
# so it's worth a couple of short retries before giving up.
_MAX_ATTEMPTS = 3
_RETRY_DELAY_S = 3.0


def chat(
    messages: list[dict[str, str]],
    logit_bias: dict[int, float],
    timeout: float = 120.0,
    max_tokens: int | None = None,
) -> str:
    """Calls llama-server's OpenAI-compatible /v1/chat/completions endpoint
    with a single, non-streamed request, applying `logit_bias` (token id ->
    bias value) so specific word-bank words are genuinely more likely to be
    sampled, not just suggested in the prompt text. `max_tokens` defaults to
    MAX_REPLY_TOKENS (see its comment in config.py); a caller whose task is
    structurally larger than a short conversational reply - e.g. JSON
    covering a whole sentence's worth of per-word glosses - can ask for
    more room so it isn't truncated mid-object."""

    payload = {
        "messages": messages,
        "stream": False,
        "logit_bias": {str(token_id): bias for token_id, bias in logit_bias.items()},
        # Qwen3's chat template enables its <think>...</think> reasoning mode
        # by default. Observed live: some replies came back completely empty
        # - the model spent its whole turn "thinking" and never emitted any
        # text after the closing think tag. Turning it off gets a direct
        # answer in the content field every time, and this app has no use
        # for a visible reasoning trace anyway.
        "chat_template_kwargs": {"enable_thinking": False},
        # See MAX_REPLY_TOKENS' comment in config.py - bounds a generation
        # that never hits a stop token instead of letting it run until the
        # model server's context window is exhausted.
        "max_tokens": max_tokens if max_tokens is not None else MAX_REPLY_TOKENS,
    }

    response = None
    for attempt in range(_MAX_ATTEMPTS):
        if attempt > 0:
            time.sleep(_RETRY_DELAY_S)
        try:
            response = httpx.post(
                f"{MODEL_SERVER_BASE_URL}/v1/chat/completions",
                json=payload,
                timeout=timeout,
            )
            response.raise_for_status()
            break
        except httpx.TimeoutException as exc:
            raise ModelServerUnavailableError(
                f"Model server at {MODEL_SERVER_BASE_URL} didn't respond within {timeout}s. "
                "It may be under-resourced (check LLAMA_ARG_THREADS isn't oversubscribed "
                "relative to the container's actual CPU allocation) rather than down."
            ) from exc
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code not in (502, 503) or attempt == _MAX_ATTEMPTS - 1:
                raise ModelServerUnavailableError(
                    f"Could not reach the model server at {MODEL_SERVER_BASE_URL}. "
                    "Make sure llama-server is running (see README)."
                ) from exc
        except httpx.HTTPError as exc:
            if attempt == _MAX_ATTEMPTS - 1:
                raise ModelServerUnavailableError(
                    f"Could not reach the model server at {MODEL_SERVER_BASE_URL}. "
                    "Make sure llama-server is running (see README)."
                ) from exc

    data = response.json()
    choices = data.get("choices") or []
    if not choices:
        return ""
    return choices[0].get("message", {}).get("content", "").strip()


def chat_stream(
    messages: list[dict[str, str]],
    logit_bias: dict[int, float],
    timeout: float = 120.0,
    max_tokens: int | None = None,
) -> Iterator[str]:
    """Same request as chat() above, but with `stream: true` - yields each
    content delta as it arrives over llama-server's SSE response instead of
    waiting for the whole completion. No retry-on-transient-failure here
    (unlike chat()): a failure after streaming has already started can't be
    cleanly retried mid-generation, so this only guards the initial
    connection attempt."""
    payload = {
        "messages": messages,
        "stream": True,
        "logit_bias": {str(token_id): bias for token_id, bias in logit_bias.items()},
        "chat_template_kwargs": {"enable_thinking": False},
        "max_tokens": max_tokens if max_tokens is not None else MAX_REPLY_TOKENS,
    }
    try:
        with httpx.stream(
            "POST", f"{MODEL_SERVER_BASE_URL}/v1/chat/completions", json=payload, timeout=timeout
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line or not line.startswith("data:"):
                    continue
                data = line[len("data:") :].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                content = choices[0].get("delta", {}).get("content")
                if content:
                    yield content
    except httpx.HTTPError as exc:
        raise ModelServerUnavailableError(
            f"Could not reach the model server at {MODEL_SERVER_BASE_URL}. "
            "Make sure llama-server is running (see README)."
        ) from exc
