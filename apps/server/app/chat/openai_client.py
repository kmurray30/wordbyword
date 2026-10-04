"""Hosted chat-completion backend - the only chat backend this app talks to.

Historically this was one of two providers (the other being a self-hosted
llama-server), selected per-request via a live DB setting - that local-model
path has since been removed entirely, so this module is now the single,
unconditional dispatch target for every chat/translation call.
"""

import json
from collections.abc import Iterator

import httpx

from app.config import MAX_REPLY_TOKENS, OPENAI_API_KEY, OPENAI_CHAT_MODEL

_OPENAI_CHAT_URL = "https://api.openai.com/v1/chat/completions"


class ModelServerUnavailableError(RuntimeError):
    pass


def chat(
    messages: list[dict[str, str]],
    timeout: float = 60.0,
    max_tokens: int | None = None,
    model: str | None = None,
) -> str:
    if not OPENAI_API_KEY:
        raise ModelServerUnavailableError("OPENAI_API_KEY is not configured on the server")

    payload = {
        "model": model or OPENAI_CHAT_MODEL,
        "messages": messages,
        # Newer OpenAI models (confirmed live against gpt-6-luna) reject the
        # classic "max_tokens" field outright ("Unsupported parameter...
        # Use 'max_completion_tokens' instead") - this is the field every
        # current model accepts, legacy ones included.
        "max_completion_tokens": max_tokens if max_tokens is not None else MAX_REPLY_TOKENS,
    }
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}

    try:
        response = httpx.post(_OPENAI_CHAT_URL, json=payload, headers=headers, timeout=timeout)
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise ModelServerUnavailableError(
            f"OpenAI API request failed ({exc.response.status_code}): {exc.response.text}"
        ) from exc
    except httpx.HTTPError as exc:
        raise ModelServerUnavailableError(f"Could not reach the OpenAI API: {exc!r}") from exc

    data = response.json()
    choices = data.get("choices") or []
    if not choices:
        return ""
    return choices[0].get("message", {}).get("content", "").strip()


def chat_stream(
    messages: list[dict[str, str]],
    timeout: float = 60.0,
    max_tokens: int | None = None,
    model: str | None = None,
) -> Iterator[str]:
    """Same request as chat() above, but with `stream: true` - yields each
    content delta as it arrives over the SSE response. No mid-stream retry:
    a failure after streaming has already started can't be cleanly retried
    mid-generation, so this only guards the initial connection attempt."""
    if not OPENAI_API_KEY:
        raise ModelServerUnavailableError("OPENAI_API_KEY is not configured on the server")

    payload = {
        "model": model or OPENAI_CHAT_MODEL,
        "messages": messages,
        "stream": True,
        "max_completion_tokens": max_tokens if max_tokens is not None else MAX_REPLY_TOKENS,
    }
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}

    try:
        with httpx.stream("POST", _OPENAI_CHAT_URL, json=payload, headers=headers, timeout=timeout) as response:
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
    except httpx.HTTPStatusError as exc:
        raise ModelServerUnavailableError(
            f"OpenAI API request failed ({exc.response.status_code}): {exc.response.text}"
        ) from exc
    except httpx.HTTPError as exc:
        raise ModelServerUnavailableError(f"Could not reach the OpenAI API: {exc!r}") from exc
