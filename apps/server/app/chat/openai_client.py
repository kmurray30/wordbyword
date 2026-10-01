"""Hosted chat-completion backend - an alternative to the self-hosted
llama-server (app/chat/llama_client.py) for when the MODEL_PROVIDER setting
(app/config.py, live-overridable via app/settings_store.py) is "openai".
Routed to by app/chat/model_client.py, which both chat.py's take_turn and
translate/llm_translate.py's helpers call through rather than either
provider module directly, so call sites don't need to know which backend
is active.

The tradeoff for not having to run/pay for a model server at all: no
word-bank vocabulary steering. `logit_bias` is accepted (for the same call
signature as llama_client.chat) but ignored - it's keyed to tokens from
the local model's own tokenizer (see logit_bias.py), which are meaningless
against this API's own, different tokenizer. Callers are responsible for
not computing it in the first place when this provider is active (see
app.settings_store.weighting_active) - this module ignoring it is a second
line of defense, not the primary one.
"""

import httpx

from app.chat.llama_client import ModelServerUnavailableError
from app.config import MAX_REPLY_TOKENS, OPENAI_API_KEY, OPENAI_CHAT_MODEL

_OPENAI_CHAT_URL = "https://api.openai.com/v1/chat/completions"


def chat(
    messages: list[dict[str, str]],
    logit_bias: dict[int, float] | None = None,
    timeout: float = 60.0,
    max_tokens: int | None = None,
) -> str:
    if not OPENAI_API_KEY:
        raise ModelServerUnavailableError(
            "OPENAI_API_KEY is not configured on the server, but the active model provider is \"openai\""
        )

    payload = {
        "model": OPENAI_CHAT_MODEL,
        "messages": messages,
        "max_tokens": max_tokens if max_tokens is not None else MAX_REPLY_TOKENS,
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
