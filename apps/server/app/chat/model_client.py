"""Routes a chat/translation completion to whichever backend is active -
the self-hosted llama-server (app/chat/llama_client.py) or a hosted,
pay-per-token API (app/chat/openai_client.py) - so every call site (chat
turns, translation, per-word glossing) goes through one function and
doesn't need its own if/else over providers.

`provider` is passed in explicitly by the caller (the live value from
app/settings_store.py, read from the DB) rather than this module reaching
into the DB itself - not every call site has a session handy (several of
translate/llm_translate.py's helpers are plain functions with no request
context), and a single value threaded down from wherever the session IS
available keeps this module itself a plain, session-free dispatcher.
Omitting it falls back to config.MODEL_PROVIDER's boot default, so
existing non-request callers (and tests) that don't care keep working.
"""

from app.chat import llama_client, openai_client
from app.chat.llama_client import ModelServerUnavailableError
from app.config import MODEL_PROVIDER

__all__ = ["ModelServerUnavailableError", "chat"]


def chat(
    messages: list[dict[str, str]],
    logit_bias: dict[int, float],
    timeout: float = 120.0,
    max_tokens: int | None = None,
    provider: str | None = None,
) -> str:
    active = provider or MODEL_PROVIDER
    if active == "openai":
        return openai_client.chat(messages, logit_bias, timeout=timeout, max_tokens=max_tokens)
    return llama_client.chat(messages, logit_bias, timeout=timeout, max_tokens=max_tokens)
