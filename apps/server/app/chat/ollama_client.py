import httpx

from app.config import OLLAMA_BASE_URL, OLLAMA_MODEL


class OllamaUnavailableError(RuntimeError):
    pass


def chat(messages: list[dict[str, str]], timeout: float = 60.0) -> str:
    """Calls Ollama's chat-completion API (POST /api/chat) with a single,
    non-streamed request and returns the assistant's reply text.

    The server only depends on this HTTP contract, not any specific model -
    swap OLLAMA_MODEL for any chat-capable model Ollama has pulled."""

    try:
        response = httpx.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json={"model": OLLAMA_MODEL, "messages": messages, "stream": False},
            timeout=timeout,
        )
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise OllamaUnavailableError(
            f"Could not reach Ollama at {OLLAMA_BASE_URL} with model '{OLLAMA_MODEL}'. "
            "Make sure `ollama serve` is running and the model has been pulled "
            f"(`ollama pull {OLLAMA_MODEL}`)."
        ) from exc

    data = response.json()
    return data.get("message", {}).get("content", "").strip()
