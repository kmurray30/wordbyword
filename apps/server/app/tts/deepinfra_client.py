"""Text-to-speech via DeepInfra's OpenAI-compatible /v1/audio/speech endpoint.

A separate provider from the chat model (app/chat/llama_client.py talks to
our own self-hosted llama-server): DeepInfra is a hosted API, keyed by its
own bearer token, so it gets its own client and its own failure mode.
"""

import httpx

from app.config import DEEPINFRA_API_TOKEN, DEEPINFRA_TTS_MODEL, DEEPINFRA_TTS_VOICE

DEEPINFRA_TTS_URL = "https://api.deepinfra.com/v1/audio/speech"


class TTSUnavailableError(RuntimeError):
    pass


def synthesize(text: str) -> bytes:
    if not DEEPINFRA_API_TOKEN:
        raise TTSUnavailableError("DEEPINFRA_API_TOKEN is not configured on the server")

    payload = {
        "model": DEEPINFRA_TTS_MODEL,
        "input": text,
        "voice": DEEPINFRA_TTS_VOICE,
        "response_format": "mp3",
    }

    try:
        response = httpx.post(
            DEEPINFRA_TTS_URL,
            json=payload,
            headers={"Authorization": f"Bearer {DEEPINFRA_API_TOKEN}"},
            timeout=30.0,
        )
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise TTSUnavailableError(
            f"DeepInfra TTS request failed ({exc.response.status_code}): {exc.response.text}"
        ) from exc
    except httpx.HTTPError as exc:
        raise TTSUnavailableError(f"Could not reach DeepInfra's TTS API: {exc}") from exc

    return response.content
