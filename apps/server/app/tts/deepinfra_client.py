"""Text-to-speech via DeepInfra's OpenAI-compatible /v1/audio/speech endpoint.

A separate provider from the chat model (app/chat/openai_client.py talks to
OpenAI's API instead): DeepInfra is a hosted API, keyed by its own bearer
token, so it gets its own client and its own failure mode.
"""

import logging

import httpx

from app.config import DEEPINFRA_API_TOKEN, DEEPINFRA_TTS_MODEL, DEEPINFRA_TTS_VOICE, TARGET_LANGUAGE
from app.tts.kokoro_voices import voice_for_language

DEEPINFRA_TTS_URL = "https://api.deepinfra.com/v1/audio/speech"

logger = logging.getLogger(__name__)


class TTSUnavailableError(RuntimeError):
    pass


async def synthesize(text: str, language: str = TARGET_LANGUAGE, voice: str | None = None) -> bytes:
    if not DEEPINFRA_API_TOKEN:
        raise TTSUnavailableError("DEEPINFRA_API_TOKEN is not configured on the server")

    # DEEPINFRA_TTS_VOICE is an admin-level escape hatch that overrides
    # everything; otherwise an explicit request voice wins over the
    # language's default.
    voice = DEEPINFRA_TTS_VOICE or voice or voice_for_language(language)

    payload = {
        "model": DEEPINFRA_TTS_MODEL,
        "input": text,
        "voice": voice,
        "response_format": "mp3",
    }
    headers = {"Authorization": f"Bearer {DEEPINFRA_API_TOKEN}"}

    # Normal calls come back in well under 1s (measured live); 15s per
    # attempt is already generous. Observed live, repeatedly: DeepInfra
    # itself occasionally stalls past that ceiling during a rough patch -
    # retry on a fresh connection, since a transport-level stall often
    # clears on a new attempt. 3 attempts, not 2 - a single retry wasn't
    # enough to reliably ride out DeepInfra's rougher patches in practice
    # ("doesn't always get generated"). Not retried on HTTPStatusError,
    # which means DeepInfra actually answered (auth/quota/bad request) - a
    # repeat attempt wouldn't change that.
    last_error: httpx.HTTPError | None = None
    for attempt in range(3):
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.post(DEEPINFRA_TTS_URL, json=payload, headers=headers)
            response.raise_for_status()
            return response.content
        except httpx.HTTPStatusError as exc:
            raise TTSUnavailableError(
                f"DeepInfra TTS request failed ({exc.response.status_code}): {exc.response.text}"
            ) from exc
        except httpx.HTTPError as exc:
            last_error = exc
            logger.warning(
                "TTS attempt %d/3 failed reaching DeepInfra (%s: %s)", attempt + 1, type(exc).__name__, exc
            )

    # repr, not str - httpx's own timeout/connection exceptions frequently
    # stringify to "" with no detail, which previously surfaced to the user
    # as an unhelpfully blank error message ("Could not reach ... API: ").
    raise TTSUnavailableError(
        f"Could not reach DeepInfra's TTS API after 3 attempts: {last_error!r}"
    ) from last_error
