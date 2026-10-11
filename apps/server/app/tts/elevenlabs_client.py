"""Text-to-speech via ElevenLabs - the voice mode COACH's voice.

Separate from app/tts/deepinfra_client.py (the partner's Kokoro voice): the
coach's speech is English with Spanish phrases embedded ("say: 'Hola,
¿cómo estás?'"), and Kokoro's voices each apply one language's
pronunciation rules, so an English Kokoro voice would butcher the Spanish
and vice versa. ElevenLabs' multilingual models switch pronunciation inside
one request, so no per-language splicing is needed. Same shape as the
DeepInfra client otherwise: own key, retries only on transport errors,
raises the shared TTSUnavailableError.
"""

import logging

import httpx

from app.config import ELEVENLABS_API_KEY, ELEVENLABS_COACH_VOICE_ID, ELEVENLABS_TTS_MODEL
from app.tts.deepinfra_client import TTSUnavailableError

ELEVENLABS_TTS_URL = "https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"

logger = logging.getLogger(__name__)


async def synthesize(text: str, voice_id: str | None = None, model_id: str | None = None) -> bytes:
    if not ELEVENLABS_API_KEY:
        raise TTSUnavailableError("ELEVENLABS_API_KEY is not configured on the server")

    url = ELEVENLABS_TTS_URL.format(voice_id=voice_id or ELEVENLABS_COACH_VOICE_ID)
    payload = {
        "text": text,
        "model_id": model_id or ELEVENLABS_TTS_MODEL,
        # No language_code on purpose: the text is mixed English/Spanish and
        # the multilingual models detect the switch themselves - pinning a
        # language would force one accent onto both halves.
        "voice_settings": {"stability": 0.5, "similarity_boost": 0.75},
    }
    headers = {"xi-api-key": ELEVENLABS_API_KEY, "Accept": "audio/mpeg"}
    params = {"output_format": "mp3_44100_128"}

    last_error: httpx.HTTPError | None = None
    for attempt in range(3):
        try:
            async with httpx.AsyncClient(timeout=20.0) as client:
                response = await client.post(url, json=payload, headers=headers, params=params)
            response.raise_for_status()
            return response.content
        except httpx.HTTPStatusError as exc:
            raise TTSUnavailableError(
                f"ElevenLabs TTS request failed ({exc.response.status_code}): {exc.response.text}"
            ) from exc
        except httpx.HTTPError as exc:
            last_error = exc
            logger.warning(
                "Coach TTS attempt %d/3 failed reaching ElevenLabs (%s: %s)", attempt + 1, type(exc).__name__, exc
            )

    raise TTSUnavailableError(f"Could not reach ElevenLabs' TTS API after 3 attempts: {last_error!r}") from last_error
