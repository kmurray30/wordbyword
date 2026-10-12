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

import asyncio
import logging

import httpx

from app.config import ELEVENLABS_API_KEY, ELEVENLABS_COACH_VOICE_ID, ELEVENLABS_TTS_MODEL
from app.tts.deepinfra_client import TTSUnavailableError

ELEVENLABS_TTS_URL = "https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
ELEVENLABS_VOICES_URL = "https://api.elevenlabs.io/v1/voices"

logger = logging.getLogger(__name__)

# The voice picked by _resolve_voice_id when no id is configured - resolved
# once per process, not per request.
_discovered_voice_id: str | None = None

# The voice coach fetches TTS one sentence at a time, several in flight at
# once (so playback can start before the model has finished writing) -
# observed live: the third parallel request came back as a failure, which
# matches ElevenLabs' per-plan concurrency cap (2 on the free tier). Cap
# in-flight requests here, and treat a 429 as retryable rather than final.
_MAX_CONCURRENT = 2
_concurrency = asyncio.Semaphore(_MAX_CONCURRENT)
_RETRY_AFTER_429_S = 0.75


def pick_voice(voices: list[dict]) -> str | None:
    """From GET /v1/voices' list, the id to use when none is configured:
    the account's own premade voices are the ones the API allows on every
    plan (observed live: a library voice id is a 402 on the free plan), so
    prefer those, and among them an English one, since the coach speaks
    mostly English. Pure, so it's testable without the API."""
    if not voices:
        return None

    def rank(v: dict) -> tuple[int, int]:
        labels = v.get("labels") or {}
        premade = 0 if v.get("category") == "premade" else 1
        english = 0 if str(labels.get("language", "")).lower().startswith("en") else 1
        return (premade, english)

    best = min(voices, key=rank)
    return best.get("voice_id") or None


async def _resolve_voice_id(client: httpx.AsyncClient, headers: dict[str, str]) -> str:
    global _discovered_voice_id
    if ELEVENLABS_COACH_VOICE_ID:
        return ELEVENLABS_COACH_VOICE_ID
    if _discovered_voice_id:
        return _discovered_voice_id
    response = await client.get(ELEVENLABS_VOICES_URL, headers=headers)
    response.raise_for_status()
    voice_id = pick_voice((response.json() or {}).get("voices") or [])
    if not voice_id:
        raise TTSUnavailableError("ElevenLabs account has no voices to use for the coach - set ELEVENLABS_COACH_VOICE_ID")
    logger.info("Coach TTS: using ElevenLabs voice %s (auto-picked from the account's voices)", voice_id)
    _discovered_voice_id = voice_id
    return voice_id


async def synthesize(text: str, voice_id: str | None = None, model_id: str | None = None) -> bytes:
    if not ELEVENLABS_API_KEY:
        raise TTSUnavailableError("ELEVENLABS_API_KEY is not configured on the server")

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
            async with _concurrency:
                async with httpx.AsyncClient(timeout=20.0) as client:
                    resolved = voice_id or await _resolve_voice_id(client, headers)
                    response = await client.post(
                        ELEVENLABS_TTS_URL.format(voice_id=resolved), json=payload, headers=headers, params=params
                    )
            response.raise_for_status()
            return response.content
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 429 and attempt < 2:
                logger.warning("Coach TTS attempt %d/3: ElevenLabs rate-limited (429), retrying", attempt + 1)
                last_error = exc
                await asyncio.sleep(_RETRY_AFTER_429_S)
                continue
            raise TTSUnavailableError(
                f"ElevenLabs TTS request failed ({exc.response.status_code}): {exc.response.text}"
            ) from exc
        except httpx.HTTPError as exc:
            last_error = exc
            logger.warning(
                "Coach TTS attempt %d/3 failed reaching ElevenLabs (%s: %s)", attempt + 1, type(exc).__name__, exc
            )

    raise TTSUnavailableError(f"Could not reach ElevenLabs' TTS API after 3 attempts: {last_error!r}") from last_error
