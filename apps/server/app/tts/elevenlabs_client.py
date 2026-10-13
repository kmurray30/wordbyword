"""Text-to-speech via ElevenLabs - every voice in the app: the coach's and
the partner's (and every 🔊).

Its multilingual models switch pronunciation inside one request, so the
coach's English with embedded Spanish ("say: 'Hola, ¿cómo estás?'") needs
no per-language splicing, and any voice can speak the partner's Spanish.
The coach and the partner get different voices (see coach_voice_id /
partner_voice_id). Retries only on transport errors and 429s; anything else
raises TTSUnavailableError.
"""

import asyncio
import logging

import httpx

import time

from app.config import (
    ELEVENLABS_API_KEY,
    ELEVENLABS_COACH_VOICE_ID,
    ELEVENLABS_PARTNER_VOICE_ID,
    ELEVENLABS_TTS_MODEL,
)
from app.tts.errors import TTSUnavailableError

ELEVENLABS_TTS_URL = "https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
ELEVENLABS_VOICES_URL = "https://api.elevenlabs.io/v1/voices"

logger = logging.getLogger(__name__)

# The voice picked by coach_voice_id when no id is configured - resolved
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


# The account's voices (GET /v1/voices), fetched at most every few minutes.
_VOICES_TTL_S = 600
_voices_cache: tuple[float, list[dict]] | None = None

# How many voices the app offers in its picker.
MAX_PICKER_VOICES = 6

_SPANISH_HINTS = ("spanish", "latin", "mexic", "argentin", "castil", "colombia", "chile", "peru", "es")


def _sounds_spanish(v: dict) -> bool:
    labels = v.get("labels") or {}
    text = " ".join(str(labels.get(k, "")) for k in ("language", "accent")).lower()
    return any(text == h or text.startswith(h) or f" {h}" in text for h in _SPANISH_HINTS)


def picker_order(voices: list[dict]) -> list[dict]:
    """The account's voices in the order the app offers them: ones the
    account added itself (cloned, from the library) first, then
    Spanish-sounding ones, then the premade rest. Pure, for tests."""

    def rank(v: dict) -> tuple[int, int]:
        own = 0 if v.get("category") not in (None, "premade") else 1
        return (own, 0 if _sounds_spanish(v) else 1)

    return sorted((v for v in voices if v.get("voice_id")), key=rank)


def pick_partner_voice(voices: list[dict], coach_id: str | None) -> str | None:
    """The partner's default voice: the first one offered that isn't the
    coach's, so the two always sound different. Pure, for tests."""
    for v in picker_order(voices):
        if v["voice_id"] != coach_id:
            return v["voice_id"]
    return coach_id


async def account_voices(client: httpx.AsyncClient | None = None) -> list[dict]:
    """The account's voices: [{voice_id, name, category, labels}, ...]."""
    global _voices_cache
    if not ELEVENLABS_API_KEY:
        raise TTSUnavailableError("ELEVENLABS_API_KEY is not configured on the server")
    if _voices_cache and time.monotonic() - _voices_cache[0] < _VOICES_TTL_S:
        return _voices_cache[1]
    headers = {"xi-api-key": ELEVENLABS_API_KEY}
    try:
        if client is None:
            async with httpx.AsyncClient(timeout=20.0) as own:
                response = await own.get(ELEVENLABS_VOICES_URL, headers=headers)
        else:
            response = await client.get(ELEVENLABS_VOICES_URL, headers=headers)
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise TTSUnavailableError(
            f"ElevenLabs voices request failed ({exc.response.status_code}): {exc.response.text}"
        ) from exc
    except httpx.HTTPError as exc:
        raise TTSUnavailableError(f"Could not reach ElevenLabs to list voices: {exc!r}") from exc
    voices = (response.json() or {}).get("voices") or []
    _voices_cache = (time.monotonic(), voices)
    return voices


async def coach_voice_id(client: httpx.AsyncClient | None = None) -> str:
    """ELEVENLABS_COACH_VOICE_ID, else one of the account's own voices
    (see pick_voice), picked once per process."""
    global _discovered_voice_id
    if ELEVENLABS_COACH_VOICE_ID:
        return ELEVENLABS_COACH_VOICE_ID
    if _discovered_voice_id:
        return _discovered_voice_id
    voice_id = pick_voice(await account_voices(client))
    if not voice_id:
        raise TTSUnavailableError("ElevenLabs account has no voices to use for the coach - set ELEVENLABS_COACH_VOICE_ID")
    logger.info("Coach TTS: using ElevenLabs voice %s (auto-picked from the account's voices)", voice_id)
    _discovered_voice_id = voice_id
    return voice_id


async def partner_voice_id(client: httpx.AsyncClient | None = None) -> str:
    """ELEVENLABS_PARTNER_VOICE_ID, else a voice that isn't the coach's."""
    if ELEVENLABS_PARTNER_VOICE_ID:
        return ELEVENLABS_PARTNER_VOICE_ID
    coach = await coach_voice_id(client)
    return pick_partner_voice(await account_voices(client), coach) or coach


async def synthesize(text: str, voice_id: str | None = None, model_id: str | None = None) -> bytes:
    """Speaks `text` in `voice_id` (default: the coach's voice)."""
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
                    resolved = voice_id or await coach_voice_id(client)
                    response = await client.post(
                        ELEVENLABS_TTS_URL.format(voice_id=resolved), json=payload, headers=headers, params=params
                    )
            response.raise_for_status()
            return response.content
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 429 and attempt < 2:
                logger.warning("TTS attempt %d/3: ElevenLabs rate-limited (429), retrying", attempt + 1)
                last_error = exc
                await asyncio.sleep(_RETRY_AFTER_429_S)
                continue
            raise TTSUnavailableError(
                f"ElevenLabs TTS request failed ({exc.response.status_code}): {exc.response.text}"
            ) from exc
        except httpx.HTTPError as exc:
            last_error = exc
            logger.warning(
                "TTS attempt %d/3 failed reaching ElevenLabs (%s: %s)", attempt + 1, type(exc).__name__, exc
            )

    raise TTSUnavailableError(f"Could not reach ElevenLabs' TTS API after 3 attempts: {last_error!r}") from last_error
