"""Kokoro-82M's voice catalog (hexgrad/Kokoro-82M, 54 voices across 9
languages) and the language -> voice mapping used to pick one.

Kokoro doesn't infer language from the input text - the voice ID itself
selects which phonemizer rules apply (its language prefix), so feeding
Spanish text through an English voice mispronounces it. Every voice this
model ships with is listed here, keyed by Kokoro's own single-letter
language code, so a wrong/future language is a KeyError instead of a
silently mismatched voice.
"""

# Full catalog, keyed by Kokoro's own single-letter language code (the
# prefix on each voice id). Order within each list doesn't matter; kept as
# published in hexgrad/Kokoro-82M's VOICES.md.
KOKORO_VOICES: dict[str, list[str]] = {
    "a": [  # American English (11F 9M)
        "af_alloy", "af_aoede", "af_bella", "af_heart", "af_jessica", "af_kore",
        "af_nicole", "af_nova", "af_river", "af_sarah", "af_sky",
        "am_adam", "am_echo", "am_eric", "am_fenrir", "am_liam", "am_michael",
        "am_onyx", "am_puck", "am_santa",
    ],
    "b": [  # British English (4F 4M)
        "bf_alice", "bf_emma", "bf_isabella", "bf_lily",
        "bm_daniel", "bm_fable", "bm_george", "bm_lewis",
    ],
    "e": ["ef_dora", "em_alex", "em_santa"],  # Spanish (1F 2M)
    "f": ["ff_siwis"],  # French (1F)
    "h": ["hf_alpha", "hf_beta", "hm_omega", "hm_psi"],  # Hindi (2F 2M)
    "i": ["if_sara", "im_nicola"],  # Italian (1F 1M)
    "j": ["jf_alpha", "jf_gongitsune", "jf_nezumi", "jf_tebukuro", "jm_kumo"],  # Japanese (4F 1M)
    "p": ["pf_dora", "pm_alex", "pm_santa"],  # Brazilian Portuguese (1F 2M)
    "z": [  # Mandarin Chinese (4F 4M)
        "zf_xiaobei", "zf_xiaoni", "zf_xiaoxiao", "zf_xiaoyi",
        "zm_yunjian", "zm_yunxi", "zm_yunxia", "zm_yunyang",
    ],
}

# The app's own language codes (matching config.TARGET_LANGUAGE /
# NATIVE_LANGUAGE style) -> Kokoro's language code, plus locale variants
# that disambiguate within a language (British vs. American English,
# Brazilian Portuguese).
_LANGUAGE_TO_KOKORO_CODE: dict[str, str] = {
    "en": "a",
    "en-us": "a",
    "en-gb": "b",
    "es": "e",
    "fr": "f",
    "hi": "h",
    "it": "i",
    "ja": "j",
    "pt": "p",
    "pt-br": "p",
    "zh": "z",
}

# One sensible default voice per language, for callers that just want "a
# Spanish voice" rather than picking a specific one by name.
DEFAULT_VOICE_BY_LANGUAGE: dict[str, str] = {
    "en": "af_heart",
    "en-us": "af_heart",
    "en-gb": "bf_emma",
    "es": "ef_dora",
    "fr": "ff_siwis",
    "hi": "hf_alpha",
    "it": "if_sara",
    "ja": "jf_alpha",
    "pt": "pf_dora",
    "pt-br": "pf_dora",
    "zh": "zf_xiaobei",
}


def voices_for_language(language: str) -> list[str]:
    """All voice ids available for `language`, or [] if unsupported."""
    code = _LANGUAGE_TO_KOKORO_CODE.get(language.lower())
    return list(KOKORO_VOICES.get(code, [])) if code else []


def voice_for_language(language: str) -> str:
    """The default voice id for `language`, falling back to American
    English if the language isn't one Kokoro supports."""
    return DEFAULT_VOICE_BY_LANGUAGE.get(language.lower(), DEFAULT_VOICE_BY_LANGUAGE["en"])
