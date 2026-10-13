from app.tts.kokoro_voices import (
    DEFAULT_VOICE_BY_LANGUAGE,
    KOKORO_VOICES,
    voice_for_language,
    voices_for_language,
)


def test_catalog_has_all_54_kokoro_voices():
    total = sum(len(voices) for voices in KOKORO_VOICES.values())
    assert total == 54


def test_catalog_voice_ids_are_unique():
    all_voices = [v for voices in KOKORO_VOICES.values() for v in voices]
    assert len(all_voices) == len(set(all_voices))


def test_every_voice_id_prefix_matches_its_language_code():
    for code, voices in KOKORO_VOICES.items():
        for voice in voices:
            assert voice.startswith(code), f"{voice!r} should start with language code {code!r}"


def test_every_default_voice_is_in_the_catalog_for_its_language():
    for language, voice in DEFAULT_VOICE_BY_LANGUAGE.items():
        assert voice in voices_for_language(language), f"{voice!r} missing from {language!r}'s voices"


def test_voice_for_language_spanish():
    assert voice_for_language("es") == "ef_dora"


def test_voice_for_language_is_case_insensitive():
    assert voice_for_language("ES") == voice_for_language("es")


def test_voice_for_language_unknown_falls_back_to_english():
    assert voice_for_language("xx") == DEFAULT_VOICE_BY_LANGUAGE["en"]


def test_voices_for_language_unknown_returns_empty():
    assert voices_for_language("xx") == []


def test_voices_for_language_spanish_has_three():
    assert voices_for_language("es") == ["ef_dora", "em_alex", "em_santa"]
