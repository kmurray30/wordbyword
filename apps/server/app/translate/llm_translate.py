"""Whole-message translation via the same local llama-server chat model
used for conversation (app/chat/llama_client.py), not Argos Translate.

Argos's offline MT produced noticeably rough, sometimes just-plain-wrong
translations on short/informal Spanish (e.g. rendering a simple "¿Qué
hobbies te gustan?" as "What do you care about?"). The LLM is already
running for chat anyway, so reusing it for this one-shot translation task
is free infrastructure-wise, just slower per call - callers are expected to
fetch this in the background after already showing the untranslated text,
not block the UI on it.
"""

from app.chat.llama_client import ModelServerUnavailableError, chat as llama_chat

_LANGUAGE_NAMES = {"es": "Spanish", "en": "English"}


class TranslationUnavailableError(RuntimeError):
    pass


def translate_text(text: str, source_lang: str, target_lang: str) -> str:
    if not text.strip():
        return ""

    source_name = _LANGUAGE_NAMES.get(source_lang, source_lang)
    target_name = _LANGUAGE_NAMES.get(target_lang, target_lang)
    messages = [
        {
            "role": "system",
            "content": (
                f"Translate the user's message from {source_name} to {target_name}. "
                "Reply with ONLY the translation, nothing else - no quotes, no "
                "explanation, no alternate phrasings."
            ),
        },
        {"role": "user", "content": text},
    ]
    try:
        return llama_chat(messages, logit_bias={}).strip()
    except ModelServerUnavailableError as exc:
        raise TranslationUnavailableError(str(exc)) from exc
