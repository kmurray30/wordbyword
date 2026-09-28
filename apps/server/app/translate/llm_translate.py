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


def _parse_labeled_lines(reply: str, native_name: str, target_name: str) -> tuple[str, str]:
    native_text = ""
    target_text = ""
    for line in reply.splitlines():
        line = line.strip()
        if not line:
            continue
        lower = line.lower()
        if lower.startswith(f"{native_name.lower()}:"):
            native_text = line.split(":", 1)[1].strip()
        elif lower.startswith(f"{target_name.lower()}:"):
            target_text = line.split(":", 1)[1].strip()
    if not native_text and not target_text:
        # The model didn't follow the requested "Label: text" format - use
        # the whole reply as the native-language line and let the caller
        # derive the target line separately rather than returning nothing.
        native_text = reply.strip()
    return native_text, target_text


def interpret_user_input(text: str, native_lang: str, target_lang: str) -> tuple[str, str]:
    """Given raw learner input that may mix native_lang and target_lang, and
    may have grammar/spelling mistakes in either, return (corrected_native_
    text, target_language_translation) - the model infers intent across both
    languages rather than doing a literal per-word pass. Always returns a
    non-empty target translation for non-empty input: if the model doesn't
    follow the requested two-line format, falls back to a plain
    translate_text call for the target line."""
    if not text.strip():
        return "", ""

    native_name = _LANGUAGE_NAMES.get(native_lang, native_lang)
    target_name = _LANGUAGE_NAMES.get(target_lang, target_lang)
    messages = [
        {
            "role": "system",
            "content": (
                f"A language learner is typing in a mix of {native_name} and "
                f"{target_name}, possibly with grammar or spelling mistakes in "
                f"either language. Your ONLY job is to rewrite what they typed "
                f"as a correct sentence in each language. You are NOT having a "
                f"conversation with them - do not reply to them, answer them, "
                f"or add anything they did not say. Preserve their meaning and "
                f"length; only fix it, don't extend it. For example, if they "
                f'type "hello sir": a BAD output replies to them ("Hello, sir. '
                f'I am here to assist you.") - the GOOD output is just their '
                f'own greeting, corrected ("Hello, sir.").\n\n'
                f"Reply with EXACTLY two lines and nothing else:\n"
                f"{native_name}: <their message, corrected to natural, "
                f"grammatically correct {native_name} - nothing added>\n"
                f"{target_name}: <the same corrected message in natural, "
                f"grammatically correct {target_name} - nothing added>"
            ),
        },
        {"role": "user", "content": text},
    ]
    try:
        reply = llama_chat(messages, logit_bias={})
    except ModelServerUnavailableError as exc:
        raise TranslationUnavailableError(str(exc)) from exc

    native_text, target_text = _parse_labeled_lines(reply, native_name, target_name)
    if not target_text:
        target_text = translate_text(native_text, native_lang, target_lang)
    return native_text, target_text
