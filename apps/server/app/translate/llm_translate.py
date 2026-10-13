"""Whole-message translation via the same chat model used for conversation
(app/chat/openai_client.py).

Argos's offline MT (long since removed) produced noticeably rough,
sometimes just-plain-wrong translations on short/informal Spanish (e.g.
rendering a simple "¿Qué hobbies te gustan?" as "What do you care about?").
The LLM is already running for chat anyway, so reusing it for this one-shot
translation task is free infrastructure-wise, just slower per call -
callers are expected to fetch this in the background after already showing
the untranslated text, not block the UI on it.
"""

import json
import logging
import re

from app.chat.openai_client import ModelServerUnavailableError, chat as model_chat, chat_stream as model_chat_stream
from app.config import OPENAI_COACH_MODEL, OPENAI_TRANSLATE_MODEL
from app.translate.lemmatizer import analyze
from app.translate.sequential_stream import Field, stream_sequential_fields

logger = logging.getLogger(__name__)

_LANGUAGE_NAMES = {"es": "Spanish", "en": "English"}

# gloss_reply's JSON response needs room for a translation plus one span
# per TOKEN in the source text (not per distinct word, unlike the old
# dict-keyed word_map this replaced - full-coverage spans don't dedupe a
# repeated word like "y" or "el" across its occurrences). A short 2-3
# sentence reply can still run 20+ words once every function word is
# counted, each needing its own {"surface", "gloss", "note", "literal"}
# entry - sized well above tag_draft's per-span budget (1400 for a 5-field
# span) despite gloss_reply's spans only having 4 fields, since this one
# can have more spans in total. Raised from 1400 after adding `literal`
# (a word-by-word breakdown, non-empty only for multi-word spans) - most
# spans leave it empty, but a reply with several idioms/phrasal verbs in
# it needs the extra room. Too tight a budget truncates the JSON mid-
# generation, which reads identically to the model just failing
# (unparseable after retries) - and since there's no per-token fallback,
# that silently drops every gloss in the WHOLE message, not just one word
# - exactly the "lots of words aren't clickable" symptom this sizing
# fixes. Raised again, 1700->2800, alongside the same bump to
# tag_draft's own budget - see _TAG_DRAFT_MAX_TOKENS's comment: against a
# reasoning-style model, a live failure there came back completely empty
# (reasoning tokens alone exhausted a 1600 budget), and this call shares
# the same model and the same new `literal` field, so the same generous
# headroom applies here too, before it gets caught by the exact same
# failure mode on some longer/idiom-heavy reply.
_GLOSS_MESSAGE_MAX_TOKENS = 3200

_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)

# Qwen3 is heavily trained on Chinese data and, on a bad roll, can emit CJK
# tokens instead of the requested Spanish/English - observed live: a
# translation click inserted Chinese characters into the input. Neither
# language this app handles uses these scripts, so any hit here means the
# reply is garbage, not a real translation.
_UNEXPECTED_SCRIPT_RE = re.compile(r"[一-鿿぀-ヿ가-힯]")


class TranslationUnavailableError(RuntimeError):
    pass


def _looks_garbled(text: str) -> bool:
    return bool(_UNEXPECTED_SCRIPT_RE.search(text))


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
    # One retry for a garbled (wrong-script) reply - cheap insurance against
    # an occasional bad sample, not a real error worth surfacing to the user.
    last_result = ""
    for _attempt in range(2):
        try:
            last_result = model_chat(messages, model=OPENAI_TRANSLATE_MODEL).strip()
        except ModelServerUnavailableError as exc:
            raise TranslationUnavailableError(str(exc)) from exc
        if not _looks_garbled(last_result):
            return last_result
    raise TranslationUnavailableError(f"Model returned a garbled reply: {last_result!r}")


def gloss_message(
    text: str, native_lang: str, target_lang: str, learner: bool = False
) -> tuple[str, list[dict[str, str]], list[dict[str, str]]]:
    """Everything the app shows when a word in a chat message is tapped, in
    one call so it all agrees on the sense each word has in THIS sentence.
    Returns (translation, words, phrases), lists ordered left to right:

    - words: EVERY word, one at a time - {"surface", "gloss", "grammar"}.
      `gloss` is just what that one word means here, as short as a
      dictionary headword ("good", "to eat", "the"). `grammar` is optional
      detail about the word's form: for a conjugated verb its infinitive
      and the tense/person ("comer (to eat): I, present"), likewise a
      plural/feminine/contracted form worth knowing - otherwise empty.
    - phrases: ONLY multi-word expressions whose meaning isn't just the
      words added up (idioms, fixed expressions like "qué bueno" = "how
      wonderful", "tener que" = "to have to") - {"surface", "gloss",
      "note"}, `note` an optional bit of usage/context, usually empty.

    `text` is the partner's reply by default (target_lang, translated into
    native_lang). With `learner`, it's the learner's own message, which may
    mix both languages and have mistakes: a target_lang word is glossed in
    native_lang, a native_lang word gets its target_lang translation, and
    no whole-message translation is asked for ("" - the learner's message
    has its own interpretation, see interpret_user_input).

    Raises TranslationUnavailableError if no usable JSON comes back after
    retries - callers then just have no glosses for that message."""
    if not text.strip():
        return "", [], []

    native_name = _LANGUAGE_NAMES.get(native_lang, native_lang)
    target_name = _LANGUAGE_NAMES.get(target_lang, target_lang)
    if learner:
        intro = (
            f"This is a message a {native_name}-speaking learner wrote to their {target_name} "
            f"conversation partner. It may mix {target_name} and {native_name} and have mistakes. "
            f"For a {target_name} word, `gloss` is its {native_name} meaning; for a {native_name} "
            f"word, `gloss` is how to say it in {target_name}. Use the meaning the learner "
            f"clearly intended."
        )
        translation_field = ""
    else:
        intro = (
            f"This is a {target_name} chat message. Translate it to {native_name}, and gloss it "
            f"for a {native_name}-speaking learner."
        )
        translation_field = f'"translation": "<the whole message in {native_name}>", '
    messages = [
        {
            "role": "system",
            "content": (
                f"{intro}\n\n"
                f"`words`: EVERY word in the message, one word per entry, in order (left to right), "
                f"including short words like el, la, de, que, y, mi - skip only punctuation. Copy "
                f"`surface` exactly as written (same spelling, case and accents). `gloss`: what that "
                f"single word means here, as short as a dictionary entry - one to three words, e.g. "
                f'"good", "to eat", "the", "you (informal)". No explanations in `gloss`. `grammar`: '
                f"optional, in {native_name}, about the word's form - for a conjugated verb give the "
                f'infinitive and its meaning, then person and tense, e.g. "comer (to eat): I, present" '
                f'or "ir (to go): they, preterite"; for a plural, feminine, contracted (del, al) or '
                f"object-pronoun form, say so briefly. Leave it EMPTY for a word already in its basic "
                f'dictionary form - never write "masculine singular" or the like for one - and for any '
                f"word that needs nothing more; most words get none.\n\n"
                f"`phrases`: ONLY groups of two or more words whose meaning isn't simply the words "
                f'added up - idioms and fixed expressions like "qué bueno" ("how wonderful"), "tener '
                f'que" ("to have to"), "me gusta" ("I like"), "a veces" ("sometimes"). Most messages '
                f"have none or one. `gloss`: the phrase's {native_name} meaning here. `note`: optional, "
                f"in {native_name}, a short bit of usage or context only if it really helps - usually "
                f"empty. Do not break a phrase down word by word.\n\n"
                f"Reply with ONLY one JSON object, no markdown fences, no explanation, exactly:\n"
                f'{{{translation_field}"words": [{{"surface": "...", "gloss": "...", "grammar": ""}}, ...], '
                f'"phrases": [{{"surface": "...", "gloss": "...", "note": ""}}, ...]}}'
            ),
        },
        {"role": "user", "content": text},
    ]

    last_reply = ""
    last_error: Exception | None = None
    # Three tries: a big JSON payload like this occasionally comes back
    # empty or cut off (see _TAG_DRAFT_MAX_TOKENS on reasoning budgets).
    for attempt in range(3):
        try:
            last_reply = model_chat(messages, max_tokens=_GLOSS_MESSAGE_MAX_TOKENS, model=OPENAI_TRANSLATE_MODEL)
        except ModelServerUnavailableError as exc:
            raise TranslationUnavailableError(str(exc)) from exc
        if not last_reply.strip() or _looks_garbled(last_reply):
            last_error = ValueError("empty or garbled reply")
            logger.warning("gloss_message attempt %d/3 for %r: empty or garbled reply (len=%d)", attempt + 1, text, len(last_reply))
            continue
        try:
            return _extract_gloss_json(last_reply)
        except (ValueError, json.JSONDecodeError, KeyError) as exc:
            last_error = exc
            logger.warning(
                "gloss_message attempt %d/3 for %r: unparseable reply (len=%d): %s", attempt + 1, text, len(last_reply), exc
            )
    raise TranslationUnavailableError(f"Model reply wasn't usable JSON: {last_reply!r} ({last_error})")


def _extract_gloss_json(reply: str) -> tuple[str, list[dict[str, str]], list[dict[str, str]]]:
    match = _JSON_OBJECT_RE.search(reply)
    if not match:
        raise ValueError(f"no JSON object found in reply: {reply!r}")
    data = json.loads(match.group(0))
    translation = data.get("translation", "")
    raw_words = data["words"]
    raw_phrases = data.get("phrases", [])
    if not isinstance(translation, str) or not isinstance(raw_words, list) or not isinstance(raw_phrases, list):
        raise ValueError(f"unexpected JSON shape: {data!r}")

    def entries(raw: list, fields: tuple[str, ...]) -> list[dict[str, str]]:
        out = []
        for item in raw:
            if not isinstance(item, dict) or not str(item.get("surface", "")).strip():
                continue
            out.append({"surface": str(item["surface"]).strip(), **{f: str(item.get(f, "") or "").strip() for f in fields}})
        return out

    words = entries(raw_words, ("gloss", "grammar"))
    phrases = [p for p in entries(raw_phrases, ("gloss", "note")) if len(p["surface"].split()) > 1]
    return translation.strip(), words, phrases


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


def _looks_like_unwanted_reply(original: str, restated: str) -> bool:
    """The model is supposed to correct what the learner typed, not answer
    or continue a conversation with them - observed live: "hello sir" ->
    "Hello, sir. I am here to assist you." A word-count blowup this extreme
    (both a large ratio AND a large absolute difference, so a short input
    can't trip it on a single legitimately-longer word) is a strong signal
    it invented content rather than corrected it. This is a coarse backstop
    for egregious cases, not the primary defense - a subtler overreach
    (e.g. adding one short unearned follow-up question) can look like an
    ordinary longer translation by word count alone and has to be headed
    off in the prompt instead."""
    original_words = len(original.split())
    restated_words = len(restated.split())
    return restated_words > original_words * 2 and restated_words - original_words >= 4


def _looks_spanish(text: str) -> bool:
    """Rough majority-vote check reusing the same Spanish/English classifier
    the word bank already relies on (app/translate/lemmatizer.py) - used to
    catch the case where the model labeled its two lines correctly but put
    the wrong language's content under one of them."""
    words = [t for t in analyze(text) if t.surface.isalpha() and len(t.surface) > 1]
    if not words:
        return False
    return sum(1 for t in words if t.is_spanish) > len(words) / 2


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
                f"greet them back, or ask them anything, even a natural-"
                f"sounding follow-up question. Preserve their meaning and "
                f"length; only fix it, don't extend it. Two examples of "
                f'exactly this mistake: type "hello sir" - a BAD output '
                f'replies to them ("Hello, sir. I am here to assist you."), '
                f'the GOOD output is just their own greeting, corrected '
                f'("Hello, sir."). Type "hola bro" - a BAD output asks them '
                f'something back ("Hello, how are you?"), the GOOD output is '
                f'just their own greeting, corrected ("Hello, bro.").\n\n'
                f"Reply with EXACTLY two lines and nothing else:\n"
                f"{native_name}: <their message, corrected to natural, "
                f"grammatically correct {native_name} - nothing added>\n"
                f"{target_name}: <the same corrected message in natural, "
                f"grammatically correct {target_name} - nothing added>"
            ),
        },
        {"role": "user", "content": text},
    ]
    reply = ""
    for _attempt in range(2):
        try:
            reply = model_chat(messages, model=OPENAI_TRANSLATE_MODEL)
        except ModelServerUnavailableError as exc:
            raise TranslationUnavailableError(str(exc)) from exc
        if not _looks_garbled(reply):
            break
    else:
        # Both attempts came back garbled - fall back to translate_text
        # directly on the raw input rather than trying to parse garbage.
        # translate_text has its own retry, so this is a genuinely
        # independent second chance, not just repeating the same failure.
        target_text = translate_text(text, native_lang, target_lang)
        native_text = translate_text(text, target_lang, native_lang) if _looks_spanish(text) else text
        return native_text, target_text

    native_text, target_text = _parse_labeled_lines(reply, native_name, target_name)

    if native_text and _looks_like_unwanted_reply(text, native_text):
        # The prompt's own instructions and examples are the primary
        # defense against this; this only catches it slipping through
        # anyway. Falls back to a plain, literal translate_text pass, which
        # has no "conversation" framing for the model to go off-script
        # with.
        target_text = translate_text(text, native_lang, target_lang)
        native_text = translate_text(text, target_lang, native_lang) if _looks_spanish(text) else text
        return native_text, target_text

    # A small model can label its two lines correctly but swap which
    # language's content goes under which label - observed live: the
    # "English" row came back showing Spanish text. Catch that rather than
    # trust the labels blindly, when we know target_lang is Spanish (the
    # only case this app actually exercises - the classifier is Spanish-
    # specific, see _looks_spanish).
    if native_text and target_text and target_lang == "es" and native_lang == "en":
        if _looks_spanish(native_text) and not _looks_spanish(target_text):
            native_text, target_text = target_text, native_text
        elif not _looks_spanish(target_text):
            # Not a swap - the model just left the "Spanish" line in
            # English too (observed live: "sup bro. there once was a red
            # monkey" came back with the same English text under both
            # labels). The swap above only fixes mislabeled lines; this
            # catches the line never being translated at all by forcing a
            # real translation rather than surfacing English where Spanish
            # was asked for.
            target_text = translate_text(native_text, native_lang, target_lang)

    # _parse_labeled_lines already guarantees native_text is non-empty
    # unless the model's reply itself was blank (native_text falls back to
    # the whole raw reply when no labels matched at all) - so only one of
    # these two branches can actually fire for non-empty input.
    if not target_text and native_text:
        target_text = translate_text(native_text, native_lang, target_lang)
    elif not native_text and target_text:
        native_text = translate_text(target_text, target_lang, native_lang)
    return native_text, target_text


_TAG_DRAFT_MAX_TOKENS = 3000  # each span here carries 6 string fields
# (surface/gloss/note/translation/alternate_gloss/literal) - most of them
# empty for a given span (alternate_gloss nearly always, literal unless
# that span is a multi-word group), but all 6 still need room in the cap.
# This is just a cap, not a target - the model stops on its own once the
# JSON is complete, so it doesn't by itself cost latency; it exists so a
# genuinely long/complex draft can't get truncated mid-object, which reads
# identically to the model just failing (unparseable after retries,
# raising TranslationUnavailableError) rather than an obviously-wrong but
# diagnosable response. The actual latency lever is how much the prompt
# asks the model to generate per call - see tag_draft's docstring on
# dropping the unused whole-draft `translation` field for exactly that
# reason.
#
# Raised from 1600 after a live failure (all 3 attempts came back
# completely empty, len=0, not just truncated) on a draft combining ALL
# THREE of tag_draft's asks at once - cognate detection
# (`alternate_gloss`), an idiom ("miss you"), and this round's new
# `literal` breakdown - against OPENAI_TRANSLATE_MODEL, which (per
# openai_client.py's own comment on requiring max_completion_tokens) is a
# reasoning-style model: its internal reasoning tokens count against this
# same budget before any visible output, and a heavier combined-task
# prompt can burn through a tight budget on reasoning alone, leaving
# nothing for the actual JSON. Chosen generously since, per above, this
# cap costs nothing when unused.


def _coerce_spans(raw_spans: list) -> list[dict[str, str]]:
    """Shared span-dict normalization for any LLM call that returns a
    surface/gloss/note/translation/alternate_gloss/literal list in this
    shape - tag_draft below, and coach_draft_stream's per-option word-by-
    word breakdown (app.routes.translate's streaming coach endpoint)
    alike."""
    spans: list[dict[str, str]] = []
    for item in raw_spans:
        if not isinstance(item, dict):
            continue
        surface = str(item.get("surface", "")).strip()
        if not surface:
            continue
        spans.append(
            {
                "surface": surface,
                "gloss": str(item.get("gloss", "")),
                "note": str(item.get("note", "")),
                "translation": str(item.get("translation", "")),
                "alternate_gloss": str(item.get("alternate_gloss", "")),
                "literal": str(item.get("literal", "")),
            }
        )
    return spans


def _extract_span_list_json(reply: str) -> tuple[str, list[dict[str, str]]]:
    match = _JSON_OBJECT_RE.search(reply)
    if not match:
        raise ValueError(f"no JSON object found in reply: {reply!r}")
    data = json.loads(match.group(0))
    # "translation" defaults to "" rather than being required - tag_draft
    # below doesn't ask the model for it at all (its only caller discarded
    # it, so asking for it was pure wasted generation - see tag_draft's own
    # docstring), while gloss_reply still does and gets a real value back.
    translation = data.get("translation", "")
    raw_spans = data["spans"]
    if not isinstance(translation, str) or not isinstance(raw_spans, list):
        raise ValueError(f"unexpected JSON shape: {data!r}")
    return translation, _coerce_spans(raw_spans)


def tag_draft(draft: str, native_lang: str, target_lang: str) -> tuple[str, list[dict[str, str]]]:
    """For text the learner is still typing (not sent yet) - possibly mixing
    native_lang/target_lang, with grammar/spelling mistakes in either (same
    messy-input framing as coach_draft/interpret_user_input, but with NO
    conversation history - this needs to stay fast/cheap, since it's called
    far more often than coach_draft, once per word/boundary rather than once
    per click). Still returns a (translation, spans) pair for the same
    shape _extract_span_list_json/gloss_reply use, but translation is
    always "" here - unlike gloss_reply, this doesn't ask the model for a
    whole-draft translation at all (its only caller, /translate/gloss-
    spans, discarded that value entirely; asking for it was pure wasted
    generation on every single hover/tap, the most latency-sensitive call
    in this file). spans is an ORDERED list, not a dict keyed by surface
    text like gloss_reply's word_map - gloss_reply can get away with a
    dict because every occurrence of a repeated word shares one gloss, but
    here the same surface text can genuinely appear twice in one draft and
    each occurrence needs its own later position-match (see
    app.translate.span_matching.match_spans) - a dict would silently
    collapse them. Each span dict is
        {"surface": <copied exactly from the draft, not paraphrased>,
         "gloss": <short native_lang meaning in context>,
         "note": <disambiguation note, or empty>,
         "translation": <target_lang replacement text, empty if the
            surface text is already natural target_lang as typed>,
         "alternate_gloss": <empty for almost every span; non-empty only
            when a standalone word is ALSO a legitimate, different word in
            native_lang (a true cross-language cognate, e.g. "once" -
            Spanish for "eleven", also an English word) - that other
            word's meaning, in native_lang>,
         "literal": <empty for a single-word span; for a multi-word span,
            a short native_lang word-by-word breakdown showing how the
            group's individual words combine - e.g. for "tener en cuenta"
            (gloss "take into account"), something like "tener (to have) +
            en (in) + cuenta (account)">}.
    The model decides span boundaries - usually one word, occasionally a
    few words grouped as an idiom/phrasal verb/fixed expression. This does
    NOT itself guarantee the surface text is actually findable in draft at
    an exact offset - that matching happens separately (span_matching.py),
    so a span can be silently dropped downstream rather than misplaced.
    Raises TranslationUnavailableError only if the model's reply isn't
    parseable JSON even after a retry."""
    if not draft.strip():
        return "", []

    native_name = _LANGUAGE_NAMES.get(native_lang, native_lang)
    target_name = _LANGUAGE_NAMES.get(target_lang, target_lang)
    messages = [
        {
            "role": "system",
            "content": (
                f"A language learner, whose native language is {native_name}, is "
                f"typing a message in {target_name} - still a draft, not sent yet. "
                f"It may mix {native_name} and {target_name}, and may have grammar, "
                f"spelling, or word-order mistakes in either. Your job: break the "
                f"draft into a list of words or short word-groups, IN THE SAME ORDER "
                f"THEY APPEAR IN THE DRAFT (left to right, earliest first), that "
                f"together cover as much of the draft as you reasonably can, "
                f"skipping pure punctuation. For each one, give a short {native_name} "
                f"gloss of what it means here, and - only if it ISN'T already "
                f"written as natural {target_name} - a corrected/translated "
                f"{target_name} replacement for it (leave this empty if the "
                f"learner's own text for that span is already correct, natural "
                f"{target_name} - there's nothing to replace). Group a few words "
                f"together ONLY when they form an idiom, phrasal verb, or fixed "
                f"expression that doesn't translate word-by-word - most spans should "
                f"be a single word. Copy each span's `surface` EXACTLY as it appears "
                f"in the draft - same spelling, same case, same accents, not "
                f"corrected or paraphrased (corrections belong only in its own "
                f"`translation` field). Also, for a span that is just ONE standalone "
                f"word, check whether that exact spelling is ALSO a legitimate, "
                f"different word in {native_name} (a true cross-language cognate - "
                f'for example "once" is Spanish for "eleven" but also an ordinary '
                f"English word). If so, give that other word's meaning as "
                f"`alternate_gloss`, in {native_name} - otherwise leave "
                f"`alternate_gloss` an empty string; it will be empty for nearly "
                f"every span. For a MULTI-WORD span only, also give a short "
                f"`literal` word-by-word breakdown in {native_name}, showing what "
                f"each individual word in the group means on its own and, if the "
                f"group's overall sense isn't a literal/word-for-word match for "
                f'`gloss`, making that plain too - e.g. for a {target_name} idiom '
                f'span like "tener en cuenta" meaning "take into account", '
                f'`literal` could be "tener (to have) + en (in) + cuenta '
                f'(account)" - so the learner can see how the phrase is built, '
                f"not just what it means overall. Leave `literal` an empty string "
                f"for a single-word span - it would just repeat `gloss`.\n\n"
                f"Reply with ONLY a single JSON object, no markdown fences, no "
                f"explanation, in exactly this shape:\n"
                f'{{"spans": [{{"surface": "<exact text from the '
                f'draft>", "gloss": "<its meaning here, in {native_name}>", "note": '
                f'"<in {native_name}: short reason if this sense could be confused '
                f'with another - else empty>", "translation": "<a corrected/'
                f'translated {target_name} replacement - empty if already '
                f'correct, natural {target_name}>", "alternate_gloss": "<empty, or '
                f"if this single-word span is ALSO a legitimate different word in "
                f'{native_name}, that word\'s meaning, in {native_name}>", '
                f'"literal": "<word-by-word breakdown for a multi-word span '
                f'only - else empty>"}}, ...]}}'
            ),
        },
        {"role": "user", "content": draft},
    ]

    last_reply = ""
    last_error: Exception | None = None
    # Observed live (Railway logs, both before and after this round's own
    # changes - not a regression this round introduced): this call
    # occasionally comes back completely empty for a longer/more complex
    # draft, back to back, exhausting the usual 2 attempts other LLM-
    # backed helpers here use. One extra attempt is cheap insurance - this
    # has the biggest, most complex JSON payload of any call in this
    # file (5 string fields per span), so it's the one most worth paying
    # for a third try.
    for attempt in range(3):
        try:
            last_reply = model_chat(messages, max_tokens=_TAG_DRAFT_MAX_TOKENS, model=OPENAI_TRANSLATE_MODEL)
        except ModelServerUnavailableError as exc:
            raise TranslationUnavailableError(str(exc)) from exc
        if not last_reply.strip() or _looks_garbled(last_reply):
            last_error = ValueError("empty or garbled reply")
            # Per-attempt, not just on final failure - if this keeps
            # happening, knowing whether EVERY attempt came back truly
            # empty (vs. some attempts returning unparseable-but-non-
            # empty content) narrows down the cause a lot faster than
            # re-diagnosing blind from just the last attempt's value.
            logger.warning("tag_draft attempt %d/3 for %r: empty or garbled reply (len=%d)", attempt + 1, draft, len(last_reply))
            continue
        try:
            return _extract_span_list_json(last_reply)
        except (ValueError, json.JSONDecodeError, KeyError) as exc:
            last_error = exc
            logger.warning(
                "tag_draft attempt %d/3 for %r: unparseable reply (len=%d): %s", attempt + 1, draft, len(last_reply), exc
            )
    raise TranslationUnavailableError(f"Model reply wasn't usable JSON: {last_reply!r} ({last_error})")


# The coach: one conversation about what the learner wants to say next and
# about the messages so far. It answers questions, explains words and
# grammar, drills, and - whenever it suggests something to say - writes
# the suggestion as a [[Spanish | English meaning]] block, which the app
# renders as a card with "Use this". Read aloud as well as shown.
_COACH_MAX_TOKENS = 1800
_COACH_HISTORY_MESSAGES = 8
_COACH_ABOUT_MARKER = "<<<ABOUT>>>"
_COACH_READY_MARKER = "<<<READY>>>"

# The model sometimes echoes a field's name into the field ("READY: yes",
# "REPLY: ..."), which would break parsing or get read aloud. Stripped from
# the start of each field; the prompt also tells it not to.
_FIELD_LABEL_RE = re.compile(r"^\s*(?:ABOUT|READY|REPLY|ANSWER)\b\s*[:\-–—]?\s*", re.IGNORECASE)
# How much of the reply to hold back before deciding whether it starts
# with an echoed label - "ANSWER" plus a separator is 8 chars at most.
_REPLY_HOLDBACK = 8


def _strip_field_label(text: str) -> str:
    return _FIELD_LABEL_RE.sub("", text, count=1)


def _run_sequential_fields(chunks, fields, initial_buffer: str = ""):
    """Drives sequential_stream.stream_sequential_fields and hands its
    leftover buffer back to a plain caller: returns (iterator, box), where
    box[0] holds the leftover once the iterator is exhausted."""
    leftover_box: list[str] = []

    def driver():
        gen = stream_sequential_fields(chunks, fields, initial_buffer)
        while True:
            try:
                item = next(gen)
            except StopIteration as stop:
                leftover_box.append(stop.value or "")
                return
            yield item

    return driver(), leftover_box


def coach_reply_stream(
    text: str,
    messages: list[dict],
    default_number: int | None,
    thread_turns: list[tuple[str, str]],
    draft_turns: list[tuple[str, str]],
    native_lang: str,
    target_lang: str,
    check: bool = False,
):
    """Streams the coach's reply to `text`.

    `messages` is the recent conversation with the partner, oldest first,
    each {"number", "role" ("user"|"assistant"), "text", optional
    "glosses"}. `default_number` is the message a question is most likely
    about (the thread open on screen), or None. `thread_turns` is the side
    conversation already under that message; `draft_turns` the learner's
    discussion with the coach about what to say next, both as (role, text)
    with role "learner" | "coach". `check` = `text` is the message the
    learner is about to send, as written in the message box.

    Yields, in order:
      ("about", number_or_None) - which partner message this is about;
        None = the learner's own next message / nothing in particular.
      ("ready", bool) - the learner's own Spanish in `text` is already
        correct and natural to send as written.
      ("reply_chunk", delta) ... then ("reply_complete", full_text).
    Raises TranslationUnavailableError if the model can't be reached or
    produced no reply at all."""
    if not text.strip():
        return

    native_name = _LANGUAGE_NAMES.get(native_lang, native_lang)
    target_name = _LANGUAGE_NAMES.get(target_lang, target_lang)
    recent = messages[-_COACH_HISTORY_MESSAGES:]
    numbers = {m["number"] for m in recent}
    listing = "\n".join(
        f"[{m['number']}] {'Learner' if m['role'] == 'user' else 'Partner'}: {m['text']}"
        + (f"\n    (word glosses: {m['glosses']})" if m.get("glosses") else "")
        for m in recent
    )

    def lines(turns: list[tuple[str, str]]) -> str:
        return "\n".join(f"{'Learner' if role == 'learner' else 'Coach'}: {t}" for role, t in turns[-10:])

    thread_lines = lines(thread_turns)
    draft_lines = lines(draft_turns)
    context = f"The conversation with their partner, numbered:\n{listing or '(nothing yet)'}\n\n"
    if default_number is not None and thread_lines:
        context += f"Your side conversation about message [{default_number}] so far:\n{thread_lines}\n\n"
    elif default_number is not None:
        context += f"They have message [{default_number}] open to talk about it.\n\n"
    if draft_lines:
        context += f"Your conversation about what they'll say next, so far:\n{draft_lines}\n\n"
    about_hint = (
        f"If they don't clearly mean something else, it's about message [{default_number}]."
        if default_number is not None
        else "Most of the time it's 0."
    )
    check_note = (
        "\nThe learner's message below is the exact message they're about to send to their partner "
        "(typed or dictated in their message box) - treat it as what they want to say.\n"
        if check
        else ""
    )

    prompt = [
        {
            "role": "system",
            "content": (
                f"You are a warm, sharp {target_name} coach for a {native_name} speaker who is "
                f"chatting in {target_name} with a conversation partner. You and the learner talk "
                f"on the side: they work out what to say next, ask what things mean, ask about "
                f"grammar, practise words, or just think out loud. Answer like a real coach in a "
                f"real conversation - whatever fits: an answer, an explanation, a suggestion, a "
                f"quick correction, encouragement, a question back.\n\n"
                f"{context}"
                f"How to read what they write:\n"
                f"- If they write {target_name} (or mixed {target_name}/{native_name}) with no "
                f"framing, it's most likely what they want to say to their partner - react to it "
                f"as an attempt.\n"
                f"- \"I want to say ...\", \"how do I tell him ...\" = they want to say that.\n"
                f"- \"More casual\", \"shorter\", \"use querer\" adjust your last suggestion.\n"
                f"- Questions about words, grammar or meaning - answer them; drill if they repeat "
                f"a word back.\n\n"
                f"Suggestions: whenever you suggest something for them to say, write it exactly "
                f"as [[{target_name} sentence | {native_name} meaning]] - one block per suggestion, "
                f"usually just one, more only if they ask for options. If their {target_name} is "
                f"already right, say so briefly and give it back as a block. Never use [[ ]] for "
                f"anything else.\n\n"
                f"Style: {native_name}, with {target_name} words in quotes outside the blocks. "
                f"Short, natural sentences - it's read aloud. No markdown, lists, headings or "
                f"emoji. Don't open with \"Did you mean\" by default; talk normally.\n\n"
                f"Write exactly three fields, in order, and never write a field's name:\n"
                f"1. The number of the partner message this is about (digits only), or 0 if it's "
                f"about what they'll say next or nothing in particular. {about_hint} Then this "
                f"marker: {_COACH_ABOUT_MARKER}\n"
                f"2. yes if the learner's own {target_name} in their latest message is already "
                f"correct and natural to send as written, otherwise no (no for questions and for "
                f"{native_name}). Then this marker: {_COACH_READY_MARKER}\n"
                f"3. Your reply, and nothing after it."
                f"{check_note}"
            ),
        },
        {"role": "user", "content": text},
    ]

    try:
        chunks = model_chat_stream(prompt, max_tokens=_COACH_MAX_TOKENS, model=OPENAI_COACH_MODEL)
        head_iter, head_leftover = _run_sequential_fields(
            chunks, [Field("about", _COACH_ABOUT_MARKER), Field("ready", _COACH_READY_MARKER)]
        )
        about_raw: str | None = None
        ready_raw: str | None = None
        for kind, value in head_iter:
            if kind == "about_complete":
                about_raw = value
            elif kind == "ready_complete":
                ready_raw = value

        about: int | None = default_number
        if about_raw is not None:
            digits = re.search(r"\d+", about_raw)
            if digits:
                number = int(digits.group())
                about = number if number in numbers else (None if number == 0 else default_number)
        yield ("about", about)
        ready = bool(ready_raw and re.search(r"\byes\b", _strip_field_label(ready_raw), re.IGNORECASE))
        yield ("ready", ready)

        # Everything after the last marker is the reply. If a marker never
        # came, the leftover holds the rest of the text - keep it as the
        # reply rather than losing it.
        pending = head_leftover[0] if head_leftover else ""
        reply = ""
        head_done = False

        def emit(piece: str) -> str | None:
            nonlocal reply, head_done, pending
            pending += piece
            if not head_done:
                if len(pending) < _REPLY_HOLDBACK:
                    return None
                pending = _strip_field_label(pending.lstrip())
                head_done = True
            out, pending = pending, ""
            reply += out
            return out or None

        first = emit("")
        if first:
            yield ("reply_chunk", first)
        for chunk in chunks:
            out = emit(chunk)
            if out:
                yield ("reply_chunk", out)
        if not head_done:
            out = _strip_field_label(pending.lstrip())
            if out:
                reply += out
                yield ("reply_chunk", out)
        reply = reply.strip()
        if not reply:
            raise TranslationUnavailableError("Model reply had no text")
        yield ("reply_complete", reply)
    except ModelServerUnavailableError as exc:
        raise TranslationUnavailableError(str(exc)) from exc
