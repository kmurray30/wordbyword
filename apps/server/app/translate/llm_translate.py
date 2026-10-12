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
from app.config import OPENAI_TRANSLATE_MODEL, OPENAI_TUTOR_MODEL
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
_GLOSS_REPLY_MAX_TOKENS = 2800

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


def gloss_reply(text: str, source_lang: str, target_lang: str) -> tuple[str, list[dict[str, str]]]:
    """Translates `text` (the assistant's own reply) and, in the same call,
    glosses it word-by-word using each word's/group's meaning IN THIS
    SENTENCE - one consistent source for both, instead of a separately-
    fetched whole-sentence translation and per-word glosses that could each
    land on a different sense of an ambiguous word. Returns
    (translation, raw_spans), the exact same ordered-span-list shape
    tag_draft returns (ready for app.translate.span_matching.match_spans) -
    deliberately NOT a dict keyed by surface text like this function used
    to return: a dict can't represent a multi-word group (e.g. "el tuyo")
    without one of its words winning and the other vanishing, and also
    silently collapses a repeated surface form's second occurrence onto
    the first. Spans must cover EVERY word, including short function words
    (y, mi, el, la, de, que, ...) - skipping those was the original bug
    this shape change fixes. Raises TranslationUnavailableError if the
    model's reply isn't parseable JSON even after a retry - callers get no
    gloss for that turn rather than a dictionary/MT fallback."""
    if not text.strip():
        return "", []

    source_name = _LANGUAGE_NAMES.get(source_lang, source_lang)
    target_name = _LANGUAGE_NAMES.get(target_lang, target_lang)
    messages = [
        {
            "role": "system",
            "content": (
                f"Translate the following {source_name} text to {target_name}, "
                f"and also break it into a list of words or short word-groups, "
                f"IN THE SAME ORDER THEY APPEAR IN THE TEXT (left to right, "
                f"earliest first), that together cover EVERY word in it - "
                f"including short function words like y, mi, el, la, de, que, "
                f"and the like. Do not skip any word. Skip only pure "
                f"punctuation. Group a few words together ONLY when they form "
                f"a fixed expression that doesn't translate word-by-word (for "
                f'example "el tuyo" meaning "yours" should be ONE span, not '
                f'"el" and "tuyo" separately) - most spans should be a single '
                f"word. Copy each span's `surface` EXACTLY as it appears in "
                f"the text - same spelling, same case, same accents. For each "
                f"span, give a short {target_name} gloss of what it means "
                f"IN THIS SENTENCE - not a generic dictionary definition, "
                f"since the same word/group can mean different things in "
                f"different sentences. For a MULTI-WORD span only, also give "
                f"a short `literal` word-by-word breakdown in {target_name}, "
                f"showing what each individual word in the group means on "
                f"its own and, if the group's overall sense isn't a literal/"
                f'word-for-word match for `gloss`, making that plain too - '
                f'e.g. for a {source_name} idiom span like "tener en cuenta" '
                f'meaning "take into account", `literal` could be "tener (to '
                f'have) + en (in) + cuenta (account)" - so the learner can '
                f"see how the phrase is built, not just what it means "
                f"overall. Leave `literal` an empty string for a single-word "
                f"span - it would just repeat `gloss`. Reply with ONLY a "
                f"single JSON object and nothing else - no markdown code "
                f"fences, no explanation - in exactly this shape: "
                f'{{"translation": "<the full {target_name} translation>", '
                f'"spans": [{{"surface": "<exact text from the source, one '
                f'word or a short fixed-expression group>", "gloss": "<its '
                f'{target_name} meaning in this sentence>", "note": "<in '
                f"{target_name}: if this word could easily be confused with "
                f"a different sense or word, one short phrase explaining why "
                f'this sense applies here - otherwise an empty string>", '
                f'"literal": "<word-by-word breakdown for a multi-word span '
                f'only - else empty>"}}, ...]}}'
            ),
        },
        {"role": "user", "content": text},
    ]

    last_reply = ""
    last_error: Exception | None = None
    # 3 attempts, not 2 - same reasoning as tag_draft's own retry count:
    # this is the other call in this file with a large, multi-field-per-
    # span JSON payload, so it's worth the same extra insurance against an
    # occasional truncated/empty completion.
    for attempt in range(3):
        try:
            last_reply = model_chat(messages, max_tokens=_GLOSS_REPLY_MAX_TOKENS, model=OPENAI_TRANSLATE_MODEL)
        except ModelServerUnavailableError as exc:
            raise TranslationUnavailableError(str(exc)) from exc
        if not last_reply.strip() or _looks_garbled(last_reply):
            last_error = ValueError("empty or garbled reply")
            logger.warning("gloss_reply attempt %d/3 for %r: empty or garbled reply (len=%d)", attempt + 1, text, len(last_reply))
            continue
        try:
            return _extract_span_list_json(last_reply)
        except (ValueError, json.JSONDecodeError, KeyError) as exc:
            last_error = exc
            logger.warning(
                "gloss_reply attempt %d/3 for %r: unparseable reply (len=%d): %s", attempt + 1, text, len(last_reply), exc
            )
    raise TranslationUnavailableError(f"Model reply wasn't usable JSON: {last_reply!r} ({last_error})")


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


def _correct_option_languages(spanish_text: str, english_text: str, native_lang: str, target_lang: str) -> tuple[str, str]:
    """Post-hoc sanity check for one coach_draft_stream FIX-verdict option's
    Spanish suggestion plus its own English back-translation - the plain-
    text, marker-delimited protocol has no structural anchor (unlike the
    old JSON-keyed one) forcing the model to keep each field in the
    language its marker name promises, and a small/fast model sometimes
    drifts. Reuses the same _looks_spanish classifier and the same
    target_lang=="es"/native_lang=="en" scoping interpret_user_input's own
    swap-correction already relies on, extended to the two cases a plain
    swap doesn't cover: either field just never got translated out of the
    OTHER language at all. Degrades to the original pair, uncorrected, if
    the extra translate_text call this needs is itself unavailable."""
    if target_lang != "es" or native_lang != "en" or not spanish_text or not english_text:
        return spanish_text, english_text

    spanish_is_spanish = _looks_spanish(spanish_text)
    english_is_spanish = _looks_spanish(english_text)

    if spanish_is_spanish and not english_is_spanish:
        return spanish_text, english_text

    try:
        if not spanish_is_spanish and english_is_spanish:
            # Classic swap - the right content landed under the wrong label.
            return english_text, spanish_text
        if not spanish_is_spanish and not english_is_spanish:
            # The "Spanish" field was never actually translated into Spanish.
            return translate_text(spanish_text, native_lang, target_lang), english_text
        # Both fields look Spanish - the "English" field was never
        # translated into English.
        return spanish_text, translate_text(english_text, target_lang, native_lang)
    except TranslationUnavailableError:
        return spanish_text, english_text


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


# Shared across the WHOLE call (verdict + meaning + up to 3x(spanish +
# english) for a FIX verdict) rather than split per-field, same reasoning
# the old PART1/PART2 split budget this replaces had for not splitting
# further: a hard per-field cap risks one field starving a later one with
# no way for the model to redistribute leftover budget, and that only gets
# worse the more fields there are (8 here vs. 2 before). Lower than the
# old combined 2800 despite more fields: the whole protocol is plain text
# now, no JSON braces/keys/quoting overhead anywhere, and - the single
# biggest token sink the old protocol had, per _COACH_TRANSLATIONS_MAX_
# TOKENS's own history below - a picked option's word-by-word gloss
# breakdown is no longer part of this call at all; it's fetched lazily via
# /translate/gloss-spans only once an option is actually picked (see
# ChatInput.tsx's handleSelectCoachOption). Still treat this as a starting
# point to tune against real output, not a settled number: this model
# (OPENAI_CHAT_MODEL, a reasoning-style model - see openai_client.py's own
# comment on requiring max_completion_tokens) has repeatedly come back
# completely empty under budget pressure elsewhere in this file
# (_TAG_DRAFT_MAX_TOKENS's history is the clearest example) when its own
# internal reasoning tokens alone exhausted a tight cap before any visible
# output - the same failure mode remains possible here.
_COACH_STREAM_MAX_TOKENS = 2400
_COACH_HISTORY_TURNS = 8
_COACH_FORMALITIES = ("neutral", "casual", "formal")  # fixed, positional - never parsed from the model
# How many suggestions a FIX verdict gives. One (Kyle: "just give one
# suggestion"); the protocol still numbers it as option 0.
_COACH_SUGGESTIONS = 1
_COACH_VERDICTS = {"clean", "minor", "fix"}

# Written literally into the prompt and watched for verbatim in the
# streamed reply, same convention every marker in this file already
# follows - each one terminates the field named after it. Unlike the
# two-marker protocol this replaces, there are now up to 8 of these in one
# call (a verdict, a meaning, then up to 6 per-option fields for a FIX
# verdict) - app.translate.sequential_stream.stream_sequential_fields
# drives the whole sequence generically; see its own docstring for how the
# marker-safe-tail withholding generalizes across all of them at once.
_COACH_VERDICT_MARKER = "<<<VERDICT>>>"
_COACH_MEANING_MARKER = "<<<MEANING>>>"
_COACH_SUGGESTION_MARKER = "<<<SUGGESTION>>>"
_COACH_OPT_MARKERS = (
    "<<<OPT1_ES>>>",
    "<<<OPT1_EN>>>",
    "<<<OPT2_ES>>>",
    "<<<OPT2_EN>>>",
    "<<<OPT3_ES>>>",
    "<<<OPT3_EN>>>",
)


# "I want to say X" / "How do I tell him X" -> X. The prompt asks for the
# message itself, but the model keeps echoing the framing (seen live), and
# the restatement is what the learner reads and hears ("Did you mean ...?").
_SAY_FRAMING_RE = re.compile(
    r"^\s*(?:i(?:\s+(?:want|wanna|would\s+like|need|am\s+trying)|'d\s+like|'m\s+trying)\s+to\s+"
    r"|how\s+(?:do|can|would|should)\s+i\s+)"
    r"(?:say|tell\s+(?:him|her|them|you|my\s+partner))\b\s*(?:that\b)?\s*[:,]?\s*",
    re.IGNORECASE,
)


def _strip_say_framing(meaning: str) -> str:
    text = meaning.strip()
    match = _SAY_FRAMING_RE.match(text)
    if not match:
        return text
    rest = text[match.end() :].strip().strip("\"'“”‘’").strip()
    if text.lower().startswith("how"):
        # "How do I tell him I'm tired?" - the "?" belonged to the framing.
        rest = rest.rstrip("?").rstrip()
    if not rest:
        return text
    return rest[0].upper() + rest[1:]


def _run_sequential_fields(chunks, fields, initial_buffer: str = ""):
    """Drives sequential_stream.stream_sequential_fields and exposes its
    leftover-buffer return value to a plain caller - a generator's own
    `return <value>` is only visible via `StopIteration.value`, which a
    normal `for` loop has no way to capture, so this hands it back instead
    as the second element of the returned tuple (an itself-exhausted-by-
    then single-item list, filled in once the wrapped generator truly
    finishes - read it only AFTER fully iterating the first element)."""
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


def coach_draft_stream(
    draft: str,
    history: list[tuple[str, str]],
    native_lang: str,
    target_lang: str,
    open_thread: tuple[str, list[tuple[str, str]]] | None = None,
    allow_ask: bool = True,
    draft_turns: list[tuple[str, str]] | None = None,
):
    """For a message the learner is still drafting (not yet sent): decides
    whether it's already good to go, restates what it thinks they meant,
    and - only if real correction is needed - suggests the ideal phrasing
    at a few formality levels, given the recent conversation's tone and
    context. ONE streamed LLM call, entirely plain text (no JSON anywhere,
    unlike this function's earlier JSON-blob-based protocol - removes a
    whole class of truncation/parse failure this file has hit repeatedly
    elsewhere), driven field-by-field via app.translate.sequential_stream:

    First, a VERDICT - "clean" (nothing to fix at all), "minor" (fine to
    send as-is, but one small cosmetic suggestion follows), or "fix" (real
    correction needed) - yielded as ("verdict", "clean"|"minor"|"fix") the
    moment it's parsed, before anything else has streamed. This is the one
    piece of the old "core" event's `feedback`/`options` a caller actually
    needs INSTANTLY: it's what lets the Send button light up the moment a
    draft is known good, well before the explanation has finished
    streaming (see ChatInput.tsx's coachReady vs. the slower-to-settle
    coachVerified).

    Then the restated meaning streams live, delta by delta - the one part
    of the reply worth reading char-by-char while it's still generating,
    same "zipper" feel /chat/turn/stream's own reply tokens have -
    yielded as ("meaning_chunk", delta) per newly-arrived, marker-safe
    piece of text, then ("meaning_complete", full_text) once settled.

    What follows depends on the verdict:
    - "clean": nothing further - the generator simply ends.
    - "minor": a single corrected suggestion streams the same way -
      ("suggestion_chunk", delta) then ("suggestion_complete", full_text).
    - "fix": three formality-ranked suggestions stream in order (index 0 =
      neutral, 1 = casual, 2 = formal - see _COACH_FORMALITIES), each
      INTERLEAVED with its own whole-sentence native_lang back-translation
      immediately after - not batched together at the end the way the old
      protocol's PART 3 was: ("option_spanish_chunk", i, delta) /
      ("option_spanish_complete", i, text), then ("option_english_chunk",
      i, delta) / ("option_english_complete", i, text), repeated for i in
      0, 1, 2. A picked option's word-by-word gloss breakdown is
      DELIBERATELY not part of this stream at all - see
      _COACH_STREAM_TOKENS's own comment on why, and ChatInput.tsx's
      handleSelectCoachOption for where it's fetched instead.

    Unlike every other LLM-backed helper here, this does NOT retry on a
    garbled/unusable reply - earlier fields may already have been yielded
    (and acted on by the caller) by the time a later one turns out
    unusable, so there's nothing to cleanly retry; a field that never
    completes (chunks ran out, or the model skipped straight past it)
    just means the generator yields one fewer event than the full
    sequence - whatever DID complete stays fully usable (see
    sequential_stream's own docstring on why fields are independent this
    way). Raises TranslationUnavailableError only when NOTHING at all is
    usable - the verdict marker itself never arrived.

    With `allow_ask` (the default), there's a fourth verdict, "ask": the
    learner's text isn't something to say to the partner at all but a
    question for the tutor ("what does llamas mean?", "break that down").
    That yields ("verdict", "ask") and stops right there - the caller hands
    the text to tutor_answer_stream instead - so routing costs only the
    verdict's few tokens. `open_thread` = (the message the learner is
    currently asking about, that thread's recent turns as (role, text)):
    with it, a bare follow-up like "llamas?" reads as another question."""
    if not draft.strip():
        return

    native_name = _LANGUAGE_NAMES.get(native_lang, native_lang)
    target_name = _LANGUAGE_NAMES.get(target_lang, target_lang)

    recent = history[-_COACH_HISTORY_TURNS:]
    transcript = "\n".join(f"{'Learner' if role == 'user' else 'Partner'}: {text}" for role, text in recent)
    context_block = f"The conversation so far:\n{transcript}\n\n" if transcript else ""
    if open_thread is not None:
        anchor_text, thread_turns = open_thread
        thread_lines = "\n".join(
            f"{'Learner' if role == 'learner' else 'Tutor'}: {text}" for role, text in thread_turns[-6:]
        )
        context_block += (
            f"The learner currently has a side conversation open with their tutor "
            f"about this message: \"{anchor_text}\"\n"
            + (f"{thread_lines}\n" if thread_lines else "")
            + "\n"
        )
    ask_rule = (
        (
            f"- ASK: the text is NOT something they want to say to their partner "
            f"but a question or request for their tutor ABOUT the language or the "
            f"conversation - e.g. \"what does llamas mean?\", \"can you break "
            f"that down?\", \"I didn't understand that\", \"why is it gusta and "
            f"not gustas?\", or a bare word/phrase echoed back as a question "
            f"(\"llamas?\", \"¿cómo?\") while a side conversation is open. "
            f"Anything they want to SAY to the partner - in any language, "
            f"including \"I want to say ...\" / \"how do I tell him ...\" - is "
            f"NOT ASK; judge it as CLEAN/MINOR/FIX. If you choose ASK, write "
            f"nothing after the verdict marker.\n"
        )
        if allow_ask
        else ""
    )
    if draft_turns:
        draft_lines = "\n".join(
            f"{'Learner' if role == 'learner' else 'Coach'}: {text}" for role, text in draft_turns[-8:]
        )
        context_block += (
            f"The learner is working out what to say NEXT, together with you; the "
            f"discussion so far:\n{draft_lines}\n"
            f"If their text asks to adjust an earlier suggestion (\"more casual\", "
            f"\"shorter\", \"use querer instead\", \"and add that I'm tired\"), "
            f"treat it as an attempt: the draft to judge is the sentence they now "
            f"want, so give FIX with the adjusted suggestion.\n\n"
        )
    verdict_words = "CLEAN, MINOR, FIX, or ASK" if allow_ask else "CLEAN, MINOR, or FIX"

    messages = [
        {
            "role": "system",
            "content": (
                f"You are a warm, encouraging writing coach for someone learning "
                f"{target_name}, whose native language is {native_name}. They are "
                f"drafting their NEXT message in the conversation below - it may "
                f"mix {native_name} and {target_name}, and may have grammar or "
                f"spelling mistakes in either.\n\n"
                f"{context_block}"
                f"FIRST, decide a VERDICT for their draft - exactly one word, "
                f"nothing else:\n"
                f"- CLEAN: the draft is already completely correct {target_name} - "
                f"not even a trivial issue. If you would change literally nothing, "
                f"it's CLEAN.\n"
                f"- MINOR: the draft is understandable, natural-enough "
                f"{target_name} that could be sent as-is, but has ONE small "
                f"cosmetic issue - a missing or wrong accent mark, a punctuation "
                f"slip (missing ¿/¡, a wrong comma), or an obvious one-letter "
                f"spelling typo. Nothing about word choice, verb conjugation, or "
                f"sentence structure is wrong - only surface-level polish. If in "
                f"doubt between MINOR and FIX, and the issue is purely visual/"
                f"orthographic, choose MINOR.\n"
                f"- FIX: anything else - a wrong verb form, wrong word choice, "
                f"missing words, unnatural phrasing, structural or grammar "
                f"problems, or the draft is in {native_name}/mixed and needs real "
                f"translation.\n"
                f"{ask_rule}\n"
                f"Write exactly one of {verdict_words}, then immediately "
                f"write this marker: {_COACH_VERDICT_MARKER}\n\n"
                f"THEN, PLAIN TEXT ONLY - no quotes, no labels, no JSON: state "
                f"their intent as a concise {native_name} sentence - if what they "
                f"typed is ALREADY clear, grammatical {native_name}, just copy it "
                f"verbatim; otherwise give the shortest natural {native_name} "
                f"phrasing of what they meant, not a description of their intent. "
                f"If they describe what they want to say (\"I want to say X\", \"how do "
                f"I tell him X\"), the meaning is X itself, as they'd say it to their "
                f"partner - never \"I want to say ...\". "
                f"Nothing else - no preamble, no explanation.\n\n"
                f"Immediately after that, write exactly this marker: "
                f"{_COACH_MEANING_MARKER}\n\n"
                f"THEN, depending on which VERDICT you chose above:\n\n"
                f"IF YOU CHOSE CLEAN: write nothing further at all - stop "
                f"immediately after {_COACH_MEANING_MARKER}.\n\n"
                f"IF YOU CHOSE MINOR: write ONE corrected version of their whole "
                f"draft, fixing ONLY that small issue - plain {target_name} text, "
                f"no quotes, no labels - then immediately write this marker: "
                f"{_COACH_SUGGESTION_MARKER}\n\n"
                f"IF YOU CHOSE FIX: write ONE suggested way to say it - the most "
                f"natural {target_name} a friendly native speaker would use here, one "
                f"sentence or short exchange ready to send as-is - then immediately "
                f"this marker: {_COACH_OPT_MARKERS[0]}\n"
                f"Then that suggestion's whole-sentence {native_name} back-translation "
                f"(so the learner can double check it), then immediately this marker: "
                f"{_COACH_OPT_MARKERS[1]}\n"
                f"The _ES piece MUST be {target_name} and the _EN piece MUST be "
                f"{native_name}.\n\n"
                f"No markdown fences, no explanation, no extra text anywhere "
                f"beyond exactly what's specified above."
            ),
        },
        {"role": "user", "content": draft},
    ]

    try:
        chunks = model_chat_stream(messages, max_tokens=_COACH_STREAM_MAX_TOKENS)

        verdict_iter, verdict_leftover = _run_sequential_fields(chunks, [Field("verdict", _COACH_VERDICT_MARKER)])
        verdict_raw: str | None = None
        for kind, text in verdict_iter:
            if kind == "verdict_complete":
                verdict_raw = text

        if verdict_raw is None:
            # The verdict marker never arrived at all - nothing downstream
            # (not even which fields to expect next) can be trusted.
            raise TranslationUnavailableError("Model reply had no usable verdict")

        verdict = verdict_raw.strip().lower()
        if allow_ask and verdict == "ask":
            yield ("verdict", "ask")
            close = getattr(chunks, "close", None)
            if close is not None:
                close()  # stop generating - nothing after the verdict is needed
            return
        if verdict not in _COACH_VERDICTS:
            # An unrecognized token defaults to the FULL correction flow,
            # never to "looks good" - showing three suggestions for a
            # draft that was secretly fine is harmless; silently telling
            # the learner a genuinely broken draft is fine to send is not.
            logger.warning("coach_draft_stream: unrecognized verdict token %r, defaulting to fix", verdict_raw)
            verdict = "fix"

        yield ("verdict", verdict)

        fields = [Field("meaning", _COACH_MEANING_MARKER)]
        if verdict == "minor":
            fields.append(Field("suggestion", _COACH_SUGGESTION_MARKER))
        elif verdict == "fix":
            for i in range(_COACH_SUGGESTIONS):
                fields.append(Field("option_spanish", _COACH_OPT_MARKERS[2 * i], extra=(i,)))
                fields.append(Field("option_english", _COACH_OPT_MARKERS[2 * i + 1], extra=(i,)))

        rest_iter, _rest_leftover = _run_sequential_fields(chunks, fields, initial_buffer=verdict_leftover[0])
        # Stashes each FIX option's Spanish text, by index, the moment it
        # completes - needed because the Spanish/English language check
        # below (_correct_option_languages) can only run once BOTH halves
        # of an option are known, which is only true once its own
        # option_english_complete arrives.
        pending_option_spanish: dict[int, str] = {}
        for kind, *rest in rest_iter:
            if kind == "meaning_chunk":
                yield ("meaning_chunk", rest[0])
            elif kind == "meaning_complete":
                yield ("meaning_complete", _strip_say_framing(rest[0]))
            elif kind == "suggestion_chunk":
                yield ("suggestion_chunk", rest[0])
            elif kind == "suggestion_complete":
                yield ("suggestion_complete", rest[0].strip())
            elif kind == "option_spanish_chunk":
                yield ("option_spanish_chunk", rest[0], rest[1])
            elif kind == "option_spanish_complete":
                index, text = rest[0], rest[1].strip()
                pending_option_spanish[index] = text
                yield ("option_spanish_complete", index, text)
            elif kind == "option_english_chunk":
                yield ("option_english_chunk", rest[0], rest[1])
            elif kind == "option_english_complete":
                index, english_text = rest[0], rest[1].strip()
                spanish_text = pending_option_spanish.get(index, "")
                corrected_spanish, corrected_english = _correct_option_languages(
                    spanish_text, english_text, native_lang, target_lang
                )
                if corrected_spanish != spanish_text:
                    # A corrective second option_spanish_complete for the
                    # same index - the frontend reducer already just
                    # overwrites that field's state on a repeat event, so
                    # no new frontend handling is needed for this.
                    yield ("option_spanish_complete", index, corrected_spanish)
                yield ("option_english_complete", index, corrected_english)
    except ModelServerUnavailableError as exc:
        raise TranslationUnavailableError(str(exc)) from exc


# The tutor: answers the learner's questions ABOUT a message (what a word
# means, the grammar, "break that down", literal vs idiomatic meaning,
# repetition drills), as a side conversation anchored to that message.
# Read aloud as well as shown, so: short spoken-friendly sentences, English
# with the Spanish in quotes, no markdown.
_TUTOR_MAX_TOKENS = 1600
_TUTOR_HISTORY_MESSAGES = 8
_TUTOR_ABOUT_MARKER = "<<<ABOUT>>>"

# Observed with the old voice coach: the model sometimes echoes a field's
# name into the field ("ANSWER: ..."), which would get read aloud. Stripped
# from the start of each field; the prompt also tells it not to.
_FIELD_LABEL_RE = re.compile(r"^\s*(?:ABOUT|ANSWER)\b\s*[:\-–—]?\s*", re.IGNORECASE)
# How much of the answer to hold back before deciding whether it starts
# with an echoed label - "ANSWER" plus a separator is 8 chars at most.
_ANSWER_HOLDBACK = 8


def _strip_field_label(text: str) -> str:
    return _FIELD_LABEL_RE.sub("", text, count=1)


def tutor_answer_stream(
    question: str,
    messages: list[dict],
    default_number: int | None,
    thread_turns: list[tuple[str, str]],
    native_lang: str,
    target_lang: str,
    draft_turns: list[tuple[str, str]] | None = None,
):
    """Streams the tutor's answer to `question`.

    `messages` is the recent conversation, oldest first, each a dict with
    "number" (1-based, what the model refers to), "role" ("user" |
    "assistant"), "text", and optional "glosses" (a short word-by-word gloss
    string for that message). `default_number` is the message the question
    is most likely about (the open thread, else the partner's latest
    message). `thread_turns` is the side conversation about that message so
    far, as (role, text) with role "learner" | "tutor".

    Yields ("about", number_or_None) first - which message the answer is
    about - then ("answer_chunk", delta) as the answer streams and finally
    ("answer_complete", full_text). Raises TranslationUnavailableError if
    the model can't be reached or produced no answer at all."""
    if not question.strip():
        return

    native_name = _LANGUAGE_NAMES.get(native_lang, native_lang)
    target_name = _LANGUAGE_NAMES.get(target_lang, target_lang)
    recent = messages[-_TUTOR_HISTORY_MESSAGES:]
    numbers = {m["number"] for m in recent}
    listing = "\n".join(
        f"[{m['number']}] {'Learner' if m['role'] == 'user' else 'Partner'}: {m['text']}"
        + (f"\n    (word glosses: {m['glosses']})" if m.get("glosses") else "")
        for m in recent
    )
    thread_lines = "\n".join(
        f"{'Learner' if role == 'learner' else 'Tutor'}: {text}" for role, text in thread_turns[-10:]
    )
    draft_lines = "\n".join(
        f"{'Learner' if role == 'learner' else 'Coach'}: {text}" for role, text in (draft_turns or [])[-8:]
    )
    default_hint = (
        f"If the question doesn't clearly point elsewhere, it's about message [{default_number}]."
        if default_number is not None
        else ""
    )

    prompt = [
        {
            "role": "system",
            "content": (
                f"You are a friendly, sharp {target_name} tutor for a {native_name} "
                f"speaker. They are chatting in {target_name} with a conversation "
                f"partner; this is a side conversation where they ask YOU about the "
                f"language - what a word or phrase means, how the grammar works, a "
                f"breakdown of a message, the literal vs real meaning, how to say "
                f"something, or to practice a word again. Your answers are read "
                f"ALOUD as well as shown, so:\n"
                f"- Answer in {native_name}, with every {target_name} word or phrase "
                f"in quotes, e.g. \"llamas\" literally means \"you call "
                f"yourself\".\n"
                f"- Short, natural spoken sentences. No markdown, no bullet points, "
                f"no headings, no emoji. A few sentences is usually right; a "
                f"breakdown of a whole message can go word group by word group.\n"
                f"- Answer exactly what was asked. If they repeat a word back "
                f"(\"llamas?\"), confirm it and add one useful detail. If they want "
                f"to say something rude or slangy, it's fine to say so lightly and "
                f"give the real options.\n"
                f"- Never write a field name like ABOUT or ANSWER in the answer.\n\n"
                f"The conversation, numbered:\n{listing or '(no messages yet)'}\n\n"
                + (f"Your side conversation about it so far:\n{thread_lines}\n\n" if thread_lines else "")
                + (
                    f"They are also working out what to say next; that discussion so far:\n{draft_lines}\n\n"
                    if draft_lines
                    else ""
                )
                + f"FIRST write only the number of the message the question is about "
                f"(digits only), or 0 if it is about what the learner wants to say "
                f"NEXT or about nothing in particular, then immediately this marker: "
                f"{_TUTOR_ABOUT_MARKER}\n"
                f"{default_hint}\n"
                f"THEN write your answer as plain text, and nothing after it."
            ),
        },
        {"role": "user", "content": question},
    ]

    try:
        chunks = model_chat_stream(prompt, max_tokens=_TUTOR_MAX_TOKENS, model=OPENAI_TUTOR_MODEL)
        about_iter, about_leftover = _run_sequential_fields(chunks, [Field("about", _TUTOR_ABOUT_MARKER)])
        about_raw: str | None = None
        for kind, text in about_iter:
            if kind == "about_complete":
                about_raw = text
        about: int | None = default_number
        if about_raw is not None:
            digits = re.search(r"\d+", about_raw)
            if digits:
                number = int(digits.group())
                about = number if number in numbers else (None if number == 0 else default_number)
        yield ("about", about)

        # Everything after the marker is the answer. If the marker never
        # came, the leftover holds the whole reply - still treat it as the
        # answer rather than losing it.
        pending = about_leftover[0] if about_leftover else ""
        if about_raw is None:
            pending = _strip_field_label(pending)
        answer = ""
        head_done = False

        def emit(text: str):
            nonlocal answer, head_done, pending
            pending += text
            if not head_done:
                if len(pending) < _ANSWER_HOLDBACK:
                    return None
                pending = _strip_field_label(pending.lstrip())
                head_done = True
            out, pending = pending, ""
            answer += out
            return out or None

        first = emit("")
        if first:
            yield ("answer_chunk", first)
        for chunk in chunks:
            out = emit(chunk)
            if out:
                yield ("answer_chunk", out)
        if not head_done:
            out = _strip_field_label(pending.lstrip())
            if out:
                answer += out
                yield ("answer_chunk", out)
        answer = answer.strip()
        if not answer:
            raise TranslationUnavailableError("Model reply had no answer")
        yield ("answer_complete", answer)
    except ModelServerUnavailableError as exc:
        raise TranslationUnavailableError(str(exc)) from exc
