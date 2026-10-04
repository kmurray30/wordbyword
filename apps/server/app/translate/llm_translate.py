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
from app.config import OPENAI_TRANSLATE_MODEL
from app.translate.lemmatizer import analyze

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


_COACH_CORE_MAX_TOKENS = 600
# Raised from 700: PART 1 and PART 2 share ONE generation budget (one HTTP
# call), so a verbose PART 1 (3 full formality variants) could leave too
# little room for PART 2's per-option word-by-word breakdowns - observed
# live, non-deterministically: the same kind of draft got usable
# translations one run and none (truncated/unparseable) the next, purely
# from how much of the shared budget PART 1 happened to use that time.
_COACH_TRANSLATIONS_MAX_TOKENS = 2200
# Raised 1100->1250 to make room for each span's new `literal` field (a
# word-by-word breakdown, non-empty only for a multi-word idiom/phrasal-
# verb group) - most spans leave it empty, but an option with a couple of
# such groups needs the extra headroom across up to 3 options. Raised
# again, 1250->2200: a live failure in tag_draft (same model, same new
# `literal` field - see _TAG_DRAFT_MAX_TOKENS's comment) came back
# completely empty because the model's own reasoning tokens alone
# exhausted a budget in this same range, before any visible output - this
# call is at just as much risk, so it gets the same generous headroom.
_COACH_STREAM_MAX_TOKENS = _COACH_CORE_MAX_TOKENS + _COACH_TRANSLATIONS_MAX_TOKENS
_COACH_HISTORY_TURNS = 8
_VALID_FORMALITIES = {"neutral", "casual", "formal"}
# Written literally into the prompt, and watched for verbatim in the
# streamed reply - delineates PART 1 (the coaching feedback + target_lang
# options themselves, shown the moment it's ready) from PART 2 (each
# option's own native_lang translation + word-by-word breakdown, which
# keeps streaming after PART 1 is already on screen). Unlikely to
# collide with real JSON string content, which is the only thing that
# could make this fire early/never.
_COACH_TRANSLATIONS_MARKER = "<<<TRANSLATIONS>>>"


def _parse_coach_core_json(text: str) -> tuple[str, str, list[tuple[str, str]]]:
    if _looks_garbled(text):
        raise ValueError("garbled reply")
    match = _JSON_OBJECT_RE.search(text)
    if not match:
        raise ValueError(f"no JSON object found in reply: {text!r}")
    data = json.loads(match.group(0))
    meaning = str(data.get("meaning", "")).strip()
    feedback = str(data.get("feedback", "")).strip()
    raw_options = data.get("options", [])
    if not isinstance(raw_options, list):
        raise ValueError(f"'options' wasn't a list: {data!r}")

    options: list[tuple[str, str]] = []
    seen_formalities: set[str] = set()
    for item in raw_options:
        if not isinstance(item, dict):
            continue
        formality = str(item.get("formality", "")).strip().lower()
        spanish = str(item.get("spanish", "")).strip()
        if not spanish or formality in seen_formalities:
            continue
        if formality not in _VALID_FORMALITIES:
            formality = "neutral"
        seen_formalities.add(formality)
        options.append((formality, spanish))
    return meaning, feedback, options


def _parse_coach_translations_json(text: str) -> list[dict]:
    match = _JSON_OBJECT_RE.search(text)
    if not match:
        raise ValueError(f"no JSON object found in reply: {text!r}")
    data = json.loads(match.group(0))
    raw_options = data.get("options", [])
    if not isinstance(raw_options, list):
        raise ValueError(f"'options' wasn't a list: {data!r}")

    result: list[dict] = []
    for item in raw_options:
        if not isinstance(item, dict):
            result.append({"english": "", "spans": []})
            continue
        english = str(item.get("english", "")).strip()
        raw_spans = item.get("spans", [])
        spans = _coerce_spans(raw_spans) if isinstance(raw_spans, list) else []
        result.append({"english": english, "spans": spans})
    return result


def coach_draft_stream(
    draft: str,
    history: list[tuple[str, str]],
    native_lang: str,
    target_lang: str,
):
    """For a message the learner is still drafting (not yet sent): infers
    what they're trying to say, gives brief feedback on how apt that
    attempt was, and suggests the ideal target_lang phrasing - at a couple
    of formality levels - given the recent conversation's tone and context.
    ONE streamed LLM call, in two ordered parts delineated by
    _COACH_TRANSLATIONS_MARKER in the raw reply:

    PART 1 (the feedback + target_lang options themselves) is yielded as
    ("core", {"meaning": ..., "feedback": ..., "options": [(formality,
    target_lang text), ...]}) the moment it's complete, so the UI can show
    it without waiting on PART 2 at all.

    PART 2 (each PART-1 option's own native_lang translation + word-by-word
    breakdown - the same surface/gloss/note/translation/alternate_gloss
    shape tag_draft's spans use, matchable back to real offsets the same
    way via app.translate.span_matching - so picking an option can seed
    the chat input's lazy gloss cache immediately, with no extra fetch)
    keeps streaming after that and is yielded separately, as
    ("translations", [{"english": ..., "spans": [...]}, ...]), lined up
    with PART 1's options by list position.

    Speed: when the draft is ALREADY correct and needs no change, the
    prompt tells the model to collapse PART 1 to a single "neutral" option
    (an unchanged copy of the draft, empty `feedback`) instead of 3 full
    formality variants, and to skip PART 2's breakdown entirely
    ({"options": []}) - nothing there needs double-checking. A much
    shorter completion for the common "looks good" case, not a separate
    pre-flight call - callers tell the two cases apart the same way either
    way (empty `feedback` - see ChatInput.tsx's coachClean).

    Unlike every other LLM-backed helper here, this does NOT retry on a
    garbled/unparseable reply - PART 1 may already have been yielded (and
    acted on by the caller) by the time PART 2 turns out unusable, so
    there's nothing to cleanly retry. Raises TranslationUnavailableError
    only if PART 1 itself never parses into anything usable; PART 2
    failing/missing just means the generator yields one event instead of
    two - a caller that only consumes "core" never notices."""
    if not draft.strip():
        return

    native_name = _LANGUAGE_NAMES.get(native_lang, native_lang)
    target_name = _LANGUAGE_NAMES.get(target_lang, target_lang)

    recent = history[-_COACH_HISTORY_TURNS:]
    transcript = "\n".join(f"{'Learner' if role == 'user' else 'Partner'}: {text}" for role, text in recent)
    context_block = f"The conversation so far:\n{transcript}\n\n" if transcript else ""

    messages = [
        {
            "role": "system",
            "content": (
                f"You are a warm, encouraging writing coach for someone learning "
                f"{target_name}, whose native language is {native_name}. They are "
                f"drafting their NEXT message in the conversation below - it may "
                f"mix {native_name} and {target_name}, and may have grammar or "
                f"spelling mistakes in either.\n\n"
                f"Respond in TWO parts, in this exact order, with nothing else.\n\n"
                f"PART 1 - a single JSON object: (1) state their intent as a "
                f"concise {native_name} sentence - if what they typed is ALREADY "
                f"clear, grammatical {native_name}, just copy it verbatim; "
                f"otherwise give the shortest natural {native_name} phrasing of "
                f"what they meant, not a description of their intent; (2) give "
                f"ONE short, encouraging sentence of feedback, in {native_name}, "
                f"on how close their attempt already is to that meaning (empty "
                f"string if it was basically already right); (3) suggest the "
                f"ideal way to actually say it in {target_name} given the "
                f"conversation's tone and context so far.\n\n"
                f"IMPORTANT for speed: if what they typed is ALREADY correct, "
                f"natural {target_name} needing no change at all, don't generate "
                f"3 formality variants - set `feedback` to an empty string and "
                f"`options` to EXACTLY ONE entry, {{\"formality\": \"neutral\", "
                f'"spanish": "<their own draft, copied exactly, unchanged>"}}, '
                f"and stop there - nothing else to suggest. Only when it DOES "
                f"need correction, suggest the ideal phrasing at 3 formality "
                f"levels as described below.\n\n"
                f"{context_block}"
                f"PART 1's JSON shape, exactly:\n"
                f'{{"meaning": "<their intent as a concise {native_name} '
                f"sentence - verbatim copy of their own text if it's already "
                f'clear, grammatical {native_name}>", "feedback": "<in '
                f'{native_name}: one short, encouraging sentence on how apt '
                f'their attempt was, or an empty string if it was already '
                f'right>", "options": [{{"formality": "neutral", '
                f'"spanish": "<the ideal {target_name} phrasing>"}}, {{"formality": '
                f'"casual", "spanish": "<a more casual/informal way to say it>"}}, '
                f'{{"formality": "formal", "spanish": "<a more formal/polite way '
                f'to say it>"}}]}} - or, when already correct (see above), just '
                f'{{"meaning": "...", "feedback": "", "options": [{{"formality": '
                f'"neutral", "spanish": "<their own draft, unchanged>"}}]}}\n\n'
                f"Keep every {target_name} option to one natural sentence or "
                f"short exchange, ready to send in chat as-is - not a lecture, "
                f"and don't repeat the same phrasing across formality levels if "
                f"you can genuinely vary it. If a formality distinction doesn't "
                f"make sense for this particular message, it's fine for two "
                f"options to be identical.\n\n"
                f"Immediately after PART 1's JSON, on its own, write exactly "
                f"this marker: {_COACH_TRANSLATIONS_MARKER}\n\n"
                f"Then PART 2 - a single JSON object translating EACH of PART "
                f"1's options, IN THE SAME ORDER, back for the learner to "
                f"double check: for every option, its whole-phrase "
                f"{native_name} translation, and a word-by-word breakdown "
                f"(skip pure punctuation; copy each `surface` EXACTLY as it "
                f"appears in that option's own {target_name} text). For a "
                f"MULTI-WORD span in that breakdown only, also give a short "
                f"`literal` word-by-word breakdown in {native_name} of how the "
                f"group's individual words combine, e.g. for a {target_name} "
                f'idiom like "tener en cuenta" meaning "take into account", '
                f'`literal` could be "tener (to have) + en (in) + cuenta '
                f'(account)" - leave it an empty string for a single-word span. '
                f"EXCEPTION, also for speed: if PART 1 had exactly the one "
                f"\"already correct\" option described above, skip its "
                f"breakdown entirely and just write {{\"options\": []}} for "
                f"PART 2 - nothing there needs double-checking. PART 2's JSON "
                f"shape, exactly:\n"
                f'{{"options": [{{"english": "<whole-phrase {native_name} '
                f'translation of this option>", "spans": [{{"surface": "<exact '
                f"word/word-group from this option's {target_name} text>\", "
                f'"gloss": "<its meaning here, in {native_name}>", "note": "<in '
                f'{native_name}: short disambiguation note, or empty>", '
                f'"literal": "<word-by-word breakdown for a multi-word span '
                f'only - else empty>"}}, ...]}}, ...]}}\n\n'
                f"No markdown fences, no explanation outside the two JSON "
                f"objects and the marker between them."
            ),
        },
        {"role": "user", "content": draft},
    ]

    buffer = ""
    core_yielded = False
    try:
        for chunk in model_chat_stream(messages, max_tokens=_COACH_STREAM_MAX_TOKENS):
            buffer += chunk
            if core_yielded:
                continue
            idx = buffer.find(_COACH_TRANSLATIONS_MARKER)
            if idx == -1:
                continue
            meaning, feedback, options = _parse_coach_core_json(buffer[:idx])
            if not options:
                raise ValueError("no usable options in reply")
            core_yielded = True
            yield ("core", {"meaning": meaning, "feedback": feedback, "options": options})
            buffer = buffer[idx + len(_COACH_TRANSLATIONS_MARKER) :]
    except ModelServerUnavailableError as exc:
        raise TranslationUnavailableError(str(exc)) from exc

    if not core_yielded:
        # The marker never arrived - under token pressure the model may
        # have skipped PART 2 (or run out of budget) but the whole buffer
        # can still be a usable PART 1 on its own.
        try:
            meaning, feedback, options = _parse_coach_core_json(buffer)
        except (ValueError, json.JSONDecodeError, KeyError) as exc:
            raise TranslationUnavailableError(f"Model reply wasn't usable JSON: {buffer!r} ({exc})") from exc
        if not options:
            raise TranslationUnavailableError(f"Model reply had no usable options: {buffer!r}")
        yield ("core", {"meaning": meaning, "feedback": feedback, "options": options})
        return

    try:
        translations = _parse_coach_translations_json(buffer)
    except (ValueError, json.JSONDecodeError, KeyError):
        return  # PART 2 missing/garbled - PART 1 alone is still useful
    yield ("translations", translations)
