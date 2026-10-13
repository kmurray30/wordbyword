"""Every language the learner can study, and everything that varies with
it: names for prompts, which script it's written in (for sanity checks and
tokenizing), speech-to-text settings and prompt examples. The rest of the
backend takes a language code and looks it up here - nothing else should
hardcode "Spanish".

A Mandarin learner chooses Simplified or Traditional characters; that's a
script variant of "zh" ("zh-Hans" / "zh-Hant", see with_script), not a
language of its own - the conversation is the same either way."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

NATIVE_LANGUAGE = "en"
DEFAULT_TARGET = "es"


@dataclass(frozen=True)
class Language:
    code: str
    name: str  # in prompts: "Spanish", "Mandarin Chinese"
    # "latin" | "japanese" | "chinese" | "korean" - see looks_like_target.
    script: str
    # Words are separated by spaces (no for Japanese and Chinese).
    spaced: bool = True
    # Deepgram: model and language. "multi" = code-switching with English
    # (the learner mixes languages mid-sentence); Korean and Mandarin aren't
    # in that set, so they're transcribed as that language alone.
    stt_model: str = "nova-3"
    stt_language: str = "multi"
    # Articles/determiners a phrase can't end on (see
    # app.chat.glosses.complete_phrase_end).
    determiners: frozenset[str] = field(default_factory=frozenset)
    # Prompt examples for gloss_message: how this language's words are
    # glossed, which groups are phrases.
    gloss_examples: str = ""
    # The partner's register/level guidance.
    partner_style: str = ""


_ES_DETERMINERS = frozenset(
    "un una unos unas el la los las lo al del mi mis tu tus su sus nuestro nuestra nuestros nuestras "
    "este esta estos estas ese esa esos esas aquel aquella".split()
)
_FR_DETERMINERS = frozenset(
    "un une des le la les l' du au aux mon ma mes ton ta tes son sa ses notre nos votre vos leur leurs "
    "ce cet cette ces".split()
)

LANGUAGES: dict[str, Language] = {
    "es": Language(
        code="es",
        name="Spanish",
        script="latin",
        determiners=_ES_DETERMINERS,
        gloss_examples=(
            'in "seguir la broma" ("to play along"), seguir is "to follow", not "to play along"; in '
            '"echar de menos", echar is "to throw". A conjugated verb is glossed as conjugated, with the '
            'subject it has here: "puede" in "puede ser" is "it can", "echo" is "I throw", "fueron" is '
            '"they went". Grammar example: "comer (to eat): I, present" or "ir (to go): they, preterite". '
            'Phrase examples: "qué bueno" ("how wonderful"), "tener que" ("to have to"), "me gusta" '
            '("I like"), "a veces" ("sometimes"); a phrase\'s surface includes the noun after an article '
            '("seguir la broma", never "seguir la").'
        ),
    ),
    "fr": Language(
        code="fr",
        name="French",
        script="latin",
        determiners=_FR_DETERMINERS,
        gloss_examples=(
            'in "avoir du mal à" ("to struggle to"), avoir is "to have", not "to struggle"; in "tomber '
            'amoureux", tomber is "to fall". A conjugated verb is glossed as conjugated, with the subject '
            'it has here: "peut" in "il peut" is "he can", "suis" is "I am", "allés" in "ils sont allés" '
            'is "gone" (and "sont" "they are"). Grammar example: "manger (to eat): I, present" or "aller '
            '(to go): they, passé composé". Phrase examples: "il y a" ("there is"), "avoir besoin de" ("to '
            'need"), "ça va" ("it\'s fine / how are you"), "en fait" ("actually"); a phrase\'s surface '
            'includes the noun after an article ("faire la fête", never "faire la"). Treat an elided '
            "word (l', j', d', qu') as its own word."
        ),
    ),
    "ja": Language(
        code="ja",
        name="Japanese",
        script="japanese",
        spaced=False,
        gloss_examples=(
            "Japanese has no spaces: split the message into words yourself, with particles as their own "
            'words - "私は日本語を勉強しています" is 私 / は / 日本語 / を / 勉強 / しています. A particle\'s '
            'gloss says what it does: は "(topic)", を "(object)", に "to / at", の "\'s / of". A word keeps '
            'its own meaning inside an expression: in "気をつける" ("to be careful"), 気 is "spirit, '
            'attention". A conjugated verb or adjective is glossed as conjugated: "行きたい" is "want to '
            'go", "食べました" is "ate", "高くない" is "not expensive". Grammar example: "行く (to go): '
            'want-to form" or "食べる (to eat): past, polite". Phrase examples: "よろしくお願いします" '
            '("nice to meet you / please"), "気をつける" ("to be careful"), "〜てもいい" ("may / it\'s ok '
            'to").'
        ),
        partner_style=(
            "Use the polite です/ます form, and write naturally with kanji and kana as a Japanese "
            "speaker would."
        ),
    ),
    "zh": Language(
        code="zh",
        name="Mandarin Chinese",
        script="chinese",
        spaced=False,
        stt_model="nova-2",
        stt_language="zh-CN",
        gloss_examples=(
            "Chinese has no spaces: split the message into words, not single characters - "
            '"我喜欢学习中文" is 我 / 喜欢 / 学习 / 中文. A particle\'s gloss says what it does: 了 "(completed '
            'action / change)", 吗 "(yes/no question)", 的 "\'s / of". A word keeps its own meaning inside '
            'an expression: in "加油" ("go for it!"), 加 is "to add" and 油 "oil". Grammar is only for '
            'something worth knowing about a word\'s use (a measure word, an aspect particle) - usually '
            'empty. Phrase examples: "马马虎虎" ("so-so"), "加油" ("go for it!"), "没关系" ("it doesn\'t '
            'matter"), "好久不见" ("long time no see").'
        ),
        partner_style="Write in Chinese characters only - no pinyin in your reply.",
    ),
    "ko": Language(
        code="ko",
        name="Korean",
        script="korean",
        stt_model="nova-3",
        stt_language="ko",
        gloss_examples=(
            "Korean words carry particles and endings: gloss each space-separated word as a whole, with "
            'its particle - "저는" is "I (topic)", "학교에" is "to school". A conjugated verb or adjective '
            'is glossed as conjugated: "갔어요" is "went", "먹고 싶어요" is "want to eat" (먹고 "eat '
            'and", 싶어요 "want"), "좋아요" is "it\'s good". Grammar example: "가다 (to go): past, polite" '
            'or "먹다 (to eat): -고 싶다 (want to)". A word keeps its own meaning inside an expression: '
            'in "잘 먹겠습니다" ("thanks for the meal"), 잘 is "well". Phrase examples: "잘 먹겠습니다" '
            '("thanks for the meal"), "괜찮아요" is one word ("it\'s okay"), "수고하세요" ("take care / '
            'keep up the good work").'
        ),
        partner_style="Use the polite 해요 style.",
    ),
}

# Script variants of a language - the learner's choice of characters for
# Mandarin. Prompt names include it so every reply is written that way.
_SCRIPT_VARIANT_NAMES = {
    "zh-Hans": "Mandarin Chinese, written in Simplified characters",
    "zh-Hant": "Mandarin Chinese, written in Traditional characters",
}

_LANGUAGE_NAMES = {"en": "English", **{code: lang.name for code, lang in LANGUAGES.items()}, **_SCRIPT_VARIANT_NAMES}


def base_code(code: str) -> str:
    """ "zh-Hant" -> "zh"; an unknown code -> the default target language."""
    base = (code or "").split("-", 1)[0].lower()
    return base if base in LANGUAGES or base == NATIVE_LANGUAGE else DEFAULT_TARGET


def get_language(code: str | None) -> Language:
    """The language for a code ("zh-Hant" -> Mandarin); unknown -> the default."""
    return LANGUAGES.get(base_code(code or ""), LANGUAGES[DEFAULT_TARGET])


def with_script(code: str | None, script: str | None) -> str:
    """The code prompts are built for: the language, plus the learner's
    Mandarin script ("simplified" | "traditional"; Simplified by default)."""
    lang = get_language(code)
    if lang.code == "zh":
        return "zh-Hant" if (script or "").lower().startswith("trad") else "zh-Hans"
    return lang.code


def language_name(code: str) -> str:
    return _LANGUAGE_NAMES.get(code, _LANGUAGE_NAMES.get(base_code(code), code))


# Characters by script, for sanity checks on model output and on what the
# learner typed.
_HAN = r"㐀-䶿一-鿿豈-﫿"
_KANA = r"぀-ヿㇰ-ㇿｦ-ﾟ"
_HANGUL = r"가-힯ᄀ-ᇿ㄰-㆏"
HAN_RE = re.compile(f"[{_HAN}]")
KANA_RE = re.compile(f"[{_KANA}]")
HANGUL_RE = re.compile(f"[{_HANGUL}]")
CJK_RE = re.compile(f"[{_HAN}{_KANA}{_HANGUL}]")

# Scripts that must never appear in text meant to be in this language (or
# English): a reply that contains them is garbage, not a real answer.
_FOREIGN_SCRIPTS = {
    "latin": CJK_RE,
    "japanese": HANGUL_RE,
    "chinese": re.compile(f"[{_KANA}{_HANGUL}]"),
    "korean": KANA_RE,
}


def has_foreign_script(text: str, code: str) -> bool:
    """`text` (meant to mix this language and English) contains a script
    neither uses - e.g. Chinese characters in a Spanish reply."""
    return bool(_FOREIGN_SCRIPTS[get_language(code).script].search(text))


def script_share(text: str, code: str) -> float | None:
    """For a non-Latin language: the share of letters in `text` written in
    its own script (None for a Latin-script language, or no letters)."""
    lang = get_language(code)
    if lang.script == "latin":
        return None
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return None
    own = {
        "japanese": lambda c: bool(HAN_RE.match(c) or KANA_RE.match(c)),
        "chinese": lambda c: bool(HAN_RE.match(c)),
        "korean": lambda c: bool(HANGUL_RE.match(c) or HAN_RE.match(c)),
    }[lang.script]
    return sum(1 for c in letters if own(c)) / len(letters)
