"""How to read Japanese, Chinese and Korean text: romanization for every
word, kana over kanji (furigana), and Chinese in the learner's choice of
Simplified or Traditional characters.

Deterministic on purpose - never the LLM: pykakasi (Japanese), pypinyin
(Chinese, with its phrase dictionary for characters read more than one
way), opencc (Simplified <-> Traditional), and the Revised Romanization of
Korean below. Results are character offsets into the text, so the frontend
can lay them over the same words it already makes tappable."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from app.languages import HAN_RE, HANGUL_RE, KANA_RE, get_language


@dataclass
class Reading:
    start: int
    end: int
    # Romanization of text[start:end] (romaji, pinyin, Korean RR).
    roman: str
    # Japanese only: the kana reading, for a piece written with kanji.
    kana: str = ""
    # Part of the same word as the reading before it (行 + き of 行き) -
    # romanized together, without a space.
    cont: bool = False


# --- Japanese ---------------------------------------------------------------


@lru_cache(maxsize=1)
def _kakasi():
    import pykakasi

    return pykakasi.kakasi()


def _to_hiragana(text: str) -> str:
    return "".join(chr(ord(c) - 0x60) if "ァ" <= c <= "ヶ" else c for c in text)


def _romaji(kana: str) -> str:
    return "".join(seg["hepburn"] for seg in _kakasi().convert(kana))


# Particles written with one kana but read as another.
_PARTICLE_ROMAJI = {"は": "wa", "へ": "e", "を": "o"}


def _japanese(text: str) -> list[Reading]:
    readings: list[Reading] = []
    cursor = 0
    for seg in _kakasi().convert(text):
        orig, hira = seg["orig"], seg["hira"]
        start = text.find(orig, cursor)
        if start == -1:
            continue
        cursor = start + len(orig)
        if not (HAN_RE.search(orig) or KANA_RE.search(orig)):
            continue
        if orig in _PARTICLE_ROMAJI:
            readings.append(Reading(start, cursor, _PARTICLE_ROMAJI[orig]))
            continue
        if not HAN_RE.search(orig):
            readings.append(Reading(start, cursor, seg["hepburn"]))
            continue
        # Kana around the kanji (送り仮名 like the き of 行き) reads as
        # itself - the furigana goes over the kanji alone.
        plain = _to_hiragana(orig)
        lead = 0
        while lead < min(len(plain), len(hira)) and plain[lead] == hira[lead] and not HAN_RE.match(orig[lead]):
            lead += 1
        tail = 0
        while (
            tail < min(len(plain), len(hira)) - lead
            and plain[-1 - tail] == hira[-1 - tail]
            and not HAN_RE.match(orig[-1 - tail])
        ):
            tail += 1
        core_end = len(orig) - tail
        core_kana = hira[lead : len(hira) - tail]
        if lead:
            readings.append(Reading(start, start + lead, _romaji(orig[:lead])))
        readings.append(Reading(start + lead, start + core_end, _romaji(core_kana), kana=core_kana, cont=bool(lead)))
        if tail:
            readings.append(Reading(start + core_end, cursor, _romaji(orig[core_end:]), cont=True))
    return readings


# --- Chinese ----------------------------------------------------------------


def _chinese(text: str) -> list[Reading]:
    from pypinyin import Style, lazy_pinyin

    # One entry per character (None for anything that isn't a character),
    # read in context so 行 in 银行 is háng and in 行走 xíng.
    syllables = lazy_pinyin(text, style=Style.TONE, errors=lambda chunk: [None] * len(chunk))
    if len(syllables) != len(text):
        return []
    return [Reading(i, i + 1, s) for i, s in enumerate(syllables) if s and HAN_RE.match(text[i])]


@lru_cache(maxsize=2)
def _opencc(config: str):
    import opencc

    return opencc.OpenCC(config)


def chinese_in_script(text: str, script: str) -> str:
    """`text` in Simplified or Traditional characters - same length, so
    offsets still line up (if a conversion would change the length, the
    text is returned as it was)."""
    if not HAN_RE.search(text):
        return text
    converted = _opencc("s2t" if script == "traditional" else "t2s").convert(text)
    return converted if len(converted) == len(text) else text


# --- Korean (Revised Romanization) ------------------------------------------

_INITIALS = ["g", "kk", "n", "d", "tt", "r", "m", "b", "pp", "s", "ss", "", "j", "jj", "ch", "k", "t", "p", "h"]
_MEDIALS = [
    "a", "ae", "ya", "yae", "eo", "e", "yeo", "ye", "o", "wa", "wae", "oe", "yo",
    "u", "wo", "we", "wi", "yu", "eu", "ui", "i",
]  # fmt: skip
# Final consonant at the end of a word / before another consonant.
_FINALS = ["", "k", "k", "k", "n", "n", "n", "t", "l", "k", "m", "l", "l", "l", "p", "l", "m", "p", "p", "t", "t", "ng", "t", "t", "k", "t", "p", "t"]
# Before a vowel (the next syllable starts with silent ㅇ) a final carries
# over: (what stays, what moves to the next syllable).
_LIAISON = {
    1: ("", "g"), 2: ("", "kk"), 3: ("k", "s"), 4: ("", "n"), 5: ("n", "j"), 6: ("", "n"), 7: ("", "d"),
    8: ("", "r"), 9: ("l", "g"), 10: ("l", "m"), 11: ("l", "b"), 12: ("l", "s"), 13: ("l", "t"),
    14: ("l", "p"), 15: ("", "r"), 16: ("", "m"), 17: ("", "b"), 18: ("p", "s"), 19: ("", "s"),
    20: ("", "ss"), 21: ("ng", ""), 22: ("", "j"), 23: ("", "ch"), 24: ("", "k"), 25: ("", "t"),
    26: ("", "p"), 27: ("", ""),
}  # fmt: skip
# Final sounds that turn nasal before ㄴ/ㅁ (국물 gungmul, 입니다 imnida).
_NASAL = {"k": "ng", "t": "n", "p": "m"}


def _korean_word(word: str) -> str:
    """One run of Hangul syllables, romanized - with the sound changes
    between syllables that matter most for reading aloud: a final consonant
    carrying over to a following vowel (한국어 hangugeo), nasalization
    (합니다 hamnida) and ㄹ (설날 seollal, 종로 jongno)."""
    syllables = []
    for c in word:
        index = ord(c) - 0xAC00
        syllables.append((index // 588, (index % 588) // 28, index % 28))
    onsets = [_INITIALS[s[0]] for s in syllables]
    codas = [_FINALS[s[2]] for s in syllables]
    for i in range(len(syllables) - 1):
        final, nxt = syllables[i][2], syllables[i + 1][0]
        if not final:
            continue
        if nxt == 11:  # the next syllable starts with a vowel
            codas[i], onsets[i + 1] = _LIAISON[final]
        elif nxt in (2, 6) and codas[i] in _NASAL:  # before ㄴ/ㅁ
            codas[i] = _NASAL[codas[i]]
        elif nxt == 5:  # before ㄹ
            if codas[i] in ("l", "n"):
                codas[i], onsets[i + 1] = "l", "l"
            else:
                onsets[i + 1] = "n"
                codas[i] = _NASAL.get(codas[i], codas[i])
        elif nxt == 2 and codas[i] == "l":  # ㄹ+ㄴ
            onsets[i + 1] = "l"
    return "".join(onset + _MEDIALS[s[1]] + coda for onset, s, coda in zip(onsets, syllables, codas))


def _korean(text: str) -> list[Reading]:
    readings: list[Reading] = []
    i = 0
    while i < len(text):
        if "가" <= text[i] <= "힣":
            j = i
            while j < len(text) and "가" <= text[j] <= "힣":
                j += 1
            readings.append(Reading(i, j, _korean_word(text[i:j])))
            i = j
        else:
            i += 1
    return readings


def romanize_korean(text: str) -> str:
    return _korean_word(text) if text and all("가" <= c <= "힣" for c in text) else text


# --- Entry point ------------------------------------------------------------


def readings_for(text: str, language: str) -> tuple[str, list[Reading]]:
    """(the text to show, its readings). The shown text differs from `text`
    only for Mandarin in the other script ("zh-Hant" = Traditional);
    readings refer to it (same offsets either way)."""
    lang = get_language(language)
    script = "traditional" if language.lower() == "zh-hant" else "simplified"
    if lang.script == "japanese":
        return text, _japanese(text)
    if lang.script == "chinese":
        # Read from the Simplified form: pypinyin's phrase dictionary (which
        # tells 银行 háng from 行走 xíng) is Simplified.
        return chinese_in_script(text, script), _chinese(chinese_in_script(text, "simplified"))
    if lang.script == "korean" and HANGUL_RE.search(text):
        return text, _korean(text)
    return text, []
