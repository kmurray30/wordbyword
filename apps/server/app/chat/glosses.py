"""Word and phrase glosses for a chat message (what a tapped word shows),
matched to character offsets and saved on the message as JSON."""

from __future__ import annotations

import re

from pydantic import ValidationError

from app.languages import get_language
from app.models import ChatMessage
from app.schemas import MessageGlosses, PhraseGloss, WordGloss
from app.translate.lemmatizer import Token, analyze
from app.translate.span_matching import match_spans


# A phrase never really ends on an article/determiner: "seguir una" glossed
# "play along with a joke" is the model reporting fewer words than its own
# gloss covers. Such a phrase is stretched to the word after it (for
# languages with spaces and articles - see Language.determiners).
_LAST_WORD = re.compile(r"([^\W\d_]+'?)\W*$")
_NEXT_WORD = re.compile(r"\s+([^\W\d_]+)")


def complete_phrase_end(text: str, start: int, end: int, language: str = "es") -> int:
    """`end`, moved past the next word for as long as the phrase ends on an
    article/determiner (at most twice)."""
    dangling = get_language(language).determiners
    if not dangling:
        return end
    for _ in range(2):
        last = _LAST_WORD.search(text[start:end])
        if not last or last.group(1).lower() not in dangling:
            break
        nxt = _NEXT_WORD.match(text, end)
        if not nxt:
            break
        end = nxt.end()
    return end


def _complete_phrases(text: str, glosses: MessageGlosses, language: str) -> MessageGlosses:
    phrases = [
        p.model_copy(update={"end": complete_phrase_end(text, p.start, p.end, language)}) for p in glosses.phrases
    ]
    return glosses.model_copy(update={"phrases": phrases})


def build_glosses(
    text: str, words: list[dict[str, str]], phrases: list[dict[str, str]], language: str = "es"
) -> MessageGlosses:
    """Locates gloss_message's words and phrases in `text` (separately: a
    phrase overlaps the words inside it). Anything that can't be found is
    dropped."""
    matched_words = match_spans(text, [{"surface": w["surface"], "gloss": w["gloss"], "note": w.get("grammar", "")} for w in words])
    matched_phrases = match_spans(text, phrases)
    return _complete_phrases(
        text,
        MessageGlosses(
            words=[WordGloss(start=m.start, end=m.end, gloss=m.gloss, grammar=m.note) for m in matched_words if m.gloss],
            phrases=[PhraseGloss(start=m.start, end=m.end, gloss=m.gloss, note=m.note) for m in matched_phrases if m.gloss],
        ),
        language,
    )


def load_glosses(row: ChatMessage) -> MessageGlosses | None:
    if not row.glosses:
        return None
    try:
        glosses = MessageGlosses.model_validate_json(row.glosses)
    except ValidationError:
        return None
    # Saved before phrases were completed - fixed on the way out.
    return _complete_phrases(row.text or "", glosses, getattr(row, "language", "") or "es")


def save_glosses(row: ChatMessage, glosses: MessageGlosses) -> None:
    row.glosses = glosses.model_dump_json()


def token_glosses(tokens: list[Token], glosses: MessageGlosses) -> list[tuple[str, str]]:
    """(gloss, grammar) for each token, from the word covering its start -
    for the per-token rows the word bank and older clients use."""
    result: list[tuple[str, str]] = []
    i = 0
    for tok in tokens:
        while i < len(glosses.words) and glosses.words[i].end <= tok.start:
            i += 1
        if i < len(glosses.words) and glosses.words[i].start <= tok.start < glosses.words[i].end:
            result.append((glosses.words[i].gloss, glosses.words[i].grammar))
        else:
            result.append(("", ""))
    return result


def reply_tokens(text: str, language: str, glosses: MessageGlosses) -> list[Token]:
    """A reply's tokens (its words and punctuation, in order) for the word
    bank and the per-token rows. Spanish: spaCy's, which lemmatize. Other
    languages: the words the LLM glossed - it splits Japanese and Chinese
    into real words, which a plain tokenizer can't - with the plain
    tokenizer filling the gaps (punctuation) and standing in when there are
    no glosses."""
    if get_language(language).code == "es" or not glosses.words:
        return analyze(text, language)
    tokens: list[Token] = []
    cursor = 0
    for w in glosses.words:
        if w.start < cursor:
            continue
        tokens.extend(
            Token(t.surface, t.lemma, t.pos, t.is_target, t.start + cursor, t.end + cursor)
            for t in analyze(text[cursor : w.start], language)
        )
        surface = text[w.start : w.end]
        tokens.append(Token(surface, surface.lower(), "", True, w.start, w.end))
        cursor = w.end
    tokens.extend(
        Token(t.surface, t.lemma, t.pos, t.is_target, t.start + cursor, t.end + cursor)
        for t in analyze(text[cursor:], language)
    )
    return tokens
