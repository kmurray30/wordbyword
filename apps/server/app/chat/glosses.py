"""Word and phrase glosses for a chat message (what a tapped word shows),
matched to character offsets and saved on the message as JSON."""

from __future__ import annotations

from pydantic import ValidationError

from app.models import ChatMessage
from app.schemas import MessageGlosses, PhraseGloss, WordGloss
from app.translate.lemmatizer import Token
from app.translate.span_matching import match_spans


def build_glosses(text: str, words: list[dict[str, str]], phrases: list[dict[str, str]]) -> MessageGlosses:
    """Locates gloss_message's words and phrases in `text` (separately: a
    phrase overlaps the words inside it). Anything that can't be found is
    dropped."""
    matched_words = match_spans(text, [{"surface": w["surface"], "gloss": w["gloss"], "note": w.get("grammar", "")} for w in words])
    matched_phrases = match_spans(text, phrases)
    return MessageGlosses(
        words=[WordGloss(start=m.start, end=m.end, gloss=m.gloss, grammar=m.note) for m in matched_words if m.gloss],
        phrases=[PhraseGloss(start=m.start, end=m.end, gloss=m.gloss, note=m.note) for m in matched_phrases if m.gloss],
    )


def load_glosses(row: ChatMessage) -> MessageGlosses | None:
    if not row.glosses:
        return None
    try:
        return MessageGlosses.model_validate_json(row.glosses)
    except ValidationError:
        return None


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
