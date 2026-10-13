from types import SimpleNamespace

from app.chat.glosses import build_glosses, complete_phrase_end, load_glosses
from app.schemas import MessageGlosses, PhraseGloss


def _span(text, phrase):
    start = text.index(phrase)
    return start, start + len(phrase)


def test_a_phrase_ending_on_an_article_takes_the_next_word():
    text = "¿Quieres seguir una broma conmigo?"
    start, end = _span(text, "seguir una")
    assert text[start : complete_phrase_end(text, start, end)] == "seguir una broma"


def test_complete_phrases_are_left_alone():
    text = "Hay que tener en cuenta eso. Me gusta la playa."
    for phrase in ("tener en cuenta", "Me gusta", "Hay que"):
        start, end = _span(text, phrase)
        assert complete_phrase_end(text, start, end) == end


def test_nothing_after_the_article_leaves_it():
    text = "Vamos a seguir la"
    start, end = _span(text, "seguir la")
    assert complete_phrase_end(text, start, end) == end


def test_build_glosses_completes_a_truncated_phrase():
    text = "Te gusta seguir la broma, ¿no?"
    g = build_glosses(text, [], [{"surface": "seguir la", "gloss": "play along with the joke", "note": ""}])
    assert [text[p.start : p.end] for p in g.phrases] == ["seguir la broma"]


def test_saved_glosses_are_completed_when_loaded():
    text = "¿Quieres seguir una broma conmigo?"
    start, end = _span(text, "seguir una")
    saved = MessageGlosses(words=[], phrases=[PhraseGloss(start=start, end=end, gloss="play along with a joke", note="")])
    row = SimpleNamespace(text=text, glosses=saved.model_dump_json())
    loaded = load_glosses(row)
    assert text[loaded.phrases[0].start : loaded.phrases[0].end] == "seguir una broma"
