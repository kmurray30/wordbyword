"""Known-Spanish lemma set used as one signal (alongside spaCy morphology) for
flagging tokens the user typed that are probably English, not Spanish."""

from app.translate.dictionary import DICTIONARY_ES_EN
from app.wordbank.frequency_list import FREQUENCY_RANKED_ES

_EXTRA_FUNCTION_WORDS = {
    "el", "la", "los", "las", "un", "una", "unos", "unas", "lo",
    "que", "quien", "cual", "cuyo", "cuando", "cuanto",
    "me", "te", "se", "nos", "os", "le", "les",
    "mi", "tu", "su", "nuestro", "vuestro",
    "de", "en", "a", "al", "del",
}

KNOWN_SPANISH_LEMMAS: set[str] = (
    {w.lower() for w in FREQUENCY_RANKED_ES}
    | {k.lower() for k in DICTIONARY_ES_EN.keys()}
    | _EXTRA_FUNCTION_WORDS
)
