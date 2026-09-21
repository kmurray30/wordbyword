"""Spanish closed-class ("function") words - prepositions, conjunctions,
determiners, pronouns - excluded from the reinforce/new-word candidate
pools that feed the prompt's "use these words" instruction and, more
importantly, `logit_bias` (app/chat/logit_bias.py).

Why: unlike open-class content words (nouns, verbs, adjectives, adverbs),
these have a narrow, fixed grammatical slot. Force-boosting one into a
short reply gives a small model nowhere sensible to put it, so it tends to
get grafted on wherever rather than used naturally - e.g. a reply ending in
a dangling "...por qué?" because "por" got hard-boosted. These words still
get tracked normally in the word bank from ordinary exposure (they show up
in replies anyway, and hover/typing still moves their familiarity) - they
just aren't *targeted* for forced introduction.
"""

# spaCy's Universal Dependencies POS tags for closed-class words. Applied to
# existing word-bank entries, which carry a real POS tag captured the first
# time spaCy tagged them (see app/routes/chat.py's record_exposure call).
EXCLUDED_REINFORCE_POS: frozenset[str] = frozenset({"ADP", "CCONJ", "SCONJ", "DET", "PRON"})

# Candidate new words are drawn from a fixed lemma list (frequency_list.py)
# that has no POS attached, so closed-class entries there are excluded by
# lemma instead of by tag.
SPANISH_FUNCTION_WORD_LEMMAS: frozenset[str] = frozenset(
    {
        # pronouns / demonstratives (with and without the accent spaCy's
        # lemmatizer would actually produce, since this list predates that
        # and was written without diacritics)
        "yo", "tu", "tú", "el", "él", "ella", "nosotros", "vosotros", "ellos", "ellas",
        "este", "esta", "ese", "esa", "aquel", "aquella", "eso", "esto", "aquello",
        "me", "te", "se", "nos", "os", "le", "les", "lo", "la", "los", "las",
        "mi", "mí", "su", "sus", "nuestro", "nuestra", "vuestro", "vuestra",
        "que", "qué", "quien", "quién", "cual", "cuál", "cuanto", "cuánto",
        # determiners / articles
        "un", "una", "unos", "unas",
        # prepositions
        "a", "ante", "bajo", "con", "contra", "de", "desde", "durante", "en",
        "entre", "hacia", "hasta", "mediante", "para", "por", "segun", "según",
        "sin", "sobre", "tras",
        # conjunctions
        "y", "e", "o", "u", "ni", "pero", "mas", "sino", "porque", "aunque",
        "mientras", "cuando", "cuándo", "como", "cómo", "donde", "dónde", "si",
    }
)
