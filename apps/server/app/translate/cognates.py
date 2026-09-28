"""A small curated list of Spanish/English cognates spelled identically in
both languages (e.g. "hotel", "animal", "color") - used by /translate/tag-
input to decide when a word in the learner's input is worth showing
translation candidates in BOTH directions, since spaCy's Spanish-vs-English
classification is binary and can't express "this is a real word in either
language". Not an attempt at exhaustive bilingual detection - just the
common, unambiguous cases most likely to actually come up in a beginner
learner's chat message.
"""

COMMON_ES_EN_COGNATES: frozenset[str] = frozenset(
    {
        "animal", "hotel", "hospital", "color", "doctor", "actor", "general",
        "natural", "normal", "total", "final", "capital", "popular", "similar",
        "rural", "local", "social", "personal", "central", "cultural",
        "terrible", "horrible", "probable", "cordial", "brutal", "formal",
        "informal", "legal", "moral", "oral", "vital", "visual", "verbal",
        "especial", "radio", "taxi", "gas", "plan", "control", "director",
        "motor", "tractor", "chocolate", "piano", "idea", "casino",
    }
)
