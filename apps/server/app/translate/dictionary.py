"""Small curated ES->EN dictionary with per-sense descriptions, used before
falling back to Argos Translate's word-level MT. Argos gives one plausible
translation with no sense disambiguation; this override exists specifically
for common words where showing the user multiple senses (with a short
description of each) is more useful than a single guess."""

DICTIONARY_ES_EN: dict[str, list[dict[str, str]]] = {
    "ser": [
        {"translation": "to be", "description": "permanent/inherent state (nationality, identity, characteristics)"},
    ],
    "estar": [
        {"translation": "to be", "description": "temporary state, location, or condition"},
    ],
    "tener": [
        {"translation": "to have", "description": "possession"},
        {"translation": "to be (years old)", "description": "used in age expressions: 'tengo 20 años'"},
    ],
    "como": [
        {"translation": "like / as", "description": "comparison"},
        {"translation": "how", "description": "question word, e.g. '¿cómo estás?'"},
        {"translation": "I eat", "description": "1st person present of 'comer'"},
    ],
    "banco": [
        {"translation": "bank", "description": "financial institution"},
        {"translation": "bench", "description": "a seat, e.g. in a park"},
    ],
    "carta": [
        {"translation": "letter", "description": "written correspondence"},
        {"translation": "menu", "description": "restaurant menu"},
        {"translation": "playing card", "description": "in a deck of cards"},
    ],
    "gato": [{"translation": "cat", "description": "the animal"}, {"translation": "jack", "description": "car tool"}],
    "derecho": [
        {"translation": "right", "description": "a legal or moral entitlement"},
        {"translation": "straight ahead", "description": "direction, e.g. 'sigue derecho'"},
        {"translation": "law", "description": "field of study, 'estudia derecho'"},
    ],
    "papa": [
        {"translation": "potato", "description": "the vegetable (Latin America)"},
        {"translation": "Pope", "description": "capitalized: 'el Papa'"},
    ],
    "vela": [
        {"translation": "candle", "description": "wax candle"},
        {"translation": "sail", "description": "boat sail"},
    ],
    "casa": [{"translation": "house / home", "description": "a dwelling"}],
    "tiempo": [
        {"translation": "time", "description": "duration"},
        {"translation": "weather", "description": "'¿qué tiempo hace?'"},
    ],
    "cara": [
        {"translation": "face", "description": "part of the body"},
        {"translation": "expensive", "description": "feminine of 'caro'"},
    ],
    "libre": [{"translation": "free", "description": "not restricted / available"}],
    "hola": [{"translation": "hello", "description": "greeting"}],
    "gracias": [{"translation": "thank you", "description": "expression of gratitude"}],
    "por favor": [{"translation": "please", "description": "polite request"}],
    "bueno": [{"translation": "good", "description": "quality"}],
    "malo": [{"translation": "bad", "description": "quality"}],
    "grande": [{"translation": "big / great", "description": "size or, before a noun, importance"}],
    "pequeño": [{"translation": "small", "description": "size"}],
    "agua": [{"translation": "water", "description": "the liquid"}],
    "comer": [{"translation": "to eat", "description": "verb"}],
    "beber": [{"translation": "to drink", "description": "verb"}],
    "hablar": [{"translation": "to speak / talk", "description": "verb"}],
    "trabajar": [{"translation": "to work", "description": "verb"}],
    "amigo": [{"translation": "friend", "description": "male friend"}],
    "amiga": [{"translation": "friend", "description": "female friend"}],
    "perro": [{"translation": "dog", "description": "the animal"}],
    "libro": [{"translation": "book", "description": "the object"}],
    "escuela": [{"translation": "school", "description": "place of education"}],
    "ciudad": [{"translation": "city", "description": "urban area"}],
    "dinero": [{"translation": "money", "description": "currency"}],
    "amor": [{"translation": "love", "description": "the emotion"}],
    "familia": [{"translation": "family", "description": "relatives"}],
}


def lookup(lemma: str) -> list[dict[str, str]]:
    return DICTIONARY_ES_EN.get(lemma.lower(), [])
