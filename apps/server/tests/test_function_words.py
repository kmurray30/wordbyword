from app.wordbank.frequency_list import FREQUENCY_RANKED_ES, NEW_WORD_CANDIDATES_ES
from app.wordbank.function_words import EXCLUDED_REINFORCE_POS, SPANISH_FUNCTION_WORD_LEMMAS


def test_known_culprits_are_excluded_lemmas():
    # These are the specific words behind the reported "...por qué?"
    # artifact - a preposition and an interrogative forced in via logit_bias.
    for lemma in ["por", "que", "qué", "el", "y", "pero"]:
        assert lemma in SPANISH_FUNCTION_WORD_LEMMAS


def test_content_words_are_not_excluded():
    for lemma in ["hablar", "comer", "casa", "perro", "importante"]:
        assert lemma not in SPANISH_FUNCTION_WORD_LEMMAS


def test_new_word_candidates_exclude_all_function_word_lemmas():
    assert not (set(NEW_WORD_CANDIDATES_ES) & SPANISH_FUNCTION_WORD_LEMMAS)


def test_new_word_candidates_still_has_plenty_of_content_words():
    # Sanity check that filtering didn't gut the pool.
    assert len(NEW_WORD_CANDIDATES_ES) > len(FREQUENCY_RANKED_ES) * 0.7


def test_excluded_reinforce_pos_covers_closed_classes():
    assert {"ADP", "CCONJ", "SCONJ", "DET", "PRON"} <= EXCLUDED_REINFORCE_POS


def test_excluded_reinforce_pos_does_not_cover_content_classes():
    assert not ({"NOUN", "VERB", "ADJ"} & EXCLUDED_REINFORCE_POS)
