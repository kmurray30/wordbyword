"""Contextual vocabulary suggestions (app/vocab/): the pure ranking/bandit
math, the embeddings client, and the full /chat/turn wiring with the
toggle off and on. No network: the chat model, embeddings and summary
calls are all patched."""

from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import httpx
import numpy as np
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.chat.openai_client import ModelServerUnavailableError
from app.chat.prompt_builder import build_vocab_block
from app.config import EMBEDDING_MODEL
from app.db import Base
from app.models import ConversationSummary, VocabSuggestion, WordBankEntry, WordEmbedding
from app.routes.chat import clear_history, take_turn
from app.routes.events import reward_event
from app.schemas import ChatTurnRequest, RewardEventRequest
from app.vocab import bandit, embeddings, suggest
from app.wordbank.scoring import WordState, review_urgency

NOW = datetime(2026, 1, 1)
DIM = 16


def _session():
    engine = create_engine("sqlite:///:memory:")
    from app import models  # noqa: F401  (register tables on Base)

    Base.metadata.create_all(bind=engine)
    return sessionmaker(bind=engine)()


@pytest.fixture(autouse=True)
def _clear_vector_cache():
    embeddings._cache.clear()
    yield
    embeddings._cache.clear()


# --- scoring.review_urgency ---------------------------------------------


def _state(familiarity=0.0, days_since=0.0, interval=1.0):
    return WordState("x", familiarity, NOW - timedelta(days=days_since), interval)


def test_urgency_is_zero_right_after_a_review():
    assert review_urgency(_state(days_since=0.0), NOW) == 0.0


def test_urgency_grows_as_a_word_comes_due():
    assert review_urgency(_state(days_since=0.5), NOW) < review_urgency(_state(days_since=3.0), NOW)


def test_known_word_is_less_urgent_than_unknown_one_at_same_overdueness():
    assert review_urgency(_state(familiarity=0.9, days_since=1.0), NOW) < review_urgency(
        _state(familiarity=0.1, days_since=1.0), NOW
    )


# --- bandit ---------------------------------------------------------------


def _pick_counts(arms, trials=2000):
    rng = np.random.default_rng(0)
    counts: dict[str, int] = {}
    for _ in range(trials):
        (lemma,) = bandit.thompson_pick(arms, 1, rng)
        counts[lemma] = counts.get(lemma, 0) + 1
    return counts


def test_thompson_prefers_words_that_tend_to_land():
    counts = _pick_counts([bandit.BanditArm("fits", 0.5, 8.0, 1.0), bandit.BanditArm("never_fits", 0.5, 1.0, 8.0)])
    assert counts["fits"] > counts.get("never_fits", 0) * 5


def test_thompson_prefers_more_urgent_words():
    counts = _pick_counts([bandit.BanditArm("due", 0.9, 1.0, 1.0), bandit.BanditArm("fresh", 0.0, 1.0, 1.0)])
    assert counts["due"] > counts["fresh"] * 2


def test_thompson_still_explores_the_less_likely_word_sometimes():
    counts = _pick_counts([bandit.BanditArm("fits", 0.5, 4.0, 1.0), bandit.BanditArm("unsure", 0.5, 1.0, 2.0)])
    assert counts.get("unsure", 0) > 0


def test_update_counts_rewards_success_and_penalizes_a_miss():
    assert bandit.update_counts(1.0, 1.0, 1.0) == (2.0, 1.0)
    assert bandit.update_counts(1.0, 1.0, 0.0) == (1.0, 2.0)


def test_update_counts_decays_old_evidence_toward_the_prior():
    alpha, beta = bandit.update_counts(1.0, 21.0, 1.0)
    assert beta < 21.0
    assert alpha == 2.0


# --- rank_relevant / prompt block ----------------------------------------


def _one_hot(i: int) -> np.ndarray:
    v = np.zeros(DIM, dtype=np.float32)
    v[i] = 1.0
    return v


def test_rank_relevant_returns_only_words_that_stand_out():
    lemmas = [f"w{i}" for i in range(12)]
    matrix = np.stack([_one_hot(i) for i in range(12)])
    query = (_one_hot(3) + 0.1 * _one_hot(5)) / np.linalg.norm(_one_hot(3) + 0.1 * _one_hot(5))
    picked = suggest.rank_relevant(query, lemmas, matrix, np.zeros(12), 3, np.random.default_rng(0))
    assert picked == ["w3"]


def test_rank_relevant_needs_a_big_enough_bank():
    lemmas = [f"w{i}" for i in range(5)]
    matrix = np.stack([_one_hot(i) for i in range(5)])
    assert suggest.rank_relevant(_one_hot(0), lemmas, matrix, np.zeros(5), 3, np.random.default_rng(0)) == []


def test_vocab_block_is_empty_with_nothing_to_suggest():
    assert build_vocab_block([], []) == ""


def test_vocab_block_separates_relevant_from_wildcard_and_says_optional():
    block = build_vocab_block([("fútbol", "soccer")], [("ventana", "window")])
    assert "fútbol (soccer)" in block
    assert "Low priority" in block and "ventana (window)" in block
    assert "none of them is completely fine" in block


def test_query_text_includes_summary_and_only_the_latest_messages():
    history = [("user", "uno"), ("assistant", "dos"), ("user", "tres"), ("assistant", "cuatro")]
    text = suggest.build_query_text("Talking about sports.", history, "cinco")
    assert text.startswith("Conversation so far: Talking about sports.")
    assert "uno" not in text
    assert text.endswith("Learner: cinco")


# --- embeddings client -----------------------------------------------------


def _embedding_response(vectors):
    response = MagicMock()
    response.raise_for_status.return_value = None
    response.json.return_value = {"data": [{"index": i, "embedding": v} for i, v in enumerate(vectors)]}
    return response


def test_embed_texts_truncates_and_renormalizes():
    with (
        patch("app.vocab.embeddings.DEEPINFRA_API_TOKEN", "tok"),
        patch("app.vocab.embeddings.EMBEDDING_DIMENSIONS", 2),
        patch("app.vocab.embeddings.httpx.post", return_value=_embedding_response([[3.0, 4.0, 99.0]])) as post,
    ):
        result = embeddings.embed_texts(["hola"])
    assert result.shape == (1, 2)
    assert np.allclose(result[0], [0.6, 0.8])
    assert post.call_args.kwargs["json"]["model"] == EMBEDDING_MODEL


def test_embed_texts_without_token_raises():
    with patch("app.vocab.embeddings.DEEPINFRA_API_TOKEN", ""):
        with pytest.raises(embeddings.EmbeddingUnavailableError):
            embeddings.embed_texts(["hola"])


def test_embed_texts_retries_once_on_transport_error():
    with (
        patch("app.vocab.embeddings.DEEPINFRA_API_TOKEN", "tok"),
        patch(
            "app.vocab.embeddings.httpx.post",
            side_effect=[httpx.ConnectError("stall"), _embedding_response([[1.0, 0.0]])],
        ),
    ):
        assert embeddings.embed_texts(["hola"]).shape[0] == 1


def test_embed_missing_words_skips_closed_class_and_already_embedded(tmp_path):
    session = _session()
    session.add_all(
        [
            WordBankEntry(lemma="perro", pos="NOUN", primary_translation="dog"),
            WordBankEntry(lemma="gato", pos="NOUN", primary_translation="cat"),
            WordBankEntry(lemma="el", pos="DET"),
            WordEmbedding(lemma="gato", model=EMBEDDING_MODEL, vector=_one_hot(0).tobytes()),
        ]
    )
    session.commit()
    with patch("app.vocab.embeddings.embed_texts", return_value=np.stack([_one_hot(1)])) as embed:
        added = embeddings.embed_missing_words(session, suggest.OPEN_CLASS_POS)
    assert added == 1
    assert embed.call_args.args[0] == ["perro: dog"]


# --- /chat/turn wiring -----------------------------------------------------

BANK = ["fútbol", "ventana", "cocina", "playa", "libro", "montaña", "tren", "camisa", "lluvia", "mesa", "perro", "río"]


def _seed_bank(session):
    for i, lemma in enumerate(BANK):
        session.add(WordBankEntry(lemma=lemma, pos="NOUN", primary_translation=f"en-{lemma}", last_reviewed_at=NOW))
        session.add(WordEmbedding(lemma=lemma, model=EMBEDDING_MODEL, vector=_one_hot(i).tobytes()))
    session.commit()


def _run_post_turn_inline(session_factory, session_id, user_message, reply):
    suggest.post_turn(session_factory, session_id, user_message, reply)


def test_toggle_off_adds_no_vocab_block_and_makes_no_embedding_calls():
    session = _session()
    _seed_bank(session)
    with (
        patch("app.routes.chat.CONTEXTUAL_VOCAB_ENABLED", False),
        patch("app.routes.chat.model_chat", return_value="Hola.") as chat,
        patch("app.translate.llm_translate.model_chat", side_effect=ModelServerUnavailableError("down")),
        patch("app.vocab.embeddings.embed_texts", side_effect=AssertionError("must not embed")),
        patch("app.vocab.suggest.schedule_post_turn", side_effect=AssertionError("must not schedule")),
    ):
        take_turn(ChatTurnRequest(session_id="s", message="hola"), session=session)
    assert "Optional vocabulary" not in chat.call_args.args[0][0]["content"]
    assert session.query(VocabSuggestion).count() == 0


def test_toggle_on_offers_relevant_word_rewards_its_use_and_updates_summary():
    session = _session()
    _seed_bank(session)
    with (
        patch("app.routes.chat.CONTEXTUAL_VOCAB_ENABLED", True),
        patch("app.vocab.embeddings.embed_query", return_value=_one_hot(0)),  # "fútbol"-shaped conversation
        patch("app.routes.chat.model_chat", return_value="Me gusta el fútbol.") as chat,
        patch("app.translate.llm_translate.model_chat", side_effect=ModelServerUnavailableError("down")),
        patch("app.vocab.summary.model_chat", return_value="Talking about soccer."),
        patch("app.vocab.embeddings.embed_texts", return_value=np.zeros((0, DIM), dtype=np.float32)),
        patch("app.vocab.suggest.schedule_post_turn", side_effect=_run_post_turn_inline),
    ):
        take_turn(ChatTurnRequest(session_id="s", message="¿Te gustan los deportes?"), session=session)

    system_prompt = chat.call_args.args[0][0]["content"]
    assert "Related to the current conversation: fútbol (en-fútbol)" in system_prompt
    assert "Low priority" in system_prompt

    rows = {row.lemma: row for row in session.query(VocabSuggestion).all()}
    assert rows["fútbol"].kind == "relevant" and rows["fútbol"].used
    futbol = session.query(WordBankEntry).filter_by(lemma="fútbol").one()
    assert futbol.suggest_alpha > 1.0 and futbol.suggest_beta == 1.0
    (wildcard,) = [row for row in rows.values() if row.kind == "wildcard"]
    assert not wildcard.used
    assert session.query(WordBankEntry).filter_by(lemma=wildcard.lemma).one().suggest_beta > 1.0

    assert session.get(ConversationSummary, "s").summary == "Talking about soccer."


def test_toggle_on_with_embeddings_down_still_offers_a_wildcard_and_never_fails_the_turn():
    session = _session()
    _seed_bank(session)
    with (
        patch("app.routes.chat.CONTEXTUAL_VOCAB_ENABLED", True),
        patch("app.vocab.embeddings.embed_query", side_effect=embeddings.EmbeddingUnavailableError("no token")),
        patch("app.routes.chat.model_chat", return_value="Hola.") as chat,
        patch("app.translate.llm_translate.model_chat", side_effect=ModelServerUnavailableError("down")),
        patch("app.vocab.suggest.schedule_post_turn"),
    ):
        result = take_turn(ChatTurnRequest(session_id="s", message="hola"), session=session)
    assert result.text == "Hola."
    system_prompt = chat.call_args.args[0][0]["content"]
    assert "Related to the current conversation" not in system_prompt
    assert "Low priority" in system_prompt


def test_words_offered_or_used_recently_are_on_cooldown():
    session = _session()
    _seed_bank(session)
    common = (
        patch("app.routes.chat.CONTEXTUAL_VOCAB_ENABLED", True),
        patch("app.vocab.embeddings.embed_query", return_value=_one_hot(0)),
        patch("app.translate.llm_translate.model_chat", side_effect=ModelServerUnavailableError("down")),
        patch("app.vocab.suggest.schedule_post_turn"),
    )
    with common[0], common[1], common[2], common[3], patch("app.routes.chat.model_chat", return_value="Me gusta el fútbol."):
        take_turn(ChatTurnRequest(session_id="s", message="deportes"), session=session)
    with common[0], common[1], common[2], common[3], patch("app.routes.chat.model_chat", return_value="Sí.") as chat:
        take_turn(ChatTurnRequest(session_id="s", message="más deportes"), session=session)
    assert "fútbol" not in chat.call_args.args[0][0]["content"].split("Optional vocabulary")[-1]


def test_learner_typing_an_offered_word_rewards_it_once():
    session = _session()
    session.add(WordBankEntry(lemma="playa", pos="NOUN"))
    session.add(VocabSuggestion(session_id="s", assistant_message_id=1, lemma="playa", kind="wildcard"))
    session.commit()

    reward_event(RewardEventRequest(event_type="userTypedSpanishWord", lemma="playa"), session=session)
    reward_event(RewardEventRequest(event_type="userTypedSpanishWord", lemma="playa"), session=session)

    entry = session.query(WordBankEntry).filter_by(lemma="playa").one()
    assert entry.suggest_alpha == 1.0 + bandit.PICKED_UP_BONUS
    assert session.query(VocabSuggestion).one().picked_up


def test_clear_history_also_drops_the_running_summary():
    session = _session()
    session.add(ConversationSummary(session_id="s", summary="old topic"))
    session.commit()
    clear_history("s", session=session)
    assert session.get(ConversationSummary, "s") is None
