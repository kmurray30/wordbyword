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
from app.models import ConversationSummary, SavedWord, VocabSuggestion, WordBankEntry, WordEmbedding
from app.routes.chat import clear_history, get_history, take_turn
from app.routes.events import reward_event
from app.schemas import ChatTurnRequest, RewardEventRequest
from app.vocab import bandit, embeddings, suggest
from app.wordbank.scoring import WordState, review_urgency

NOW = datetime(2026, 1, 1)
DIM = 64


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


def _score(query, n, k=3, cooldown=frozenset(), urgencies=None):
    lemmas = [f"w{i}" for i in range(n)]
    matrix = np.stack([_one_hot(i) for i in range(n)])
    urg = np.zeros(n) if urgencies is None else urgencies
    return suggest.score_candidates(query, lemmas, matrix, urg, set(cooldown), k, np.random.default_rng(0))


def test_score_candidates_offers_only_words_that_stand_out():
    query = _one_hot(3) + 0.1 * _one_hot(5)
    query = query / np.linalg.norm(query)
    offered, info = _score(query, 12)
    assert offered == ["w3"]
    assert info["w3"][2] == "offered"
    assert info["w3"][1] > info["w5"][1] > info["w0"][1]  # z-scores follow similarity
    assert info["w0"][2] == "below"


def _noisy_pool(n, relevant, rng):
    """`relevant` words close to the query, the rest unrelated noise."""
    # The query has a little noise of its own, so unrelated words don't all
    # score exactly 0 (real embeddings never tie like that).
    query = rng.normal(0, 0.1, DIM).astype(np.float32)
    query[0] = 1.0
    query /= np.linalg.norm(query)
    rows = []
    for i in range(n):
        v = rng.normal(0, 0.1, DIM).astype(np.float32)
        v[0] = 1.0 if i < relevant else 0.0
        rows.append(v / np.linalg.norm(v))
    return query, [f"w{i}" for i in range(n)], np.stack(rows)


def test_a_big_cluster_of_relevant_words_still_clears_the_cutoff_in_a_small_pool():
    # 6 of 20 words are on-topic. A plain mean/std z-score tops out near
    # 1.5 here (the cluster inflates both) and would drop every one of them.
    query, lemmas, matrix = _noisy_pool(20, 6, np.random.default_rng(1))
    offered, info = suggest.score_candidates(
        query, lemmas, matrix, np.zeros(20), set(), 3, np.random.default_rng(0)
    )
    assert len(offered) == 3 and all(int(lemma[1:]) < 6 for lemma in offered)
    assert all(info[f"w{i}"][2] in ("offered", "passed") for i in range(6))
    assert all(info[f"w{i}"][2] == "below" for i in range(6, 20))


def test_noise_alone_does_not_flood_the_offer_list():
    # Nothing relevant: random vectors. With the default cutoff almost none
    # should clear it (a looser one lets ~3 in 100 through - and up to 3 get
    # offered per turn, all mislabeled "related to the conversation").
    cleared = 0
    for seed in range(100):
        rng = np.random.default_rng(seed)
        query, lemmas, matrix = _noisy_pool(100, 0, rng)
        query = rng.normal(0, 1, DIM).astype(np.float32)
        query /= np.linalg.norm(query)
        _, info = suggest.score_candidates(query, lemmas, matrix, np.zeros(100), set(), 3, rng)
        cleared += sum(1 for _, _, status in info.values() if status != "below")
    assert cleared / 100 < 1.0  # of 100 words, on average under one clears the cutoff


def test_score_candidates_reports_the_ones_cut_by_the_limit_and_by_cooldown():
    # Three equally-relevant words, room for one, one of them on cooldown.
    query = (_one_hot(1) + _one_hot(2) + _one_hot(3)) / np.sqrt(3)
    offered, info = _score(query, 40, k=1, cooldown={"w1"})
    assert len(offered) == 1 and offered[0] in {"w2", "w3"}
    assert info["w1"][2] == "cooldown"
    assert {info["w2"][2], info["w3"][2]} == {"offered", "passed"}


def test_score_candidates_prefers_the_word_due_for_review():
    query = (_one_hot(1) + _one_hot(2)) / np.sqrt(2)
    urgencies = np.zeros(40)
    urgencies[2] = 1.0
    offered, _ = _score(query, 40, k=1, urgencies=urgencies)
    assert offered == ["w2"]


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


def test_embed_missing_skips_already_embedded_and_is_per_language():
    session = _session()
    session.add(WordEmbedding(language="es", lemma="gato", model=EMBEDDING_MODEL, vector=_one_hot(0).tobytes()))
    session.commit()
    words = [("perro", "dog"), ("gato", "cat")]
    with patch("app.vocab.embeddings.embed_texts", return_value=np.stack([_one_hot(1)])) as embed:
        added = embeddings.embed_missing(session, "es", words)
        # "gato" in French is a different word that still needs embedding.
        added_fr = embeddings.embed_missing(session, "fr", [("gato", "cat")])
    assert added == 1 and added_fr == 1
    assert embed.call_args_list[0].args[0] == ["perro: dog"]


def test_remember_vectors_keeps_vectors_in_memory_only_until_persisted():
    session = _session()
    with patch("app.vocab.embeddings.embed_texts", return_value=np.stack([_one_hot(2)])) as embed:
        embeddings.remember_vectors("es", [("playa", "beach")])
        assert session.query(WordEmbedding).count() == 0
        assert "playa" in embeddings.load_vectors(session, "es", ["playa"])
        # Persisting it reuses the cached vector instead of calling the API again.
        assert embeddings.embed_missing(session, "es", [("playa", "beach")]) == 1
    assert embed.call_count == 1
    assert session.query(WordEmbedding).filter_by(lemma="playa").count() == 1


# --- /chat/turn wiring -----------------------------------------------------

BANK = ["fútbol", "ventana", "cocina", "playa", "libro", "montaña", "tren", "camisa", "lluvia", "mesa", "perro", "río"]


def _seed_bank(session, language="es", bank=BANK):
    for i, lemma in enumerate(bank):
        session.add(
            WordBankEntry(
                language=language, lemma=lemma, pos="NOUN", primary_translation=f"en-{lemma}", last_reviewed_at=datetime.utcnow()
            )
        )
        session.add(WordEmbedding(language=language, lemma=lemma, model=EMBEDDING_MODEL, vector=_one_hot(i).tobytes()))
    session.commit()


def _run_post_turn_inline(session_factory, session_id, language, user_message, reply):
    suggest.post_turn(session_factory, session_id, language, user_message, reply)


def _turn(session, reply="Hola.", message="hola", session_id="s", language="es", query=_one_hot(0), post_turn=None):
    """One /chat/turn with contextual vocab on and every model call patched;
    returns (response, the partner model's system prompt)."""
    with (
        patch("app.routes.chat.CONTEXTUAL_VOCAB_ENABLED", True),
        patch("app.vocab.embeddings.embed_query", return_value=query),
        patch("app.routes.chat.model_chat", return_value=reply) as chat,
        patch("app.translate.llm_translate.model_chat", side_effect=ModelServerUnavailableError("down")),
        patch("app.vocab.suggest.schedule_post_turn", side_effect=post_turn or (lambda *a, **k: None)),
    ):
        result = take_turn(ChatTurnRequest(session_id=session_id, message=message, language=language), session=session)
    return result, chat.call_args.args[0][0]["content"]


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
        result = take_turn(ChatTurnRequest(session_id="s", message="hola"), session=session)
    assert "Optional vocabulary" not in chat.call_args.args[0][0]["content"]
    assert session.query(VocabSuggestion).count() == 0
    assert result.vocab_debug is None


def test_toggle_on_offers_relevant_word_rewards_its_use_and_updates_summary():
    session = _session()
    _seed_bank(session)
    with (
        patch("app.vocab.summary.model_chat", return_value="Talking about soccer."),
        patch("app.vocab.embeddings.embed_texts", return_value=np.zeros((0, DIM), dtype=np.float32)),
    ):
        _, system_prompt = _turn(
            session, reply="Me gusta el fútbol.", message="¿Te gustan los deportes?", post_turn=_run_post_turn_inline
        )

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
    with patch("app.vocab.embeddings.embed_query", side_effect=embeddings.EmbeddingUnavailableError("no token")):
        with (
            patch("app.routes.chat.CONTEXTUAL_VOCAB_ENABLED", True),
            patch("app.routes.chat.model_chat", return_value="Hola.") as chat,
            patch("app.translate.llm_translate.model_chat", side_effect=ModelServerUnavailableError("down")),
            patch("app.vocab.suggest.schedule_post_turn"),
        ):
            result = take_turn(ChatTurnRequest(session_id="s", message="hola"), session=session)
    assert result.text == "Hola."
    system_prompt = chat.call_args.args[0][0]["content"]
    assert "Related to the current conversation" not in system_prompt
    assert "Low priority" in system_prompt
    # The debug view says why there are no search results.
    assert result.vocab_debug.searched is False
    assert "no token" in result.vocab_debug.error
    assert len(result.vocab_debug.wildcards) == 1


def test_words_offered_or_used_recently_are_on_cooldown():
    session = _session()
    _seed_bank(session)
    _turn(session, reply="Me gusta el fútbol.", message="deportes")
    result, system_prompt = _turn(session, reply="Sí.", message="más deportes")
    assert "fútbol" not in system_prompt.split("Optional vocabulary")[-1]
    row = next(w for w in result.vocab_debug.search if w.lemma == "fútbol")
    assert row.status == "cooldown" and row.z >= result.vocab_debug.cutoff_z


def test_debug_payload_shows_scores_cutoff_and_use_and_is_saved_for_history():
    session = _session()
    _seed_bank(session)
    result, _ = _turn(session, reply="Me gusta el fútbol.", message="¿Te gustan los deportes?")

    debug = result.vocab_debug
    assert debug.searched and debug.error == ""
    futbol = debug.search[0]
    assert (futbol.lemma, futbol.status, futbol.used) == ("fútbol", "offered", True)
    assert futbol.sim > 0.99 and futbol.z >= debug.cutoff_z and futbol.translation == "en-fútbol"
    assert all(w.status == "below" for w in debug.search[1:])
    assert [w.sim for w in debug.search] == sorted((w.sim for w in debug.search), reverse=True)
    (wildcard,) = debug.wildcards
    assert wildcard.status == "wildcard" and wildcard.sim is None and not wildcard.used

    from_history = next(m for m in get_history("s", session=session).messages if m.role == "assistant")
    assert from_history.vocab_debug == debug
    assert next(m for m in get_history("s", session=session).messages if m.role == "user").vocab_debug is None


def test_saved_words_join_the_pool_even_with_no_word_bank_entry():
    session = _session()
    _seed_bank(session)
    session.add(SavedWord(session_id="s", word="Cerveza", word_key="cerveza", gloss="beer"))
    session.commit()
    # The saved word was never embedded - it's embedded on the spot, and the
    # conversation is shaped exactly like it.
    with patch("app.vocab.embeddings.embed_texts", return_value=np.stack([_one_hot(20)])) as embed:
        result, system_prompt = _turn(session, reply="Una cerveza, por favor.", query=_one_hot(20))

    assert embed.call_args.args[0] == ["cerveza: beer"]
    assert "cerveza (beer)" in system_prompt
    row = result.vocab_debug.search[0]
    assert (row.lemma, row.status, row.saved, row.used) == ("cerveza", "offered", True, True)
    # Its bandit counts now have somewhere to live.
    entry = session.query(WordBankEntry).filter_by(lemma="cerveza").one()
    assert entry.primary_translation == "beer" and entry.suggest_alpha > 1.0


def test_saved_words_are_preferred_wildcards():
    session = _session()
    _seed_bank(session)
    session.add(SavedWord(session_id="s", word="tren", word_key="tren", gloss="train"))
    session.commit()
    suggestions = [suggest.suggest_for_turn(session, "s", "es", [], "hola") for _ in range(60)]
    saved_hits = sum(1 for sug in suggestions for w in sug.wildcard if w.lemma == "tren")
    # One word of 12 - but saved, so it should win far more often than 1 in 12.
    assert saved_hits > 60 * 0.4


def test_another_languages_words_are_never_offered():
    session = _session()
    _seed_bank(session, language="fr", bank=["chat", "chien", "maison", "table", "livre", "rue", "pain", "eau", "vin", "lait", "sel", "riz"])
    _seed_bank(session, language="es")
    result, system_prompt = _turn(session, session_id="fr-session", language="fr", query=_one_hot(0))
    offered = {w.lemma for w in result.vocab_debug.search} | {w.lemma for w in result.vocab_debug.wildcards}
    assert offered and offered <= {"chat", "chien", "maison", "table", "livre", "rue", "pain", "eau", "vin", "lait", "sel", "riz"}
    assert "fútbol" not in system_prompt


def test_closed_class_and_name_words_are_not_candidates_but_untagged_words_are():
    session = _session()
    session.add_all(
        [
            WordBankEntry(language="es", lemma="el", pos="DET"),
            WordBankEntry(language="es", lemma="ana", pos="PROPN"),
            WordBankEntry(language="es", lemma="casa", pos="NOUN"),
            WordBankEntry(language="ja", lemma="猫", pos="", primary_translation="cat"),
            WordBankEntry(language="es", lemma="!!", pos=""),
        ]
    )
    session.commit()
    assert {c.lemma for c in suggest._candidates(session, "s", "es")} == {"casa"}
    assert {c.lemma for c in suggest._candidates(session, "s", "ja")} == {"猫"}


def test_learner_typing_an_offered_word_rewards_it_once():
    session = _session()
    session.add(WordBankEntry(lemma="playa", pos="NOUN"))
    session.add(VocabSuggestion(session_id="s", assistant_message_id=1, lemma="playa", kind="wildcard"))
    session.commit()

    reward_event(RewardEventRequest(event_type="userTypedTargetWord", lemma="playa"), session=session)
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
