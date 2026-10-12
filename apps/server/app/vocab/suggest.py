"""Contextual vocabulary suggestions: which word-bank words the system
prompt offers the model this turn, and the bookkeeping after it replies.

Per turn (only when CONTEXTUAL_VOCAB_ENABLED):
  1. suggest_for_turn - before the model call, read-only: rank the bank by
     embedding similarity to the conversation (running summary + latest
     messages + the new user message), weighted toward words that are due
     for review; then let the bandit add a wildcard or two.
  2. record_outcome - inside the turn's own short write transaction: log
     what was offered and reward each word by whether the reply used it.
  3. schedule_post_turn - after the turn commits, in a background thread:
     refresh the running summary and embed any new word-bank lemmas, so
     neither LLM/embedding call sits in front of the reply.

Every step degrades to "no suggestions" on failure - this feature never
fails a turn.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.chat.openai_client import ModelServerUnavailableError
from app.config import RELEVANCE_MIN_Z, RELEVANT_WORDS_PER_TURN, SUGGEST_COOLDOWN_TURNS, WILDCARD_WORDS_PER_TURN
from app.models import ChatMessage, MessageToken, VocabSuggestion, WordBankEntry
from app.vocab import bandit, embeddings, summary
from app.wordbank.scoring import review_urgency
from app.wordbank.store import _to_state

logger = logging.getLogger(__name__)

# spaCy UPOS tags worth suggesting. Closed-class words (articles,
# prepositions, pronouns, conjunctions...) show up in every reply anyway.
OPEN_CLASS_POS = frozenset({"NOUN", "VERB", "ADJ", "ADV"})

# Below this many embedded candidates, "relevant relative to the rest of
# the bank" isn't meaningful yet - skip the relevance side entirely.
_MIN_BANK_FOR_RELEVANCE = 10

# How many of the latest messages go into the relevance query verbatim
# (the running summary covers everything before them).
_QUERY_RECENT_MESSAGES = 3

# How long after being offered a word still counts as "the suggestion
# paid off" if the learner types it themselves.
_PICK_UP_WINDOW = timedelta(minutes=30)


@dataclass
class SuggestedWord:
    lemma: str
    translation: str
    kind: str  # "relevant" | "wildcard"


@dataclass
class TurnSuggestions:
    relevant: list[SuggestedWord] = field(default_factory=list)
    wildcard: list[SuggestedWord] = field(default_factory=list)

    @property
    def all(self) -> list[SuggestedWord]:
        return self.relevant + self.wildcard


def build_query_text(conversation_summary: str, history: list[tuple[str, str]], user_message: str) -> str:
    parts = []
    if conversation_summary:
        parts.append(f"Conversation so far: {conversation_summary}")
    for role, text in history[-_QUERY_RECENT_MESSAGES:]:
        parts.append(f"{'Learner' if role == 'user' else 'Partner'}: {text}")
    parts.append(f"Learner: {user_message}")
    return "\n".join(parts)


def rank_relevant(
    query: np.ndarray,
    lemmas: list[str],
    matrix: np.ndarray,
    urgencies: np.ndarray,
    k: int,
    rng: np.random.Generator,
    min_z: float = RELEVANCE_MIN_Z,
) -> list[str]:
    """Top-k lemmas by similarity to `query`, considering only those that
    stand out from the bank (z-score >= min_z), then nudged toward words
    due for review and jittered slightly so the same conversation doesn't
    always produce the identical list."""
    if k <= 0 or len(lemmas) < _MIN_BANK_FOR_RELEVANCE:
        return []
    sims = matrix @ query
    std = float(sims.std())
    if std == 0.0:
        return []
    eligible = (sims - sims.mean()) / std >= min_z
    if not eligible.any():
        return []
    scores = sims * (0.5 + 0.5 * urgencies) * rng.uniform(0.9, 1.1, size=len(sims))
    scores[~eligible] = -np.inf
    order = np.argsort(scores)[::-1][: min(k, int(eligible.sum()))]
    return [lemmas[i] for i in order]


def _recent_assistant_ids(session: Session, session_id: str, n: int) -> list[int]:
    return list(
        session.scalars(
            select(ChatMessage.id)
            .where(ChatMessage.session_id == session_id, ChatMessage.role == "assistant")
            .order_by(ChatMessage.id.desc())
            .limit(n)
        ).all()
    )


def _cooldown_lemmas(session: Session, session_id: str) -> set[str]:
    """Words the assistant itself used, or that were offered, in the last
    few turns - no point pushing a word that's already been around."""
    ids = _recent_assistant_ids(session, session_id, SUGGEST_COOLDOWN_TURNS)
    if not ids:
        return set()
    used = session.scalars(select(MessageToken.lemma).where(MessageToken.message_id.in_(ids))).all()
    offered = session.scalars(select(VocabSuggestion.lemma).where(VocabSuggestion.assistant_message_id.in_(ids))).all()
    return set(used) | set(offered)


def suggest_for_turn(
    session: Session,
    session_id: str,
    history: list[tuple[str, str]],
    user_message: str,
    rng: np.random.Generator | None = None,
) -> TurnSuggestions:
    """Read-only (no DB writes - see chat.py's _gloss_and_persist_reply on
    why nothing may hold SQLite's write lock before the reply is in)."""
    rng = rng or np.random.default_rng()
    try:
        cooldown = _cooldown_lemmas(session, session_id)
        entries = [
            e
            for e in session.scalars(select(WordBankEntry).where(WordBankEntry.pos.in_(OPEN_CLASS_POS))).all()
            if e.lemma not in cooldown and e.lemma.isalpha()
        ]
        if not entries:
            return TurnSuggestions()
        now = datetime.utcnow()
        urgency = {e.lemma: review_urgency(_to_state(e), now) for e in entries}
        by_lemma = {e.lemma: e for e in entries}

        relevant: list[str] = []
        vectors = embeddings.load_vectors(session, list(by_lemma))
        if len(vectors) >= _MIN_BANK_FOR_RELEVANCE:
            try:
                query_text = build_query_text(summary.get_summary(session, session_id), history, user_message)
                query = embeddings.embed_query(query_text)
                lemmas = list(vectors)
                relevant = rank_relevant(
                    query,
                    lemmas,
                    np.stack([vectors[lemma] for lemma in lemmas]),
                    np.array([urgency[lemma] for lemma in lemmas]),
                    RELEVANT_WORDS_PER_TURN,
                    rng,
                )
            except embeddings.EmbeddingUnavailableError as exc:
                # Wildcards below still work without embeddings.
                logger.warning("contextual vocab: relevance skipped: %s", exc)

        arms = [
            bandit.BanditArm(e.lemma, urgency[e.lemma], e.suggest_alpha, e.suggest_beta)
            for e in entries
            if e.lemma not in relevant
        ]
        wildcard = bandit.thompson_pick(arms, WILDCARD_WORDS_PER_TURN, rng)

        def word(lemma: str, kind: str) -> SuggestedWord:
            return SuggestedWord(lemma, by_lemma[lemma].primary_translation, kind)

        return TurnSuggestions(
            relevant=[word(lemma, "relevant") for lemma in relevant],
            wildcard=[word(lemma, "wildcard") for lemma in wildcard],
        )
    except Exception:
        logger.exception("contextual vocab: suggestion failed; continuing without suggestions")
        return TurnSuggestions()


def record_outcome(
    session: Session,
    session_id: str,
    assistant_message_id: int,
    suggestions: TurnSuggestions,
    reply_lemmas: set[str],
) -> None:
    """Logs what was offered and updates each offered word's bandit counts.
    Runs inside the caller's transaction (no commit here)."""
    for suggested in suggestions.all:
        used = suggested.lemma in reply_lemmas
        entry = session.scalar(select(WordBankEntry).where(WordBankEntry.lemma == suggested.lemma))
        if entry is not None:
            entry.suggest_alpha, entry.suggest_beta = bandit.update_counts(
                entry.suggest_alpha, entry.suggest_beta, 1.0 if used else 0.0
            )
        session.add(
            VocabSuggestion(
                session_id=session_id,
                assistant_message_id=assistant_message_id,
                lemma=suggested.lemma,
                kind=suggested.kind,
                used=used,
            )
        )


def record_pick_up(session: Session, lemma: str) -> bool:
    """The learner typed `lemma` themselves. If it was offered recently,
    that's the strongest evidence the suggestion paid off - reward it.
    Runs inside the caller's transaction (no commit here)."""
    offered = session.scalar(
        select(VocabSuggestion)
        .where(
            VocabSuggestion.lemma == lemma,
            VocabSuggestion.picked_up.is_(False),
            VocabSuggestion.created_at >= datetime.utcnow() - _PICK_UP_WINDOW,
        )
        .order_by(VocabSuggestion.id.desc())
    )
    if offered is None:
        return False
    offered.picked_up = True
    entry = session.scalar(select(WordBankEntry).where(WordBankEntry.lemma == lemma))
    if entry is not None:
        entry.suggest_alpha += bandit.PICKED_UP_BONUS
    return True


def post_turn(session_factory, session_id: str, user_message: str, assistant_reply: str) -> None:
    """The background half of a turn: refresh the running summary, then
    embed whatever new words the turn added to the bank."""
    session = session_factory()
    try:
        try:
            summary.update_summary(session, session_id, user_message, assistant_reply)
        except ModelServerUnavailableError as exc:
            logger.warning("contextual vocab: summary update failed: %s", exc)
        try:
            embeddings.embed_missing_words(session, OPEN_CLASS_POS)
        except embeddings.EmbeddingUnavailableError as exc:
            logger.warning("contextual vocab: embedding new words failed: %s", exc)
    except Exception:
        logger.exception("contextual vocab: post-turn job failed")
    finally:
        session.close()


# One worker: post-turn jobs run in order, so two quick turns can't race
# each other's summary update, and SQLite only ever sees one of them writing.
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="vocab-post-turn")


def schedule_post_turn(session_factory, session_id: str, user_message: str, assistant_reply: str) -> None:
    _executor.submit(post_turn, session_factory, session_id, user_message, assistant_reply)
