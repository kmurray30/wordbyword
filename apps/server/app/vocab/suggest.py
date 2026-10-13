"""Contextual vocabulary suggestions: which words the system prompt offers
the model this turn, and the bookkeeping after it replies.

The candidate pool is the learner's word bank (every word the partner has
used - app.models.WordBankEntry) plus their saved word list (SavedWord, the
words they chose to keep, which get priority). Per turn, only when
CONTEXTUAL_VOCAB_ENABLED:
  1. suggest_for_turn - before the model call, read-only: rank the pool by
     embedding similarity to the conversation (running summary + latest
     messages + the new user message), keep the words that clear a cutoff,
     weighted toward words due for review; then let the bandit add a
     wildcard or two.
  2. record_outcome - inside the turn's own short write transaction: log
     what was offered and reward each word by whether the reply used it.
  3. schedule_post_turn - after the turn commits, in a background thread:
     refresh the running summary and embed new words, so neither LLM nor
     embedding call sits in front of the reply.

Every step degrades to "no suggestions" on failure - this feature never
fails a turn. The whole search (scores, who cleared the cutoff, the
wildcards) is kept as a small JSON blob on the reply for the debug view.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.chat.openai_client import ModelServerUnavailableError
from app.config import RELEVANCE_MIN_Z, RELEVANT_WORDS_PER_TURN, SUGGEST_COOLDOWN_TURNS, WILDCARD_WORDS_PER_TURN
from app.languages import get_language, language_name
from app.models import ChatMessage, MessageToken, SavedWord, VocabSuggestion, WordBankEntry
from app.vocab import bandit, embeddings, summary
from app.wordbank import store
from app.wordbank.scoring import review_urgency
from app.wordbank.store import _to_state

logger = logging.getLogger(__name__)

# spaCy UPOS tags that make a word not worth suggesting: closed-class words
# (articles, prepositions, pronouns, conjunctions...) show up in every reply
# anyway, and names/numbers aren't vocabulary. Languages without a tagger
# store pos "" - those words stay candidates, and simply don't score as
# relevant to a topic.
EXCLUDED_POS = frozenset({"DET", "ADP", "PRON", "CCONJ", "SCONJ", "AUX", "PART", "PUNCT", "NUM", "SYM", "PROPN", "X"})

# Words the learner saved to their list are wanted explicitly: they count as
# at least this urgent even when freshly seen, and get this extra weight
# both as wildcards and when ranking relevant words.
SAVED_URGENCY_FLOOR = 0.6
SAVED_WEIGHT = 1.5

# Below this many embedded candidates, "relevant relative to the rest of
# the pool" isn't meaningful yet - skip the relevance side entirely.
_MIN_BANK_FOR_RELEVANCE = 10

# How many of the latest messages go into the relevance query verbatim
# (the running summary covers everything before them).
_QUERY_RECENT_MESSAGES = 3

# How long after being offered a word still counts as "the suggestion
# paid off" if the learner types it themselves.
_PICK_UP_WINDOW = timedelta(minutes=30)

# How many of the top search results the debug view keeps (words that were
# offered are always kept).
_DEBUG_TOP_RESULTS = 20

# Saved words with no vector yet, embedded on the spot so a word added a
# moment ago can already count as relevant (everything else waits for the
# background job).
_EMBED_NOW_LIMIT = 24


@dataclass
class SuggestedWord:
    lemma: str
    translation: str
    kind: str  # "relevant" | "wildcard"


@dataclass
class DebugWord:
    """One row of the debug view."""

    lemma: str
    translation: str = ""
    sim: float | None = None  # cosine similarity to the conversation
    z: float | None = None  # how many std-devs above the pool's average
    urgency: float = 0.0  # how due for review (0-1)
    # offered: cleared the cutoff and was picked | passed: cleared the cutoff
    # but beyond the per-turn limit | cooldown: relevant, but used/offered
    # too recently | below: under the cutoff | wildcard: the random pick
    status: str = "below"
    saved: bool = False
    used: bool = False  # the reply actually used it


@dataclass
class TurnSuggestions:
    relevant: list[SuggestedWord] = field(default_factory=list)
    wildcard: list[SuggestedWord] = field(default_factory=list)
    search: list[DebugWord] = field(default_factory=list)
    wildcard_debug: list[DebugWord] = field(default_factory=list)
    searched: bool = False  # the embedding search actually ran
    error: str = ""  # why it didn't, if it couldn't

    @property
    def all(self) -> list[SuggestedWord]:
        return self.relevant + self.wildcard

    def debug_dict(self, reply_words: set[str]) -> dict:
        """The JSON-able debug payload, with each word's `used` filled in."""
        offered = {w.lemma for w in self.all}
        rows = []
        for source in (self.search, self.wildcard_debug):
            rows.append([asdict(w) | {"used": w.lemma in offered and w.lemma in reply_words} for w in source])
        return {
            "cutoff_z": RELEVANCE_MIN_Z,
            "searched": self.searched,
            "error": self.error,
            "search": rows[0],
            "wildcards": rows[1],
        }


@dataclass
class _Candidate:
    lemma: str
    translation: str
    saved: bool
    urgency: float
    alpha: float
    beta: float

    @property
    def weight(self) -> float:
        return SAVED_WEIGHT if self.saved else 1.0


def build_query_text(conversation_summary: str, history: list[tuple[str, str]], user_message: str) -> str:
    parts = []
    if conversation_summary:
        parts.append(f"Conversation so far: {conversation_summary}")
    for role, text in history[-_QUERY_RECENT_MESSAGES:]:
        parts.append(f"{'Learner' if role == 'user' else 'Partner'}: {text}")
    parts.append(f"Learner: {user_message}")
    return "\n".join(parts)


def _candidates(session: Session, session_id: str, language: str) -> list[_Candidate]:
    """The pool: word-bank words (minus closed-class ones) plus the learner's
    saved words, one entry per lemma."""
    now = datetime.utcnow()
    entries = {
        e.lemma: e
        for e in session.scalars(select(WordBankEntry).where(WordBankEntry.language == language)).all()
        if e.pos not in EXCLUDED_POS and any(c.isalpha() for c in e.lemma)
    }
    saved = session.scalars(select(SavedWord).where(SavedWord.session_id == session_id)).all()
    by_lemma: dict[str, _Candidate] = {}
    for lemma, e in entries.items():
        by_lemma[lemma] = _Candidate(
            lemma, e.primary_translation, False, review_urgency(_to_state(e), now), e.suggest_alpha, e.suggest_beta
        )
    for word in saved:
        key = word.word_key
        existing = by_lemma.get(key)
        if existing is not None:
            existing.saved = True
            existing.urgency = max(existing.urgency, SAVED_URGENCY_FLOOR)
            existing.translation = existing.translation or word.gloss
            continue
        entry = session.scalar(
            select(WordBankEntry).where(WordBankEntry.language == language, WordBankEntry.lemma == key)
        )
        urgency = review_urgency(_to_state(entry), now) if entry else 0.0
        by_lemma[key] = _Candidate(
            key,
            (entry.primary_translation if entry else "") or word.gloss,
            True,
            max(urgency, SAVED_URGENCY_FLOOR),
            entry.suggest_alpha if entry else 1.0,
            entry.suggest_beta if entry else 1.0,
        )
    return list(by_lemma.values())


def pool_words(session: Session, session_id: str, language: str) -> list[tuple[str, str]]:
    return [(c.lemma, c.translation) for c in _candidates(session, session_id, language)]


def score_candidates(
    query: np.ndarray,
    lemmas: list[str],
    matrix: np.ndarray,
    urgencies: np.ndarray,
    cooldown: set[str],
    k: int,
    rng: np.random.Generator,
    min_z: float = RELEVANCE_MIN_Z,
    weights: np.ndarray | None = None,
) -> tuple[list[str], dict[str, tuple[float, float, str]]]:
    """Ranks the pool against `query`. Returns (the lemmas to offer, and for
    every lemma its (similarity, z-score, status) - see DebugWord.status).

    A word clears the cutoff if its similarity stands out from the pool
    (z-score >= min_z - relative, so it doesn't depend on any one embedding
    model's raw scale). Of those, the top-k - by similarity, nudged toward
    words due for review (and toward saved words, via `weights`), with a
    little jitter so the same conversation doesn't always produce the
    identical list - are offered."""
    sims = matrix @ query
    std = float(sims.std())
    z = (sims - sims.mean()) / std if std > 0 else np.zeros_like(sims)
    info: dict[str, tuple[float, float, str]] = {}
    pickable: list[int] = []
    for i, lemma in enumerate(lemmas):
        if z[i] < min_z or std == 0.0:
            status = "below"
        elif lemma in cooldown:
            status = "cooldown"
        else:
            status = "passed"
            pickable.append(i)
        info[lemma] = (float(sims[i]), float(z[i]), status)
    scores = sims * (0.5 + 0.5 * urgencies) * rng.uniform(0.9, 1.1, size=len(sims))
    if weights is not None:
        scores = scores * weights
    chosen = sorted(pickable, key=lambda i: scores[i], reverse=True)[: max(k, 0)]
    for i in chosen:
        sim, zi, _ = info[lemmas[i]]
        info[lemmas[i]] = (sim, zi, "offered")
    return [lemmas[i] for i in chosen], info


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


def _vectors_for(session: Session, language: str, candidates: list[_Candidate]) -> dict[str, np.ndarray]:
    vectors = embeddings.load_vectors(session, language, [c.lemma for c in candidates])
    # A word saved a moment ago may not have been embedded by the background
    # job yet - embed those few now (in memory only; the job persists them).
    fresh = [c for c in candidates if c.saved and c.lemma not in vectors][:_EMBED_NOW_LIMIT]
    if fresh:
        embeddings.remember_vectors(language, [(c.lemma, c.translation) for c in fresh])
        vectors = embeddings.load_vectors(session, language, [c.lemma for c in candidates])
    return vectors


def suggest_for_turn(
    session: Session,
    session_id: str,
    language: str,
    history: list[tuple[str, str]],
    user_message: str,
    rng: np.random.Generator | None = None,
) -> TurnSuggestions:
    """Read-only (no DB writes - see routes/chat.py's _gloss_and_persist_reply
    on why nothing may hold SQLite's write lock before the reply is in)."""
    rng = rng or np.random.default_rng()
    result = TurnSuggestions()
    try:
        language = get_language(language).code
        candidates = _candidates(session, session_id, language)
        if not candidates:
            return result
        cooldown = _cooldown_lemmas(session, session_id)
        by_lemma = {c.lemma: c for c in candidates}

        offered: list[str] = []
        search_rows: dict[str, DebugWord] = {}
        try:
            vectors = _vectors_for(session, language, candidates)
            if len(vectors) < _MIN_BANK_FOR_RELEVANCE:
                result.error = f"only {len(vectors)} words embedded so far (need {_MIN_BANK_FOR_RELEVANCE})"
            else:
                query_text = build_query_text(summary.get_summary(session, session_id), history, user_message)
                query = embeddings.embed_query(query_text, language_name(language))
                lemmas = list(vectors)
                offered, info = score_candidates(
                    query,
                    lemmas,
                    np.stack([vectors[lemma] for lemma in lemmas]),
                    np.array([by_lemma[lemma].urgency for lemma in lemmas]),
                    cooldown,
                    RELEVANT_WORDS_PER_TURN,
                    rng,
                    weights=np.array([by_lemma[lemma].weight for lemma in lemmas]),
                )
                result.searched = True
                for lemma, (sim, z, status) in info.items():
                    c = by_lemma[lemma]
                    search_rows[lemma] = DebugWord(lemma, c.translation, sim, z, c.urgency, status, c.saved)
        except embeddings.EmbeddingUnavailableError as exc:
            # Wildcards below still work without embeddings.
            logger.warning("contextual vocab: relevance skipped: %s", exc)
            result.error = str(exc)

        arms = [
            bandit.BanditArm(c.lemma, c.urgency, c.alpha, c.beta, c.weight)
            for c in candidates
            if c.lemma not in offered and c.lemma not in cooldown
        ]
        wildcards = bandit.thompson_pick(arms, WILDCARD_WORDS_PER_TURN, rng)

        result.relevant = [SuggestedWord(lemma, by_lemma[lemma].translation, "relevant") for lemma in offered]
        result.wildcard = [SuggestedWord(lemma, by_lemma[lemma].translation, "wildcard") for lemma in wildcards]
        result.wildcard_debug = [
            DebugWord(lemma, by_lemma[lemma].translation, None, None, by_lemma[lemma].urgency, "wildcard", by_lemma[lemma].saved)
            for lemma in wildcards
        ]
        ranked = sorted(search_rows.values(), key=lambda w: w.sim or 0.0, reverse=True)
        result.search = [w for i, w in enumerate(ranked) if i < _DEBUG_TOP_RESULTS or w.status == "offered"]
        return result
    except Exception:
        logger.exception("contextual vocab: suggestion failed; continuing without suggestions")
        return TurnSuggestions(error="suggestion failed (see server log)")


def record_outcome(
    session: Session,
    session_id: str,
    language: str,
    assistant_message_id: int,
    suggestions: TurnSuggestions,
    reply_words: set[str],
) -> None:
    """Logs what was offered and updates each offered word's bandit counts.
    Runs inside the caller's transaction (no commit here)."""
    language = get_language(language).code
    for suggested in suggestions.all:
        used = suggested.lemma in reply_words
        # get_or_create: a saved word the partner never used has no bank
        # entry yet, and its counts need somewhere to live.
        entry = store.get_or_create(session, suggested.lemma, translation=suggested.translation, language=language)
        if suggested.translation and not entry.primary_translation:
            entry.primary_translation = suggested.translation
        entry.suggest_alpha, entry.suggest_beta = bandit.update_counts(
            entry.suggest_alpha, entry.suggest_beta, 1.0 if used else 0.0
        )
        session.add(
            VocabSuggestion(
                session_id=session_id,
                language=language,
                assistant_message_id=assistant_message_id,
                lemma=suggested.lemma,
                kind=suggested.kind,
                used=used,
            )
        )


def record_pick_up(session: Session, lemma: str, language: str = "es") -> bool:
    """The learner typed `lemma` themselves. If it was offered recently,
    that's the strongest evidence the suggestion paid off - reward it.
    Runs inside the caller's transaction (no commit here)."""
    language = get_language(language).code
    offered = session.scalar(
        select(VocabSuggestion)
        .where(
            VocabSuggestion.language == language,
            VocabSuggestion.lemma == lemma,
            VocabSuggestion.picked_up.is_(False),
            VocabSuggestion.created_at >= datetime.utcnow() - _PICK_UP_WINDOW,
        )
        .order_by(VocabSuggestion.id.desc())
    )
    if offered is None:
        return False
    offered.picked_up = True
    entry = session.scalar(select(WordBankEntry).where(WordBankEntry.language == language, WordBankEntry.lemma == lemma))
    if entry is not None:
        entry.suggest_alpha += bandit.PICKED_UP_BONUS
    return True


def post_turn(session_factory, session_id: str, language: str, user_message: str, assistant_reply: str) -> None:
    """The background half of a turn: refresh the running summary, then
    embed whatever new words are in the pool."""
    session = session_factory()
    try:
        try:
            summary.update_summary(session, session_id, user_message, assistant_reply)
        except ModelServerUnavailableError as exc:
            logger.warning("contextual vocab: summary update failed: %s", exc)
        try:
            language = get_language(language).code
            embeddings.embed_missing(session, language, pool_words(session, session_id, language))
        except embeddings.EmbeddingUnavailableError as exc:
            logger.warning("contextual vocab: embedding new words failed: %s", exc)
    except Exception:
        logger.exception("contextual vocab: post-turn job failed")
    finally:
        session.close()


# One worker: post-turn jobs run in order, so two quick turns can't race
# each other's summary update, and SQLite only ever sees one of them writing.
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="vocab-post-turn")


def schedule_post_turn(session_factory, session_id: str, language: str, user_message: str, assistant_reply: str) -> None:
    _executor.submit(post_turn, session_factory, session_id, language, user_message, assistant_reply)
