from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, LargeBinary, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.config import DEFAULT_REVIEW_INTERVAL_DAYS
from app.db import Base


def utcnow() -> datetime:
    # Naive UTC, not timezone-aware: SQLite (via SQLAlchemy) round-trips
    # DateTime columns as naive datetimes regardless of column type, so
    # mixing naive/aware here would raise on comparison after a DB read.
    return datetime.utcnow()


class WordBankEntry(Base):
    """Per-lemma mastery record. This is the 'word bank' the RL-style
    weighting algorithm reads from and writes back to."""

    __tablename__ = "word_bank"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    lemma: Mapped[str] = mapped_column(String(128), unique=True, index=True)
    pos: Mapped[str] = mapped_column(String(32), default="")
    primary_translation: Mapped[str] = mapped_column(String(256), default="")

    familiarity: Mapped[float] = mapped_column(Float, default=0.0)
    exposure_count: Mapped[int] = mapped_column(Integer, default=0)

    introduced_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    last_seen_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    last_reviewed_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    review_interval_days: Mapped[float] = mapped_column(Float, default=DEFAULT_REVIEW_INTERVAL_DAYS)

    # Contextual-vocab bandit (app/vocab/bandit.py): a Beta(alpha, beta)
    # estimate of how likely this word is to actually get worked into a
    # reply when the prompt offers it. Starts at the uniform prior.
    suggest_alpha: Mapped[float] = mapped_column(Float, default=1.0)
    suggest_beta: Mapped[float] = mapped_column(Float, default=1.0)


class ChatMessage(Base):
    __tablename__ = "chat_message"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    # Per-browser session (see lib/session.ts on the frontend), not a login -
    # this app is still single-user/no-auth (see README), but each session
    # gets its own conversation thread rather than sharing one global chat
    # history. Empty string on rows written before this column existed;
    # those are simply invisible to every session now, which is fine.
    session_id: Mapped[str] = mapped_column(String(64), default="", index=True)
    role: Mapped[str] = mapped_column(String(16))  # "user" | "assistant"
    text: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    # Only populated for role == "user": the same corrected-native-text/
    # target-translation pair /translate/interpret computes live, persisted
    # here too so a history-hydrated message doesn't need to re-fetch it
    # (and show "Translating…" indefinitely if that fetch is ever slow/
    # stuck) on every page reload. Empty string if the LLM call failed when
    # this message was first sent - same "no fallback" convention as the
    # assistant side's note/literal columns below.
    native_text: Mapped[str] = mapped_column(String(1024), default="")
    target_text: Mapped[str] = mapped_column(String(1024), default="")

    tokens: Mapped[list["MessageToken"]] = relationship(
        back_populates="message", cascade="all, delete-orphan", order_by="MessageToken.position"
    )


class MessageToken(Base):
    """Per-token annotation for an assistant message, so hover/translate-all
    reward events can be tied back to the lemma they affect."""

    __tablename__ = "message_token"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    message_id: Mapped[int] = mapped_column(ForeignKey("chat_message.id"))
    position: Mapped[int] = mapped_column(Integer)
    surface: Mapped[str] = mapped_column(String(128))
    lemma: Mapped[str] = mapped_column(String(128))
    pos: Mapped[str] = mapped_column(String(32), default="")
    gloss: Mapped[str] = mapped_column(String(256), default="")
    is_new: Mapped[bool] = mapped_column(default=False)
    # Same disambiguation-note/word-by-word-breakdown fields TokenAnnotation
    # carries live (see app/schemas.py) - persisted too, not just returned
    # in the moment, so GET /chat/history (which rebuilds tokens purely
    # from these rows) doesn't silently flatten a reloaded message's words
    # back down to their bare gloss on the very next page load.
    note: Mapped[str] = mapped_column(String(512), default="")
    literal: Mapped[str] = mapped_column(String(512), default="")

    message: Mapped[ChatMessage] = relationship(back_populates="tokens")


class WordEmbedding(Base):
    """Cached embedding of one word-bank lemma (app/vocab/embeddings.py).
    Keyed by model too, so switching EMBEDDING_MODEL just re-embeds rather
    than comparing vectors from two different spaces."""

    __tablename__ = "word_embedding"

    lemma: Mapped[str] = mapped_column(String(128), primary_key=True)
    model: Mapped[str] = mapped_column(String(128), primary_key=True)
    vector: Mapped[bytes] = mapped_column(LargeBinary)  # float32, L2-normalized
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class ConversationSummary(Base):
    """A short running summary of one session's conversation, refreshed in
    the background after every turn (app/vocab/summary.py) so the next
    turn's relevance query gets the conversation's gist without waiting on
    an extra LLM call."""

    __tablename__ = "conversation_summary"

    session_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    summary: Mapped[str] = mapped_column(Text, default="")
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class VocabSuggestion(Base):
    """One word the prompt offered on one turn, and whether the reply ended
    up using it - the bandit's reward log, and the cooldown's memory of
    what was offered recently."""

    __tablename__ = "vocab_suggestion"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    session_id: Mapped[str] = mapped_column(String(64), index=True)
    assistant_message_id: Mapped[int] = mapped_column(Integer, index=True)
    lemma: Mapped[str] = mapped_column(String(128), index=True)
    kind: Mapped[str] = mapped_column(String(16))  # "relevant" | "wildcard"
    used: Mapped[bool] = mapped_column(Boolean, default=False)
    # The learner later typed this word themselves (a userTypedSpanishWord
    # reward event) - the strongest signal the suggestion paid off.
    picked_up: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, index=True)
