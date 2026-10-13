from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, Index, Integer, String, Text
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
    # One entry per word per language ("la" is Spanish's and French's both).
    __table_args__ = (Index("uq_word_bank_language_lemma", "language", "lemma", unique=True),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    language: Mapped[str] = mapped_column(String(8), default="es")
    lemma: Mapped[str] = mapped_column(String(128), index=True)
    pos: Mapped[str] = mapped_column(String(32), default="")
    primary_translation: Mapped[str] = mapped_column(String(256), default="")

    familiarity: Mapped[float] = mapped_column(Float, default=0.0)
    exposure_count: Mapped[int] = mapped_column(Integer, default=0)

    introduced_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    last_seen_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    last_reviewed_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    review_interval_days: Mapped[float] = mapped_column(Float, default=DEFAULT_REVIEW_INTERVAL_DAYS)


class ChatMessage(Base):
    __tablename__ = "chat_message"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    # Per-browser session (see lib/session.ts on the frontend), not a login -
    # this app is still single-user/no-auth (see README), but each session
    # gets its own conversation thread rather than sharing one global chat
    # history. Empty string on rows written before this column existed;
    # those are simply invisible to every session now, which is fine.
    session_id: Mapped[str] = mapped_column(String(64), default="", index=True)
    # The language the conversation is in (each language has its own
    # session - see lib/session.ts on the frontend). Rows from before more
    # languages existed are Spanish.
    language: Mapped[str] = mapped_column(String(8), default="es")
    role: Mapped[str] = mapped_column(String(16))  # "user" | "assistant"
    text: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    # The message's translation, saved so a reload doesn't have to fetch it
    # again (and show "Translating…" meanwhile). role == "user": the
    # corrected-native-text/target-translation pair /translate/interpret
    # computes. role == "assistant": native_text is the reply's translation
    # from the same gloss pass as its word glosses; target_text unused.
    # Empty if the LLM call failed - filled in later by the frontend via
    # POST /chat/message/interpretation.
    native_text: Mapped[str] = mapped_column(String(1024), default="")
    target_text: Mapped[str] = mapped_column(String(1024), default="")
    # The message's word/phrase glosses as JSON (schemas.MessageGlosses),
    # for either role - see app/chat/glosses.py. Empty until made.
    glosses: Mapped[str] = mapped_column(Text, default="")

    tokens: Mapped[list["MessageToken"]] = relationship(
        back_populates="message", cascade="all, delete-orphan", order_by="MessageToken.position"
    )
    # The side conversation hanging off this message (see ThreadEntry) -
    # the main chat shows only the phrases actually exchanged; everything
    # said ABOUT a message lives here.
    thread: Mapped[list["ThreadEntry"]] = relationship(
        back_populates="message", cascade="all, delete-orphan", order_by="ThreadEntry.position"
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


class ThreadEntry(Base):
    """One step of the side conversation attached to a chat message:
    - kind "check": a coach check of a draft the learner ran before sending
      this (user) message. `text` is the draft as checked; `data` is the
      JSON of the coach's result (verdict, meaning, suggestion, options,
      which option was picked).
    - kind "question" / "answer": a learner question about this message and
      the tutor's answer (drill-downs on words, grammar, "break that down").
    """

    __tablename__ = "thread_entry"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    message_id: Mapped[int] = mapped_column(ForeignKey("chat_message.id"), index=True)
    position: Mapped[int] = mapped_column(Integer)
    role: Mapped[str] = mapped_column(String(16))  # "learner" | "coach" | "tutor"
    kind: Mapped[str] = mapped_column(String(16))  # "check" | "question" | "answer"
    text: Mapped[str] = mapped_column(Text, default="")
    data: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    message: Mapped[ChatMessage] = relationship(back_populates="thread")


class SavedWord(Base):
    """A word the learner chose to keep (tap a word twice in a chat
    message). Just a list for now - meant to feed a flashcard app later, so
    it keeps the word's meaning and the sentence it came from."""

    __tablename__ = "saved_word"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    session_id: Mapped[str] = mapped_column(String(64), default="", index=True)
    # As it appeared, and lowercased (what "already saved" is checked by).
    word: Mapped[str] = mapped_column(String(128))
    word_key: Mapped[str] = mapped_column(String(128), index=True)
    gloss: Mapped[str] = mapped_column(String(256), default="")
    grammar: Mapped[str] = mapped_column(String(512), default="")
    sentence: Mapped[str] = mapped_column(Text, default="")
    message_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
