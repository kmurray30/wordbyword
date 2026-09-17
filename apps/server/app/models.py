from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text
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


class ChatMessage(Base):
    __tablename__ = "chat_message"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    role: Mapped[str] = mapped_column(String(16))  # "user" | "assistant"
    text: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    tokens: Mapped[list["MessageToken"]] = relationship(back_populates="message", cascade="all, delete-orphan")


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

    message: Mapped[ChatMessage] = relationship(back_populates="tokens")
