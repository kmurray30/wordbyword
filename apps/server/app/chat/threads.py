"""Per-message side conversations (see models.ThreadEntry): the coach checks
behind a sent message, and the learner's questions about a message with the
tutor's answers."""

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models import ThreadEntry
from app.schemas import ThreadEntryIn, ThreadEntryOut

_VALID_ROLES = {"learner", "coach", "tutor"}
_VALID_KINDS = {"check", "question", "answer"}
# Generous caps - a check's JSON (meaning + three options with
# translations) is well under this; the point is only to refuse junk.
_MAX_TEXT = 4000
_MAX_DATA = 16000
_MAX_ENTRIES_PER_SEND = 20


def append_entries(session: Session, message_id: int, entries: list[ThreadEntryIn]) -> list[ThreadEntry]:
    """Adds `entries` to the end of a message's thread (not committed -
    the caller owns the transaction). Malformed entries are skipped rather
    than failing the whole write: the thread is supplementary, the message
    itself must still go through."""
    position = session.scalar(
        select(func.coalesce(func.max(ThreadEntry.position), -1)).where(ThreadEntry.message_id == message_id)
    )
    rows: list[ThreadEntry] = []
    for entry in entries[:_MAX_ENTRIES_PER_SEND]:
        if entry.role not in _VALID_ROLES or entry.kind not in _VALID_KINDS:
            continue
        position += 1
        row = ThreadEntry(
            message_id=message_id,
            position=position,
            role=entry.role,
            kind=entry.kind,
            text=entry.text[:_MAX_TEXT],
            data=entry.data[:_MAX_DATA],
        )
        session.add(row)
        rows.append(row)
    return rows


def entry_out(row: ThreadEntry) -> ThreadEntryOut:
    return ThreadEntryOut(id=row.id, role=row.role, kind=row.kind, text=row.text, data=row.data)
