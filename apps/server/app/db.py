from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.config import DATABASE_URL


class Base(DeclarativeBase):
    pass


engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def init_db() -> None:
    from app import models  # noqa: F401  (ensure models are registered)

    Base.metadata.create_all(bind=engine)
    add_missing_columns(engine)


def add_missing_columns(target_engine) -> None:
    """`create_all` only creates missing TABLES - it never alters a table
    that already exists, so a column added to a model after the table was
    first created (in an already-running deployment's DB) needs a manual
    ALTER. This project has no migration tool (prototype scope, see
    README's known limitations), so handle it by hand, one column at a
    time, rather than pulling one in for what's so far a single column.
    Takes an engine explicitly (rather than closing over the module-level
    one) so it's exercisable against a throwaway test DB."""
    with target_engine.begin() as conn:
        # PRAGMA table_info silently returns zero rows for a table that
        # doesn't exist at all (rather than erroring), which an ALTER
        # against that same nonexistent table would NOT - every patch below
        # is guarded on its table actually existing first (a fresh DB
        # without it yet gets it created already-current by create_all, so
        # there's nothing to patch).
        def table_exists(name: str) -> bool:
            return bool(conn.exec_driver_sql(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{name}'").first())

        if table_exists("chat_message"):
            existing = {row[1] for row in conn.exec_driver_sql("PRAGMA table_info(chat_message)")}
            if "session_id" not in existing:
                conn.exec_driver_sql("ALTER TABLE chat_message ADD COLUMN session_id VARCHAR(64) DEFAULT ''")

        if table_exists("message_token"):
            existing_token_columns = {row[1] for row in conn.exec_driver_sql("PRAGMA table_info(message_token)")}
            if "note" not in existing_token_columns:
                conn.exec_driver_sql("ALTER TABLE message_token ADD COLUMN note VARCHAR(512) DEFAULT ''")
            if "literal" not in existing_token_columns:
                conn.exec_driver_sql("ALTER TABLE message_token ADD COLUMN literal VARCHAR(512) DEFAULT ''")


def get_session() -> Session:
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
