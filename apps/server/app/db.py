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
        existing = {row[1] for row in conn.exec_driver_sql("PRAGMA table_info(chat_message)")}
        if "session_id" not in existing:
            conn.exec_driver_sql("ALTER TABLE chat_message ADD COLUMN session_id VARCHAR(64) DEFAULT ''")


def get_session() -> Session:
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
