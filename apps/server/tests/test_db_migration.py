"""Covers app/db.py's manual schema migration - the one place this
prototype-scope app (no migration tool, see README) needs to alter a table
that may already exist in a live deployment's DB rather than just create it
fresh. Exercises a real SQLite file via a throwaway engine, not the app's
shared global one, so it doesn't touch whatever DB the rest of the suite
might be pointed at."""

import sqlite3
from pathlib import Path

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from app.db import add_missing_columns


def _make_old_schema_db(path: Path) -> None:
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE chat_message (id INTEGER PRIMARY KEY, role VARCHAR(16), text TEXT, created_at DATETIME)"
    )
    conn.execute("INSERT INTO chat_message (role, text, created_at) VALUES ('user', 'legacy row', '2026-01-01')")
    conn.commit()
    conn.close()


def test_adds_session_id_column_to_pre_existing_table(tmp_path):
    db_path = tmp_path / "old.db"
    _make_old_schema_db(db_path)
    engine = create_engine(f"sqlite:///{db_path}")

    add_missing_columns(engine)

    with engine.connect() as conn:
        columns = {row[1] for row in conn.exec_driver_sql("PRAGMA table_info(chat_message)")}
        assert "session_id" in columns


def test_legacy_rows_get_empty_string_session_id_not_null(tmp_path):
    db_path = tmp_path / "old.db"
    _make_old_schema_db(db_path)
    engine = create_engine(f"sqlite:///{db_path}")
    add_missing_columns(engine)

    Session = sessionmaker(bind=engine)
    session = Session()
    row = session.execute(text("SELECT session_id, text FROM chat_message")).first()
    assert row.session_id == ""
    assert row.text == "legacy row"


def test_idempotent_on_a_table_that_already_has_the_column(tmp_path):
    db_path = tmp_path / "new.db"
    engine = create_engine(f"sqlite:///{db_path}")
    with engine.begin() as conn:
        conn.exec_driver_sql(
            "CREATE TABLE chat_message (id INTEGER PRIMARY KEY, session_id VARCHAR(64) DEFAULT '', "
            "role VARCHAR(16), text TEXT, created_at DATETIME)"
        )

    add_missing_columns(engine)  # should not raise (e.g. duplicate column error)

    with engine.connect() as conn:
        columns = [row[1] for row in conn.exec_driver_sql("PRAGMA table_info(chat_message)")]
        assert columns.count("session_id") == 1
