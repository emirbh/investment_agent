from __future__ import annotations

import os
import sqlite3

_DEFAULT_DB_PATH: str = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "investment.db"
)


def get_db_path() -> str:
    return os.environ.get("INVESTMENT_DB_PATH", _DEFAULT_DB_PATH)


def get_connection(db_path: str | None = None) -> sqlite3.Connection:
    path: str = db_path or get_db_path()
    conn: sqlite3.Connection = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = sqlite3.Row
    return conn
