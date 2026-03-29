from __future__ import annotations

from db.connection import get_connection

SCHEMA_SQL: str = """
CREATE TABLE IF NOT EXISTS portfolio (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL UNIQUE DEFAULT 'default',
    created_at  TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS positions (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    portfolio_id  INTEGER NOT NULL REFERENCES portfolio(id) ON DELETE CASCADE,
    ticker        TEXT NOT NULL,
    shares        REAL NOT NULL DEFAULT 0,
    cost_basis    REAL,
    added_at      TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(portfolio_id, ticker)
);

CREATE TABLE IF NOT EXISTS etf_universe (
    ticker        TEXT PRIMARY KEY,
    name          TEXT NOT NULL,
    asset_class   TEXT NOT NULL DEFAULT 'dividend_etf',
    category      TEXT,
    expense_ratio REAL,
    aum           REAL,
    dividend_yield REAL,
    is_portfolio  INTEGER NOT NULL DEFAULT 0,
    is_peer       INTEGER NOT NULL DEFAULT 0,
    discovered_at TEXT,
    last_updated  TEXT
);

CREATE TABLE IF NOT EXISTS price_history (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker    TEXT NOT NULL,
    date      TEXT NOT NULL,
    open      REAL,
    high      REAL,
    low       REAL,
    close     REAL NOT NULL,
    volume    INTEGER,
    dividends REAL DEFAULT 0,
    UNIQUE(ticker, date)
);

CREATE INDEX IF NOT EXISTS idx_price_ticker_date ON price_history(ticker, date);

CREATE TABLE IF NOT EXISTS macro_series (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    series_id  TEXT NOT NULL,
    date       TEXT NOT NULL,
    value      REAL NOT NULL,
    UNIQUE(series_id, date)
);

CREATE INDEX IF NOT EXISTS idx_macro_series_date ON macro_series(series_id, date);

CREATE TABLE IF NOT EXISTS features (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker      TEXT NOT NULL,
    date        TEXT NOT NULL,
    feature_vec TEXT NOT NULL,
    target      REAL,
    UNIQUE(ticker, date)
);

CREATE TABLE IF NOT EXISTS predictions (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    run_date       TEXT NOT NULL,
    ticker         TEXT NOT NULL,
    predicted_rank REAL NOT NULL,
    predicted_ret  REAL,
    confidence     REAL,
    action         TEXT,
    UNIQUE(run_date, ticker)
);

CREATE TABLE IF NOT EXISTS backtest_results (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    run_date            TEXT NOT NULL,
    period_start        TEXT NOT NULL,
    period_end          TEXT NOT NULL,
    strategy_return     REAL NOT NULL,
    baseline_return     REAL NOT NULL,
    strategy_sharpe     REAL,
    baseline_sharpe     REAL,
    max_drawdown        REAL,
    details             TEXT
);
"""


def init_db(db_path: str | None = None) -> None:
    """Create all tables if they don't exist."""
    conn = get_connection(db_path)
    conn.executescript(SCHEMA_SQL)
    conn.commit()
    conn.close()
