from __future__ import annotations

import sqlite3
from datetime import datetime
from typing import Any

from db.connection import get_connection


class PortfolioManager:

    def __init__(self, conn: sqlite3.Connection | None = None) -> None:
        self.conn: sqlite3.Connection = conn or get_connection()

    def create_portfolio(self, name: str = "default") -> int:
        cur = self.conn.execute(
            "INSERT OR IGNORE INTO portfolio (name) VALUES (?)", (name,)
        )
        self.conn.commit()
        if cur.lastrowid:
            return cur.lastrowid
        row = self.conn.execute(
            "SELECT id FROM portfolio WHERE name = ?", (name,)
        ).fetchone()
        return int(row["id"])

    def get_portfolio(self, name: str = "default") -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT * FROM portfolio WHERE name = ?", (name,)
        ).fetchone()
        return dict(row) if row else None

    def list_portfolios(self) -> list[dict[str, Any]]:
        rows = self.conn.execute("SELECT * FROM portfolio ORDER BY name").fetchall()
        return [dict(r) for r in rows]

    def add_position(
        self,
        ticker: str,
        shares: float,
        cost_basis: float | None = None,
        portfolio_name: str = "default",
    ) -> None:
        portfolio: dict[str, Any] | None = self.get_portfolio(portfolio_name)
        if not portfolio:
            self.create_portfolio(portfolio_name)
            portfolio = self.get_portfolio(portfolio_name)

        assert portfolio is not None
        ticker = ticker.upper()
        self.conn.execute(
            """INSERT INTO positions (portfolio_id, ticker, shares, cost_basis)
               VALUES (?, ?, ?, ?)
               ON CONFLICT(portfolio_id, ticker)
               DO UPDATE SET shares = shares + excluded.shares,
                            cost_basis = COALESCE(excluded.cost_basis, cost_basis)""",
            (portfolio["id"], ticker, shares, cost_basis),
        )
        # Also mark in universe
        self.conn.execute(
            """INSERT INTO etf_universe (ticker, name, is_portfolio)
               VALUES (?, ?, 1)
               ON CONFLICT(ticker) DO UPDATE SET is_portfolio = 1""",
            (ticker, ticker),
        )
        self.conn.commit()

    def remove_position(self, ticker: str, portfolio_name: str = "default") -> bool:
        portfolio: dict[str, Any] | None = self.get_portfolio(portfolio_name)
        if not portfolio:
            return False
        ticker = ticker.upper()
        cur = self.conn.execute(
            "DELETE FROM positions WHERE portfolio_id = ? AND ticker = ?",
            (portfolio["id"], ticker),
        )
        if cur.rowcount > 0:
            self.conn.execute(
                "UPDATE etf_universe SET is_portfolio = 0 WHERE ticker = ?",
                (ticker,),
            )
            self.conn.commit()
            return True
        return False

    def update_position(
        self,
        ticker: str,
        shares: float,
        cost_basis: float | None = None,
        portfolio_name: str = "default",
    ) -> bool:
        portfolio: dict[str, Any] | None = self.get_portfolio(portfolio_name)
        if not portfolio:
            return False
        ticker = ticker.upper()
        if cost_basis is not None:
            cur = self.conn.execute(
                """UPDATE positions SET shares = ?, cost_basis = ?
                   WHERE portfolio_id = ? AND ticker = ?""",
                (shares, cost_basis, portfolio["id"], ticker),
            )
        else:
            cur = self.conn.execute(
                "UPDATE positions SET shares = ? WHERE portfolio_id = ? AND ticker = ?",
                (shares, portfolio["id"], ticker),
            )
        self.conn.commit()
        return cur.rowcount > 0

    def get_positions(self, portfolio_name: str = "default") -> list[dict[str, Any]]:
        portfolio: dict[str, Any] | None = self.get_portfolio(portfolio_name)
        if not portfolio:
            return []
        rows = self.conn.execute(
            """SELECT p.ticker, p.shares, p.cost_basis, p.added_at,
                      u.name, u.category, u.dividend_yield
               FROM positions p
               LEFT JOIN etf_universe u ON p.ticker = u.ticker
               WHERE p.portfolio_id = ?
               ORDER BY p.ticker""",
            (portfolio["id"],),
        ).fetchall()
        return [dict(r) for r in rows]


class UniverseManager:

    def __init__(self, conn: sqlite3.Connection | None = None) -> None:
        self.conn: sqlite3.Connection = conn or get_connection()

    def upsert_etf(
        self,
        ticker: str,
        name: str,
        asset_class: str = "dividend_etf",
        category: str | None = None,
        expense_ratio: float | None = None,
        aum: float | None = None,
        dividend_yield: float | None = None,
        is_peer: bool = False,
    ) -> None:
        now: str = datetime.now().isoformat()
        self.conn.execute(
            """INSERT INTO etf_universe (ticker, name, asset_class, category,
                                        expense_ratio, aum, dividend_yield,
                                        is_peer, discovered_at, last_updated)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(ticker) DO UPDATE SET
                   name = COALESCE(excluded.name, name),
                   asset_class = COALESCE(excluded.asset_class, asset_class),
                   category = COALESCE(excluded.category, category),
                   expense_ratio = COALESCE(excluded.expense_ratio, expense_ratio),
                   aum = COALESCE(excluded.aum, aum),
                   dividend_yield = COALESCE(excluded.dividend_yield, dividend_yield),
                   is_peer = MAX(is_peer, excluded.is_peer),
                   last_updated = excluded.last_updated""",
            (
                ticker.upper(),
                name,
                asset_class,
                category,
                expense_ratio,
                aum,
                dividend_yield,
                int(is_peer),
                now,
                now,
            ),
        )
        self.conn.commit()

    def get_all_tickers(self) -> list[str]:
        rows = self.conn.execute(
            "SELECT ticker FROM etf_universe ORDER BY ticker"
        ).fetchall()
        return [r["ticker"] for r in rows]

    def get_portfolio_tickers(self) -> list[str]:
        rows = self.conn.execute(
            "SELECT ticker FROM etf_universe WHERE is_portfolio = 1 ORDER BY ticker"
        ).fetchall()
        return [r["ticker"] for r in rows]

    def get_peer_tickers(self) -> list[str]:
        rows = self.conn.execute(
            "SELECT ticker FROM etf_universe WHERE is_peer = 1 ORDER BY ticker"
        ).fetchall()
        return [r["ticker"] for r in rows]

    def get_universe(self) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM etf_universe ORDER BY is_portfolio DESC, ticker"
        ).fetchall()
        return [dict(r) for r in rows]

    def seed_from_config(self, etf_dict: dict[str, str]) -> int:
        count: int = 0
        for ticker, name in etf_dict.items():
            self.conn.execute(
                """INSERT INTO etf_universe (ticker, name, last_updated)
                   VALUES (?, ?, datetime('now'))
                   ON CONFLICT(ticker) DO UPDATE SET
                       name = COALESCE(excluded.name, name),
                       last_updated = datetime('now')""",
                (ticker.upper(), name),
            )
            count += 1
        self.conn.commit()
        return count


class PriceHistoryManager:

    def __init__(self, conn: sqlite3.Connection | None = None) -> None:
        self.conn: sqlite3.Connection = conn or get_connection()

    def upsert_prices(self, ticker: str, rows: list[dict[str, Any]]) -> int:
        """Insert/update price rows. Each dict: date, open, high, low, close, volume, dividends."""
        ticker = ticker.upper()
        count: int = 0
        for row in rows:
            self.conn.execute(
                """INSERT INTO price_history (ticker, date, open, high, low, close, volume, dividends)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(ticker, date) DO UPDATE SET
                       open = excluded.open, high = excluded.high,
                       low = excluded.low, close = excluded.close,
                       volume = excluded.volume, dividends = excluded.dividends""",
                (
                    ticker,
                    row["date"],
                    row.get("open"),
                    row.get("high"),
                    row.get("low"),
                    row["close"],
                    row.get("volume"),
                    row.get("dividends", 0),
                ),
            )
            count += 1
        self.conn.commit()
        return count

    def get_prices(
        self,
        ticker: str,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> list[dict[str, Any]]:
        query: str = "SELECT * FROM price_history WHERE ticker = ?"
        params: list[str] = [ticker.upper()]
        if start_date:
            query += " AND date >= ?"
            params.append(start_date)
        if end_date:
            query += " AND date <= ?"
            params.append(end_date)
        query += " ORDER BY date"
        rows = self.conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    def get_latest_date(self, ticker: str) -> str | None:
        row = self.conn.execute(
            "SELECT MAX(date) as max_date FROM price_history WHERE ticker = ?",
            (ticker.upper(),),
        ).fetchone()
        return row["max_date"] if row else None

    def stats(self) -> dict[str, dict[str, Any]]:
        rows = self.conn.execute("""SELECT ticker, COUNT(*) as count,
                      MIN(date) as first_date, MAX(date) as last_date
               FROM price_history GROUP BY ticker ORDER BY ticker""").fetchall()
        return {r["ticker"]: dict(r) for r in rows}


class MacroSeriesManager:

    def __init__(self, conn: sqlite3.Connection | None = None) -> None:
        self.conn: sqlite3.Connection = conn or get_connection()

    def upsert_series(self, series_id: str, rows: list[dict[str, Any]]) -> int:
        count: int = 0
        for row in rows:
            self.conn.execute(
                """INSERT INTO macro_series (series_id, date, value)
                   VALUES (?, ?, ?)
                   ON CONFLICT(series_id, date) DO UPDATE SET value = excluded.value""",
                (series_id, row["date"], row["value"]),
            )
            count += 1
        self.conn.commit()
        return count

    def get_series(
        self,
        series_id: str,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> list[dict[str, Any]]:
        query: str = "SELECT * FROM macro_series WHERE series_id = ?"
        params: list[str] = [series_id]
        if start_date:
            query += " AND date >= ?"
            params.append(start_date)
        if end_date:
            query += " AND date <= ?"
            params.append(end_date)
        query += " ORDER BY date"
        rows = self.conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    def stats(self) -> dict[str, dict[str, Any]]:
        rows = self.conn.execute("""SELECT series_id, COUNT(*) as count,
                      MIN(date) as first_date, MAX(date) as last_date
               FROM macro_series GROUP BY series_id ORDER BY series_id""").fetchall()
        return {r["series_id"]: dict(r) for r in rows}
