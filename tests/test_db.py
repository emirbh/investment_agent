"""
Tests for the SQLite database layer.
"""

from __future__ import annotations

import pytest

from db.connection import get_connection
from db.models import (
    MacroSeriesManager,
    PortfolioManager,
    PriceHistoryManager,
    UniverseManager,
)


class TestPortfolioManager:
    def test_create_default_portfolio(self, tmp_db: str) -> None:
        mgr = PortfolioManager(get_connection(tmp_db))
        pid: int = mgr.create_portfolio("default")
        assert pid > 0

        portfolio = mgr.get_portfolio("default")
        assert portfolio is not None
        assert portfolio["name"] == "default"

    def test_add_and_get_positions(self, tmp_db: str) -> None:
        mgr = PortfolioManager(get_connection(tmp_db))
        mgr.add_position("SCHD", 100, 25.50)
        mgr.add_position("VYM", 50)

        positions = mgr.get_positions()
        assert len(positions) == 2
        tickers: list[str] = [p["ticker"] for p in positions]
        assert "SCHD" in tickers
        assert "VYM" in tickers

    def test_add_position_accumulates_shares(self, tmp_db: str) -> None:
        mgr = PortfolioManager(get_connection(tmp_db))
        mgr.add_position("SCHD", 100)
        mgr.add_position("SCHD", 50)

        positions = mgr.get_positions()
        assert len(positions) == 1
        assert positions[0]["shares"] == 150

    def test_remove_position(self, tmp_db: str) -> None:
        mgr = PortfolioManager(get_connection(tmp_db))
        mgr.add_position("SCHD", 100)
        assert mgr.remove_position("SCHD")
        assert mgr.get_positions() == []

    def test_remove_nonexistent_returns_false(self, tmp_db: str) -> None:
        mgr = PortfolioManager(get_connection(tmp_db))
        assert not mgr.remove_position("NOPE")

    def test_update_position(self, tmp_db: str) -> None:
        mgr = PortfolioManager(get_connection(tmp_db))
        mgr.add_position("SCHD", 100, 25.00)
        assert mgr.update_position("SCHD", 200, 26.00)

        positions = mgr.get_positions()
        assert positions[0]["shares"] == 200
        assert positions[0]["cost_basis"] == 26.00


class TestUniverseManager:
    def test_upsert_and_list(self, tmp_db: str) -> None:
        mgr = UniverseManager(get_connection(tmp_db))
        mgr.upsert_etf("SCHD", "Schwab US Dividend Equity ETF", category="high_yield")
        mgr.upsert_etf("VYM", "Vanguard High Dividend Yield ETF", is_peer=True)

        tickers: list[str] = mgr.get_all_tickers()
        assert "SCHD" in tickers
        assert "VYM" in tickers

    def test_seed_from_config(self, tmp_db: str) -> None:
        mgr = UniverseManager(get_connection(tmp_db))
        count: int = mgr.seed_from_config({"SCHD": "Schwab", "VYM": "Vanguard"})
        assert count == 2
        assert len(mgr.get_all_tickers()) == 2

    def test_peer_vs_portfolio(self, tmp_db: str) -> None:
        conn = get_connection(tmp_db)
        uni_mgr = UniverseManager(conn)
        port_mgr = PortfolioManager(conn)

        port_mgr.add_position("SCHD", 100)
        uni_mgr.upsert_etf("VYM", "Vanguard", is_peer=True)

        assert "SCHD" in uni_mgr.get_portfolio_tickers()
        assert "VYM" in uni_mgr.get_peer_tickers()


class TestPriceHistoryManager:
    def test_upsert_and_query(self, tmp_db: str) -> None:
        mgr = PriceHistoryManager(get_connection(tmp_db))
        rows = [
            {
                "date": "2025-01-01",
                "open": 25,
                "high": 26,
                "low": 24,
                "close": 25.5,
                "volume": 1000,
                "dividends": 0,
            },
            {
                "date": "2025-01-02",
                "open": 25.5,
                "high": 27,
                "low": 25,
                "close": 26.0,
                "volume": 1200,
                "dividends": 0.1,
            },
        ]
        count: int = mgr.upsert_prices("SCHD", rows)
        assert count == 2

        prices = mgr.get_prices("SCHD")
        assert len(prices) == 2
        assert prices[0]["close"] == 25.5

    def test_latest_date(self, tmp_db: str) -> None:
        mgr = PriceHistoryManager(get_connection(tmp_db))
        mgr.upsert_prices(
            "SCHD",
            [
                {"date": "2025-01-01", "close": 25},
                {"date": "2025-01-05", "close": 26},
            ],
        )
        assert mgr.get_latest_date("SCHD") == "2025-01-05"

    def test_date_range_filter(self, tmp_db: str) -> None:
        mgr = PriceHistoryManager(get_connection(tmp_db))
        mgr.upsert_prices(
            "SCHD",
            [
                {"date": "2025-01-01", "close": 25},
                {"date": "2025-01-02", "close": 25.5},
                {"date": "2025-01-03", "close": 26},
            ],
        )
        prices = mgr.get_prices("SCHD", start_date="2025-01-02", end_date="2025-01-02")
        assert len(prices) == 1


class TestMacroSeriesManager:
    def test_upsert_and_query(self, tmp_db: str) -> None:
        mgr = MacroSeriesManager(get_connection(tmp_db))
        rows = [
            {"date": "2025-01-01", "value": 4.5},
            {"date": "2025-01-02", "value": 4.55},
        ]
        count: int = mgr.upsert_series("DGS10", rows)
        assert count == 2

        data = mgr.get_series("DGS10")
        assert len(data) == 2

    def test_stats(self, tmp_db: str) -> None:
        mgr = MacroSeriesManager(get_connection(tmp_db))
        mgr.upsert_series("DGS10", [{"date": "2025-01-01", "value": 4.5}])
        mgr.upsert_series("FEDFUNDS", [{"date": "2025-01-01", "value": 5.0}])

        stats = mgr.stats()
        assert "DGS10" in stats
        assert "FEDFUNDS" in stats
