"""
Tests for feature engineering.
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any

import pytest

from db.connection import get_connection
from db.models import MacroSeriesManager, PriceHistoryManager
from pipeline.features import _compute_rsi, _safe_log_return, compute_features


class TestHelperFunctions:
    def test_safe_log_return(self) -> None:
        assert _safe_log_return(110, 100) == pytest.approx(0.09531, abs=0.001)
        assert _safe_log_return(0, 100) is None
        assert _safe_log_return(100, 0) is None

    def test_rsi_basic(self) -> None:
        # 15 prices with steady gains = RSI near 100
        closes: list[float] = [100.0 + i for i in range(16)]
        rsi: float | None = _compute_rsi(closes)
        assert rsi is not None
        assert rsi > 90

    def test_rsi_insufficient_data(self) -> None:
        assert _compute_rsi([100.0, 101.0]) is None


class TestFeatureComputation:
    def _seed_prices(
        self, conn: sqlite3.Connection, ticker: str, n_days: int = 80
    ) -> None:
        """Seed synthetic price data."""
        mgr = PriceHistoryManager(conn)
        rows: list[dict[str, Any]] = []
        base_price: float = 25.0
        for i in range(n_days):
            day: str = f"2025-{(i // 30) + 1:02d}-{(i % 28) + 1:02d}"
            price: float = base_price + (i * 0.1)
            rows.append(
                {
                    "date": day,
                    "open": price - 0.05,
                    "high": price + 0.2,
                    "low": price - 0.2,
                    "close": price,
                    "volume": 1000000 + i * 1000,
                    "dividends": 0.25 if i % 30 == 0 else 0,
                }
            )
        mgr.upsert_prices(ticker, rows)

    def test_compute_features_writes_to_db(self, tmp_db: str) -> None:
        conn: sqlite3.Connection = get_connection(tmp_db)
        self._seed_prices(conn, "SCHD", 80)

        count: int = compute_features(["SCHD"])
        assert count > 0

        rows = conn.execute(
            "SELECT * FROM features WHERE ticker = 'SCHD' ORDER BY date"
        ).fetchall()
        assert len(rows) > 0

        # Check feature vector structure
        fv: dict[str, Any] = json.loads(rows[0]["feature_vec"])
        assert "ret_1d" in fv
        assert "vol_10d" in fv
        assert "rsi_14" in fv
        assert "trailing_div_yield" in fv

    def test_skips_ticker_with_insufficient_data(self, tmp_db: str) -> None:
        conn: sqlite3.Connection = get_connection(tmp_db)
        mgr = PriceHistoryManager(conn)
        mgr.upsert_prices(
            "SHORT",
            [
                {"date": "2025-01-01", "close": 25},
                {"date": "2025-01-02", "close": 25.5},
            ],
        )
        count: int = compute_features(["SHORT"])
        assert count == 0
