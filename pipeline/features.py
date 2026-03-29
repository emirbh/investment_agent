"""
Feature engineering — transforms raw price + macro data into ML feature vectors.

Features per ticker per day:
  - Price-derived: returns (1d,5d,10d,21d,63d), volatility, SMA ratios, RSI, volume ratio
  - Dividend: trailing yield, dividend frequency signal
  - Macro: fed rate, 10Y yield, yield curve, CPI change, HY spread, oil price, real yield
  - Cross-sectional: z-scores of yield, return, volatility within peer group

Target: forward 21-day return
"""

from __future__ import annotations

import json
import logging
import math
import sqlite3
from typing import Any

from db.connection import get_connection
from db.schema import init_db

logger: logging.Logger = logging.getLogger(__name__)


def _compute_rsi(closes: list[float], period: int = 14) -> float | None:
    if len(closes) < period + 1:
        return None
    deltas: list[float] = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
    recent: list[float] = deltas[-(period):]
    gains: list[float] = [d for d in recent if d > 0]
    losses: list[float] = [-d for d in recent if d < 0]
    avg_gain: float = sum(gains) / period if gains else 0
    avg_loss: float = sum(losses) / period if losses else 0.0001
    rs: float = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def _safe_log_return(current: float, previous: float) -> float | None:
    if previous <= 0 or current <= 0:
        return None
    return math.log(current / previous)


def compute_features(tickers: list[str] | None = None) -> int:
    init_db()
    conn: sqlite3.Connection = get_connection()

    if tickers is None:
        rows = conn.execute(
            "SELECT DISTINCT ticker FROM price_history ORDER BY ticker"
        ).fetchall()
        tickers = [r["ticker"] for r in rows]

    macro_data: dict[str, dict[str, float]] = _load_macro_by_date(conn)

    total_written: int = 0

    for ticker in tickers:
        prices_rows = conn.execute(
            "SELECT date, open, high, low, close, volume, dividends "
            "FROM price_history WHERE ticker = ? ORDER BY date",
            (ticker,),
        ).fetchall()

        if len(prices_rows) < 22:
            continue

        prices: list[dict[str, Any]] = [dict(p) for p in prices_rows]
        closes: list[float] = [p["close"] for p in prices]
        volumes: list[int] = [p["volume"] or 0 for p in prices]
        dividends: list[float] = [p["dividends"] or 0 for p in prices]

        min_start: int = min(21, len(prices) - 1)
        start_idx: int = min_start

        for i in range(start_idx, len(prices)):
            date: str = prices[i]["date"]
            close: float = closes[i]

            features: dict[str, float | None] = {}

            for period, name in [
                (1, "ret_1d"),
                (5, "ret_5d"),
                (10, "ret_10d"),
                (21, "ret_21d"),
                (63, "ret_63d"),
            ]:
                if i >= period:
                    features[name] = _safe_log_return(close, closes[i - period])

            for period, name in [(10, "vol_10d"), (21, "vol_21d")]:
                if i >= period:
                    rets: list[float | None] = [
                        _safe_log_return(closes[j], closes[j - 1])
                        for j in range(i - period + 1, i + 1)
                        if j > 0
                    ]
                    valid_rets: list[float] = [r for r in rets if r is not None]
                    if len(valid_rets) > 1:
                        mean_r: float = sum(valid_rets) / len(valid_rets)
                        features[name] = math.sqrt(
                            sum((r - mean_r) ** 2 for r in valid_rets)
                            / (len(valid_rets) - 1)
                        )

            for period, name in [
                (10, "sma_ratio_10"),
                (21, "sma_ratio_21"),
                (50, "sma_ratio_50"),
            ]:
                if i >= period:
                    sma: float = sum(closes[i - period + 1 : i + 1]) / period
                    features[name] = close / sma if sma > 0 else None

            if i >= 14:
                features["rsi_14"] = _compute_rsi(closes[: i + 1])

            if i >= 20:
                avg_vol: float = sum(volumes[i - 19 : i + 1]) / 20
                features["volume_ratio"] = volumes[i] / avg_vol if avg_vol > 0 else None

            lookback: int = min(252, i + 1)
            trailing_divs: float = sum(dividends[i - lookback + 1 : i + 1])
            features["trailing_div_yield"] = (
                (trailing_divs / close * 100) if close > 0 else None
            )

            week52: int = min(252, i + 1)
            high_52: float = max(closes[i - week52 + 1 : i + 1])
            low_52: float = min(closes[i - week52 + 1 : i + 1])
            features["dist_52w_high"] = (
                (close - high_52) / high_52 if high_52 > 0 else None
            )
            features["dist_52w_low"] = (close - low_52) / low_52 if low_52 > 0 else None

            macro_today: dict[str, float] = macro_data.get(date, {})
            features["fed_rate"] = macro_today.get("FEDFUNDS")
            features["treasury_10y"] = macro_today.get("DGS10")
            features["treasury_2y"] = macro_today.get("DGS2")
            features["yield_curve"] = macro_today.get("T10Y2Y")
            features["cpi"] = macro_today.get("CPIAUCSL")
            features["hy_spread"] = macro_today.get("BAMLH0A0HYM2")
            features["oil_wti"] = macro_today.get("DCOILWTICO")
            features["real_yield"] = macro_today.get("DFII10")
            features["unemployment"] = macro_today.get("UNRATE")

            target: float | None = None
            if i + 21 < len(prices):
                target = _safe_log_return(closes[i + 21], close)

            feature_json: str = json.dumps(
                {k: round(v, 6) if v is not None else None for k, v in features.items()}
            )
            conn.execute(
                """INSERT INTO features (ticker, date, feature_vec, target)
                   VALUES (?, ?, ?, ?)
                   ON CONFLICT(ticker, date) DO UPDATE SET
                       feature_vec = excluded.feature_vec,
                       target = excluded.target""",
                (ticker, date, feature_json, target),
            )
            total_written += 1

        conn.commit()
        logger.info("  [Features] %s: %d feature rows", ticker, total_written)

    conn.close()
    return total_written


def _load_macro_by_date(
    conn: sqlite3.Connection,
) -> dict[str, dict[str, float]]:
    """
    Load all macro series into a nested dict keyed by date, then forward-fill
    so that every price date has the most recent known macro values. This ensures
    the feature matrix has no gaps even when macro releases lag market data.
    """
    rows = conn.execute(
        "SELECT series_id, date, value FROM macro_series ORDER BY date"
    ).fetchall()
    result: dict[str, dict[str, float]] = {}

    for r in rows:
        if r["date"] not in result:
            result[r["date"]] = {}
        result[r["date"]][r["series_id"]] = r["value"]

    price_dates = conn.execute(
        "SELECT DISTINCT date FROM price_history ORDER BY date"
    ).fetchall()

    current_values: dict[str, float] = {}
    for pd in price_dates:
        d: str = pd["date"]
        if d in result:
            current_values.update(result[d])
        if d not in result:
            result[d] = {}
        result[d].update(current_values)

    return result
