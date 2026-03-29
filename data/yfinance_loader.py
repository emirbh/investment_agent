from __future__ import annotations

import logging
from typing import Any

from config import END_DATE, START_DATE

logger: logging.Logger = logging.getLogger(__name__)


def load_yfinance_price_rows(ticker: str) -> list[dict[str, Any]]:
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("Install yfinance: pip install yfinance")

    ticker = ticker.upper()
    t = yf.Ticker(ticker)
    rows: list[dict[str, Any]] = []

    try:
        hist = t.history(start=START_DATE, end=END_DATE)
        if hist.empty:
            return rows

        for date_idx, row in hist.iterrows():
            rows.append(
                {
                    "date": date_idx.strftime("%Y-%m-%d"),
                    "open": (
                        round(float(row.get("Open", 0)), 4) if row.get("Open") else None
                    ),
                    "high": (
                        round(float(row.get("High", 0)), 4) if row.get("High") else None
                    ),
                    "low": (
                        round(float(row.get("Low", 0)), 4) if row.get("Low") else None
                    ),
                    "close": round(float(row["Close"]), 4),
                    "volume": (
                        int(row.get("Volume", 0)) if row.get("Volume") else None
                    ),
                    "dividends": round(float(row.get("Dividends", 0)), 6),
                }
            )
    except Exception as e:
        logger.warning("[yfinance] Price rows failed for %s: %s", ticker, e)

    return rows


def get_live_price(ticker: str) -> dict[str, Any]:
    import yfinance as yf

    t = yf.Ticker(ticker.upper())
    info: dict[str, Any] = t.info
    div_yield: float | None = (
        info.get("trailingAnnualDividendYield")
        or info.get("dividendYield")
        or info.get("yield")
    )
    return {
        "ticker": ticker.upper(),
        "current_price": info.get("currentPrice") or info.get("regularMarketPrice"),
        "previous_close": info.get("previousClose"),
        "day_high": info.get("dayHigh"),
        "day_low": info.get("dayLow"),
        "volume": info.get("volume"),
        "total_assets": info.get("totalAssets") or info.get("marketCap"),
        "dividend_yield_pct": round(div_yield * 100, 2) if div_yield else None,
        "annual_dividend_rate": info.get("trailingAnnualDividendRate")
        or info.get("dividendRate"),
        "ex_dividend_date": info.get("exDividendDate"),
        "last_dividend_value": info.get("lastDividendValue"),
    }
