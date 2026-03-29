from __future__ import annotations

import logging

from db.models import MacroSeriesManager, PriceHistoryManager, UniverseManager
from db.schema import init_db
from data.fred_loader import load_fred_series_rows
from data.yfinance_loader import load_yfinance_price_rows

logger: logging.Logger = logging.getLogger(__name__)


def collect_prices(tickers: list[str] | None = None) -> dict[str, int]:
    init_db()
    price_mgr = PriceHistoryManager()
    universe_mgr = UniverseManager()

    if tickers is None:
        tickers = universe_mgr.get_all_tickers()

    results: dict[str, int] = {}
    for ticker in tickers:
        try:
            rows: list[dict[str, object]] = load_yfinance_price_rows(ticker)
            count: int = price_mgr.upsert_prices(ticker, rows)
            results[ticker] = count
            logger.info("  [Collect] %s: %d price rows", ticker, count)
        except Exception as e:
            logger.error("  [Collect] %s: FAILED - %s", ticker, e)
            results[ticker] = 0

    return results


def collect_macro() -> dict[str, int]:
    init_db()
    macro_mgr = MacroSeriesManager()

    results: dict[str, int] = {}
    try:
        series_data: dict[str, list[dict[str, object]]] = load_fred_series_rows()
        for series_id, rows in series_data.items():
            count: int = macro_mgr.upsert_series(series_id, rows)
            results[series_id] = count
    except Exception as e:
        logger.error("  [Collect] Macro FAILED: %s", e)

    return results


def collect_all(tickers: list[str] | None = None) -> dict[str, dict[str, int]]:
    init_db()

    logger.info("\n── Collecting price data ─────────────────────────────────")
    price_results: dict[str, int] = collect_prices(tickers)

    logger.info("\n── Collecting macro data ─────────────────────────────────")
    macro_results: dict[str, int] = collect_macro()

    return {
        "prices": price_results,
        "macro": macro_results,
    }
