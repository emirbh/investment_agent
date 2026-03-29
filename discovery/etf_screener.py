"""
ETF peer discovery — finds dividend-focused ETFs automatically.

Scans a curated seed list of 40+ well-known dividend ETFs across categories
(high yield, dividend growth, covered call, preferred stock, international,
REITs, utilities). For each seed, fetches metadata from yfinance, filters
by minimum AUM and dividend yield, classifies into strategy categories,
and returns the qualified set for universe expansion.
"""

from __future__ import annotations

import logging
import time
from typing import Any

logger: logging.Logger = logging.getLogger(__name__)

DIVIDEND_ETF_SEEDS: list[str] = [
    # High dividend yield
    "VYM",
    "SCHD",
    "DVY",
    "HDV",
    "SPHD",
    "FDL",
    "DHS",
    "SPYD",
    # Dividend growth
    "DGRO",
    "VIG",
    "DGRW",
    "RDVY",
    "TDVG",
    # Dividend aristocrats / quality
    "SDY",
    "NOBL",
    "KNG",
    "WDIV",
    # Covered call / enhanced income
    "JEPI",
    "JEPQ",
    "XYLD",
    "QYLD",
    "DIVO",
    "SVOL",
    # Preferred stock
    "PFF",
    "PGX",
    "PFFD",
    "PSK",
    # International dividend
    "VYMI",
    "IDV",
    "DWX",
    "SDIV",
    "EFAV",
    # REITs (dividend-adjacent)
    "VNQ",
    "SCHH",
    "XLRE",
    # Utilities (high dividend sector)
    "XLU",
    "VPU",
]


def _get_etf_info(ticker: str) -> dict[str, Any] | None:
    try:
        import yfinance as yf

        t = yf.Ticker(ticker)
        info: dict[str, Any] = t.info
        if not info or not info.get("longName"):
            return None

        quote_type: str = info.get("quoteType", "").upper()
        if quote_type not in ("ETF", "MUTUALFUND"):
            return None

        div_yield: float | None = (
            info.get("trailingAnnualDividendYield")
            or info.get("dividendYield")
            or info.get("yield")
        )

        return {
            "ticker": ticker.upper(),
            "name": info.get("longName") or info.get("shortName") or ticker,
            "category": info.get("category") or "",
            "expense_ratio": info.get("annualReportExpenseRatio"),
            "aum": info.get("totalAssets"),
            "dividend_yield": round(div_yield * 100, 4) if div_yield else None,
        }
    except Exception:
        return None


def _classify_category(name: str, category: str) -> str | None:
    combined: str = (name + " " + category).lower()

    if any(k in combined for k in ["covered call", "buy-write", "premium income"]):
        return "covered_call"
    if any(k in combined for k in ["preferred"]):
        return "preferred_stock"
    if any(
        k in combined
        for k in ["international", "foreign", "global", "world", "emerging"]
    ):
        return "international_dividend"
    if any(k in combined for k in ["reit", "real estate"]):
        return "reit"
    if any(k in combined for k in ["utilities", "utility"]):
        return "utilities"
    if any(k in combined for k in ["aristocrat", "growth", "grower"]):
        return "dividend_growth"
    if any(k in combined for k in ["dividend", "income", "yield", "high div"]):
        return "high_yield"
    return None


def discover_dividend_etfs(
    min_aum: float = 100_000_000,
    min_yield: float = 0.5,
    delay: float = 0.5,
) -> list[dict[str, Any]]:
    discovered: dict[str, dict[str, Any]] = {}
    seeds: list[str] = list(set(DIVIDEND_ETF_SEEDS))

    logger.info("[Discovery] Scanning %d seed ETFs...", len(seeds))

    for i, ticker in enumerate(seeds):
        info: dict[str, Any] | None = _get_etf_info(ticker)
        if info is None:
            continue

        if info["aum"] and info["aum"] < min_aum:
            continue

        if info["dividend_yield"] is not None and info["dividend_yield"] < min_yield:
            continue

        cat: str | None = _classify_category(info["name"], info["category"])
        if cat:
            info["category"] = cat
        elif info["dividend_yield"] and info["dividend_yield"] >= 1.0:
            info["category"] = "high_yield"
        else:
            continue

        discovered[ticker] = info

        if (i + 1) % 10 == 0:
            logger.info(
                "  ... scanned %d/%d, found %d so far",
                i + 1,
                len(seeds),
                len(discovered),
            )

        time.sleep(delay)

    logger.info(
        "[Discovery] Found %d dividend ETFs (min AUM: $%s, min yield: %s%%).",
        len(discovered),
        f"{min_aum:,.0f}",
        min_yield,
    )
    return list(discovered.values())
