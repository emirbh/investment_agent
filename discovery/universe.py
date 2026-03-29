"""
Universe management — refresh ETF universe from discovery + config.
"""

from __future__ import annotations

from typing import Any

from db.models import UniverseManager
from db.schema import init_db


def refresh_universe(discovered: list[dict[str, Any]]) -> int:
    init_db()
    mgr: UniverseManager = UniverseManager()
    added: int = 0

    for etf in discovered:
        existing = mgr.conn.execute(
            "SELECT ticker FROM etf_universe WHERE ticker = ?", (etf["ticker"],)
        ).fetchone()

        mgr.upsert_etf(
            ticker=etf["ticker"],
            name=etf["name"],
            asset_class="dividend_etf",
            category=etf.get("category"),
            expense_ratio=etf.get("expense_ratio"),
            aum=etf.get("aum"),
            dividend_yield=etf.get("dividend_yield"),
            is_peer=True,
        )

        if not existing:
            added += 1

    return added
