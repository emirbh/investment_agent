from __future__ import annotations

import logging
from typing import Any

from config import END_DATE, FRED_API_KEY, FRED_SERIES, START_DATE

logger: logging.Logger = logging.getLogger(__name__)


def load_fred_series_rows() -> dict[str, list[dict[str, Any]]]:
    try:
        from fredapi import Fred
    except ImportError:
        raise ImportError("Install fredapi: pip install fredapi")

    if not FRED_API_KEY:
        logger.warning("[FRED] FRED_API_KEY not set — skipping.")
        return {}

    fred = Fred(api_key=FRED_API_KEY)
    result: dict[str, list[dict[str, Any]]] = {}

    for series_id in FRED_SERIES:
        try:
            series = fred.get_series(
                series_id, observation_start=START_DATE, observation_end=END_DATE
            )
            values = series.dropna()
            if values.empty:
                continue
            rows: list[dict[str, Any]] = [
                {"date": d.strftime("%Y-%m-%d"), "value": round(float(v), 6)}
                for d, v in values.items()
            ]
            result[series_id] = rows
        except Exception as e:
            logger.warning("[FRED] Could not fetch %s: %s", series_id, e)

    logger.info("[FRED] Loaded %d macro series for SQLite.", len(result))
    return result
