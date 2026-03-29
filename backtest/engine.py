"""
Walk-forward backtesting engine.

Simulates weekly rebalancing over a trailing period. Each week the strategy
picks the top-N tickers by model-predicted return and equal-weights them.
The baseline is an equal-weight buy-and-hold of the user's portfolio positions.
Aggregate metrics (cumulative return, Sharpe, max drawdown, win rate) are
computed over the full lookback window and persisted for reporting.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timedelta
from typing import Any

from backtest.metrics import (
    annualized_return,
    cumulative_return,
    max_drawdown,
    sharpe_ratio,
    win_rate,
)
from db.connection import get_connection
from db.schema import init_db

logger: logging.Logger = logging.getLogger(__name__)


def _get_weekly_dates(start: str, end: str) -> list[str]:
    dates: list[str] = []
    d: datetime = datetime.strptime(start, "%Y-%m-%d")
    end_d: datetime = datetime.strptime(end, "%Y-%m-%d")
    while d.weekday() != 0:
        d += timedelta(days=1)
    while d <= end_d:
        dates.append(d.strftime("%Y-%m-%d"))
        d += timedelta(days=7)
    return dates


def _get_return_between(
    conn: sqlite3.Connection,
    ticker: str,
    start_date: str,
    end_date: str,
) -> float | None:
    start_row = conn.execute(
        "SELECT close FROM price_history "
        "WHERE ticker = ? AND date >= ? ORDER BY date ASC LIMIT 1",
        (ticker, start_date),
    ).fetchone()
    end_row = conn.execute(
        "SELECT close FROM price_history "
        "WHERE ticker = ? AND date <= ? ORDER BY date DESC LIMIT 1",
        (ticker, end_date),
    ).fetchone()

    if not start_row or not end_row:
        return None
    if start_row["close"] <= 0:
        return None
    return (end_row["close"] - start_row["close"]) / start_row["close"]


def run_backtest(
    lookback_weeks: int = 13,
    top_n: int = 5,
    portfolio_name: str = "default",
) -> dict[str, Any]:
    init_db()
    conn: sqlite3.Connection = get_connection()

    latest = conn.execute("SELECT MAX(date) as d FROM price_history").fetchone()
    if not latest or not latest["d"]:
        logger.warning("[Backtest] No price data available.")
        return {"status": "no_data"}

    end_date: str = latest["d"]
    start_d: datetime = datetime.strptime(end_date, "%Y-%m-%d") - timedelta(
        weeks=lookback_weeks
    )
    start_date: str = start_d.strftime("%Y-%m-%d")

    portfolio_rows = conn.execute(
        """SELECT p.ticker FROM positions p
           JOIN portfolio pf ON p.portfolio_id = pf.id
           WHERE pf.name = ?""",
        (portfolio_name,),
    ).fetchall()
    baseline_tickers: list[str] = [r["ticker"] for r in portfolio_rows]

    if not baseline_tickers:
        logger.warning(
            "[Backtest] No portfolio positions found. Add positions first."
        )
        return {"status": "no_portfolio"}

    weeks: list[str] = _get_weekly_dates(start_date, end_date)
    if len(weeks) < 2:
        logger.warning("[Backtest] Not enough weeks for backtesting.")
        return {"status": "insufficient_data"}

    strategy_returns: list[float] = []
    baseline_returns: list[float] = []
    per_period: list[dict[str, Any]] = []

    for i in range(len(weeks) - 1):
        week_start: str = weeks[i]
        week_end: str = weeks[i + 1]

        preds = conn.execute(
            """SELECT ticker, predicted_ret FROM predictions
               WHERE run_date <= ? ORDER BY predicted_ret DESC""",
            (week_start,),
        ).fetchall()

        if preds:
            strategy_tickers: list[str] = [r["ticker"] for r in preds[:top_n]]
        else:
            strategy_tickers = baseline_tickers[:top_n]

        strat_rets: list[float] = []
        for t in strategy_tickers:
            ret: float | None = _get_return_between(conn, t, week_start, week_end)
            if ret is not None:
                strat_rets.append(ret)

        base_rets: list[float] = []
        for t in baseline_tickers:
            ret = _get_return_between(conn, t, week_start, week_end)
            if ret is not None:
                base_rets.append(ret)

        strat_week: float = sum(strat_rets) / len(strat_rets) if strat_rets else 0.0
        base_week: float = sum(base_rets) / len(base_rets) if base_rets else 0.0

        strategy_returns.append(strat_week)
        baseline_returns.append(base_week)

        per_period.append(
            {
                "week_start": week_start,
                "week_end": week_end,
                "strategy_return": round(strat_week, 6),
                "baseline_return": round(base_week, 6),
                "strategy_tickers": strategy_tickers,
                "baseline_tickers": baseline_tickers,
            }
        )

    strat_cum: float = cumulative_return(strategy_returns)
    base_cum: float = cumulative_return(baseline_returns)
    n_periods: int = len(strategy_returns)

    results: dict[str, Any] = {
        "status": "completed",
        "period_start": start_date,
        "period_end": end_date,
        "weeks": n_periods,
        "strategy": {
            "cumulative_return": round(strat_cum, 6),
            "annualized_return": round(annualized_return(strat_cum, n_periods), 6),
            "sharpe_ratio": round(sharpe_ratio(strategy_returns), 4),
            "max_drawdown": round(max_drawdown(strategy_returns), 6),
        },
        "baseline": {
            "cumulative_return": round(base_cum, 6),
            "annualized_return": round(annualized_return(base_cum, n_periods), 6),
            "sharpe_ratio": round(sharpe_ratio(baseline_returns), 4),
            "max_drawdown": round(max_drawdown(baseline_returns), 6),
        },
        "win_rate": round(win_rate(strategy_returns, baseline_returns), 4),
        "per_period": per_period,
    }

    run_date: str = datetime.now().strftime("%Y-%m-%d")
    conn.execute(
        """INSERT INTO backtest_results
           (run_date, period_start, period_end, strategy_return, baseline_return,
            strategy_sharpe, baseline_sharpe, max_drawdown, details)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            run_date,
            start_date,
            end_date,
            strat_cum,
            base_cum,
            results["strategy"]["sharpe_ratio"],
            results["baseline"]["sharpe_ratio"],
            results["strategy"]["max_drawdown"],
            json.dumps(per_period),
        ),
    )
    conn.commit()
    conn.close()

    return results
