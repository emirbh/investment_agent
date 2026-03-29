from __future__ import annotations

import csv
import logging
import os
import sqlite3
from datetime import datetime
from typing import Any

from config import REPORT_DIR
from db.connection import get_connection
from db.schema import init_db

logger: logging.Logger = logging.getLogger(__name__)


def generate_report(
    predictions: list[dict[str, Any]] | None = None,
    backtest_results: dict[str, Any] | None = None,
    portfolio_name: str = "default",
    output_dir: str | None = None,
) -> dict[str, str]:
    init_db()
    conn: sqlite3.Connection = get_connection()
    output_dir = output_dir or REPORT_DIR
    os.makedirs(output_dir, exist_ok=True)

    run_date: str = datetime.now().strftime("%Y-%m-%d")
    paths: dict[str, str] = {}

    if predictions is None:
        predictions = _load_latest_predictions(conn)
    if backtest_results is None:
        backtest_results = _load_latest_backtest(conn)

    portfolio: list[dict[str, Any]] = _load_portfolio(conn, portfolio_name)
    macro_summary: dict[str, dict[str, Any]] = _load_macro_summary(conn)

    md_path: str = os.path.join(output_dir, f"{run_date}_weekly_report.md")
    md_content: str = _build_markdown(
        run_date, predictions, backtest_results, portfolio, macro_summary
    )
    with open(md_path, "w") as f:
        f.write(md_content)
    paths["markdown"] = md_path
    logger.info("[Report] Markdown: %s", md_path)

    if predictions:
        csv_path: str = os.path.join(output_dir, f"{run_date}_predictions.csv")
        _write_predictions_csv(csv_path, predictions)
        paths["csv"] = csv_path
        logger.info("[Report] CSV: %s", csv_path)

    try:
        import subprocess

        pdf_path: str = os.path.join(output_dir, f"{run_date}_weekly_report.pdf")
        result = subprocess.run(
            ["pandoc", md_path, "-o", pdf_path, "--pdf-engine=wkhtmltopdf"],
            capture_output=True,
            timeout=30,
        )
        if result.returncode == 0:
            paths["pdf"] = pdf_path
            logger.info("[Report] PDF: %s", pdf_path)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    conn.close()
    return paths


def _load_latest_predictions(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    row = conn.execute("SELECT MAX(run_date) as d FROM predictions").fetchone()
    if not row or not row["d"]:
        return []
    rows = conn.execute(
        "SELECT * FROM predictions WHERE run_date = ? ORDER BY predicted_rank",
        (row["d"],),
    ).fetchall()
    return [dict(r) for r in rows]


def _load_latest_backtest(conn: sqlite3.Connection) -> dict[str, Any] | None:
    row = conn.execute(
        "SELECT * FROM backtest_results ORDER BY run_date DESC LIMIT 1"
    ).fetchone()
    return dict(row) if row else None


def _load_portfolio(conn: sqlite3.Connection, name: str) -> list[dict[str, Any]]:
    rows = conn.execute(
        """SELECT p.ticker, p.shares, p.cost_basis,
                  u.name, u.category, u.dividend_yield
           FROM positions p
           JOIN portfolio pf ON p.portfolio_id = pf.id
           LEFT JOIN etf_universe u ON p.ticker = u.ticker
           WHERE pf.name = ?
           ORDER BY p.ticker""",
        (name,),
    ).fetchall()
    return [dict(r) for r in rows]


def _load_macro_summary(conn: sqlite3.Connection) -> dict[str, dict[str, Any]]:
    rows = conn.execute("""SELECT series_id, value, date FROM macro_series m1
           WHERE date = (
               SELECT MAX(date) FROM macro_series m2
               WHERE m2.series_id = m1.series_id
           )
           ORDER BY series_id""").fetchall()
    return {r["series_id"]: {"value": r["value"], "date": r["date"]} for r in rows}


def _build_markdown(
    run_date: str,
    predictions: list[dict[str, Any]],
    backtest: dict[str, Any] | None,
    portfolio: list[dict[str, Any]],
    macro: dict[str, dict[str, Any]],
) -> str:
    lines: list[str] = []
    lines.append("# Weekly Investment Report")
    lines.append(f"**Date:** {run_date}\n")

    lines.append("## Executive Summary\n")
    if predictions:
        top_buys: list[dict[str, Any]] = [
            p for p in predictions if p.get("action") in ("BUY", "OVERWEIGHT")
        ][:3]
        top_sells: list[dict[str, Any]] = [
            p for p in predictions if p.get("action") in ("SELL", "UNDERWEIGHT")
        ][:3]
        if top_buys:
            lines.append("**Top Recommendations:**")
            for p in top_buys:
                lines.append(
                    f"- **{p['action']}** {p['ticker']} "
                    f"(predicted return: {p['predicted_ret']:.2%})"
                )
        if top_sells:
            lines.append("\n**Consider Reducing:**")
            for p in top_sells:
                lines.append(
                    f"- **{p['action']}** {p['ticker']} "
                    f"(predicted return: {p['predicted_ret']:.2%})"
                )
    else:
        lines.append(
            "No predictions available yet. "
            "Run the full pipeline to generate recommendations.\n"
        )
    lines.append("")

    lines.append("## Portfolio Snapshot\n")
    if portfolio:
        lines.append("| Ticker | Name | Shares | Cost Basis | Div Yield |")
        lines.append("|--------|------|-------:|----------:|----------:|")
        for p in portfolio:
            name: str = p.get("name") or p["ticker"]
            cost: str = f"${p['cost_basis']:.2f}" if p.get("cost_basis") else "—"
            dy: str = f"{p['dividend_yield']:.2f}%" if p.get("dividend_yield") else "—"
            lines.append(
                f"| {p['ticker']} | {name} | {p['shares']:.0f} | {cost} | {dy} |"
            )
    else:
        lines.append("No portfolio positions.\n")
    lines.append("")

    lines.append("## ML Predictions (Ranked)\n")
    if predictions:
        lines.append("| Rank | Ticker | Action | Predicted Return | Confidence |")
        lines.append("|-----:|--------|--------|----------------:|-----------:|")
        for p in predictions:
            lines.append(
                f"| {p['predicted_rank']} | {p['ticker']} | {p['action']} | "
                f"{p['predicted_ret']:.2%} | {p.get('confidence', 0):.0%} |"
            )
    else:
        lines.append("No predictions available.\n")
    lines.append("")

    lines.append("## Backtest Results\n")
    if backtest:
        lines.append(
            f"**Period:** {backtest.get('period_start', '?')} "
            f"→ {backtest.get('period_end', '?')}\n"
        )
        lines.append("| Metric | Strategy | Baseline |")
        lines.append("|--------|--------:|---------:|")
        sr: float = backtest.get("strategy_return", 0)
        br: float = backtest.get("baseline_return", 0)
        lines.append(f"| Cumulative Return | {sr:.2%} | {br:.2%} |")
        ss: float = backtest.get("strategy_sharpe", 0) or 0
        bs: float = backtest.get("baseline_sharpe", 0) or 0
        lines.append(f"| Sharpe Ratio | {ss:.4f} | {bs:.4f} |")
        md: float = backtest.get("max_drawdown", 0) or 0
        lines.append(f"| Max Drawdown | {md:.2%} | — |")
    else:
        lines.append("No backtest results available.\n")
    lines.append("")

    lines.append("## Macro Environment\n")
    if macro:
        from config import FRED_SERIES

        lines.append("| Indicator | Latest Value | Date |")
        lines.append("|-----------|------------:|----- |")
        for series_id, info in macro.items():
            desc: str = FRED_SERIES.get(series_id, series_id)
            lines.append(f"| {desc} | {info['value']:.4f} | {info['date']} |")
    else:
        lines.append("No macro data available.\n")
    lines.append("")

    lines.append("---")
    lines.append("*Generated by Investment Portfolio Optimizer*\n")

    return "\n".join(lines)


def _write_predictions_csv(path: str, predictions: list[dict[str, Any]]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "predicted_rank",
                "ticker",
                "action",
                "predicted_ret",
                "confidence",
            ],
        )
        writer.writeheader()
        for p in predictions:
            writer.writerow(
                {
                    "predicted_rank": p.get("predicted_rank"),
                    "ticker": p.get("ticker"),
                    "action": p.get("action"),
                    "predicted_ret": f"{p.get('predicted_ret', 0):.6f}",
                    "confidence": f"{p.get('confidence', 0):.4f}",
                }
            )
