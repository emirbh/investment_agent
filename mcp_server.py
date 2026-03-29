"""
MCP Server for the Investment Portfolio Optimizer.

Exposes the full pipeline as tools that any MCP-compatible client (Claude Desktop,
Claude Code, etc.) can call conversationally. Each tool maps to a pipeline step
or portfolio management operation.

Run:
  python mcp_server.py              # stdio transport (default for Claude Desktop)
  python mcp_server.py --sse        # SSE transport on port 8000
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Any

from mcp.server.fastmcp import FastMCP

from db.schema import init_db

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger: logging.Logger = logging.getLogger(__name__)

mcp = FastMCP(
    "investment-agent",
    instructions=(
        "Investment Portfolio Optimizer — manages dividend ETF portfolios, "
        "collects market data, trains LSTM models, generates predictions, "
        "runs backtests, and produces weekly reports. "
        "Start with 'portfolio_show' to see current holdings, "
        "'stats' to check data coverage, or 'pipeline' to run everything end-to-end."
    ),
)


@mcp.tool()
def portfolio_show(portfolio_name: str = "default") -> str:
    """Show all positions in a portfolio with shares, cost basis, and names."""
    init_db()
    from db.models import PortfolioManager

    mgr = PortfolioManager()
    positions: list[dict[str, Any]] = mgr.get_positions(portfolio_name)
    if not positions:
        return f"Portfolio '{portfolio_name}' is empty. Use portfolio_add to add positions."
    return json.dumps(positions, indent=2, default=str)


@mcp.tool()
def portfolio_add(
    ticker: str,
    shares: float,
    cost_basis: float | None = None,
    portfolio_name: str = "default",
) -> str:
    """Add shares of an ETF to a portfolio. Creates the portfolio if it doesn't exist."""
    init_db()
    from db.models import PortfolioManager

    mgr = PortfolioManager()
    mgr.add_position(ticker, shares, cost_basis, portfolio_name)
    return f"Added {shares} shares of {ticker.upper()} to '{portfolio_name}'."


@mcp.tool()
def portfolio_remove(ticker: str, portfolio_name: str = "default") -> str:
    """Remove an ETF position entirely from a portfolio."""
    init_db()
    from db.models import PortfolioManager

    mgr = PortfolioManager()
    if mgr.remove_position(ticker, portfolio_name):
        return f"Removed {ticker.upper()} from '{portfolio_name}'."
    return f"{ticker.upper()} not found in '{portfolio_name}'."


@mcp.tool()
def portfolio_update(
    ticker: str,
    shares: float,
    cost_basis: float | None = None,
    portfolio_name: str = "default",
) -> str:
    """Update the share count (and optionally cost basis) for an existing position."""
    init_db()
    from db.models import PortfolioManager

    mgr = PortfolioManager()
    if mgr.update_position(ticker, shares, cost_basis, portfolio_name):
        return f"Updated {ticker.upper()} to {shares} shares."
    return f"{ticker.upper()} not found in '{portfolio_name}'."


@mcp.tool()
def universe_show() -> str:
    """List all ETFs in the investment universe with their metadata."""
    init_db()
    from db.models import UniverseManager

    mgr = UniverseManager()
    universe: list[dict[str, Any]] = mgr.get_universe()
    if not universe:
        return "Universe is empty. Use universe_seed to populate from config."
    return json.dumps(universe, indent=2, default=str)


@mcp.tool()
def universe_seed() -> str:
    """Seed the ETF universe from the built-in config of 16 dividend ETFs."""
    init_db()
    from config import DIVIDEND_ETFS
    from db.models import UniverseManager

    mgr = UniverseManager()
    count: int = mgr.seed_from_config(DIVIDEND_ETFS)
    return f"Seeded {count} ETFs from config into universe."


@mcp.tool()
def discover(min_aum: float = 100_000_000) -> str:
    """
    Auto-discover dividend ETFs by scanning 40+ seed tickers via yfinance.
    Filters by minimum AUM and dividend yield, classifies by strategy category,
    and adds qualifying ETFs to the universe as peers.
    """
    init_db()
    from discovery.etf_screener import discover_dividend_etfs
    from discovery.universe import refresh_universe

    etfs: list[dict[str, Any]] = discover_dividend_etfs(min_aum=min_aum)
    added: int = refresh_universe(etfs)
    return json.dumps(
        {
            "discovered": len(etfs),
            "new_additions": added,
            "etfs": etfs,
        },
        indent=2,
        default=str,
    )


@mcp.tool()
def collect(tickers: list[str] | None = None) -> str:
    """
    Collect price data (via yfinance) and macro data (via FRED) into SQLite.
    Pass specific tickers or leave empty to collect for the full universe.
    """
    from pipeline.collector import collect_all

    results: dict[str, dict[str, int]] = collect_all(
        [t.upper() for t in tickers] if tickers else None
    )
    n_prices: int = sum(results["prices"].values())
    n_macro: int = sum(results["macro"].values())
    return json.dumps(
        {
            "total_price_rows": n_prices,
            "total_macro_rows": n_macro,
            "per_ticker": results["prices"],
            "per_series": results["macro"],
        },
        indent=2,
        default=str,
    )


@mcp.tool()
def featurize(tickers: list[str] | None = None) -> str:
    """
    Compute ML feature vectors (24 features per ticker per day) from raw
    price and macro data. Features include returns, volatility, RSI, SMA
    ratios, dividend yield, and macro indicators.
    """
    from pipeline.features import compute_features

    count: int = compute_features([t.upper() for t in tickers] if tickers else None)
    return f"Computed {count} feature rows."


@mcp.tool()
def train(
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    patience: int = 7,
    resume: bool = False,
) -> str:
    """
    Train the LSTM + Temporal Attention model on feature data.
    Uses MPS (Apple Silicon GPU) when available, falls back to CUDA or CPU.
    Supports early stopping and checkpoint resumption.
    """
    from ml.train import train_model

    result: dict[str, Any] = train_model(
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        patience=patience,
        resume=resume,
    )
    return json.dumps(result, indent=2, default=str)


@mcp.tool()
def predict(tickers: list[str] | None = None) -> str:
    """
    Generate ranked predictions for all universe tickers (or specific ones).
    Each ticker gets a predicted forward return, rank, action (BUY/HOLD/SELL),
    and confidence score. Results are persisted to the predictions table.
    """
    from ml.predict import generate_predictions

    predictions: list[dict[str, Any]] = generate_predictions(
        tickers=[t.upper() for t in tickers] if tickers else None
    )
    if not predictions:
        return (
            "No predictions generated. Ensure model is trained and feature data exists."
        )
    return json.dumps(predictions, indent=2, default=str)


@mcp.tool()
def backtest(
    lookback_weeks: int = 13,
    top_n: int = 5,
    portfolio_name: str = "default",
) -> str:
    """
    Run a walk-forward backtest over the trailing period. Each week the strategy
    picks the top-N tickers by model prediction and equal-weights them. Compares
    against an equal-weight baseline of your portfolio positions. Returns
    cumulative return, Sharpe ratio, max drawdown, and win rate for both.
    """
    from backtest.engine import run_backtest

    results: dict[str, Any] = run_backtest(
        lookback_weeks=lookback_weeks,
        top_n=top_n,
        portfolio_name=portfolio_name,
    )
    return json.dumps(results, indent=2, default=str)


@mcp.tool()
def report(portfolio_name: str = "default", output_dir: str | None = None) -> str:
    """
    Generate a weekly investment report (Markdown + CSV, optionally PDF).
    Includes executive summary, portfolio snapshot, ML predictions table,
    backtest results, and macro environment overview.
    """
    from report.generator import generate_report

    paths: dict[str, str] = generate_report(
        portfolio_name=portfolio_name,
        output_dir=output_dir,
    )
    if not paths:
        return "No report generated. Run the pipeline first to produce data."
    return json.dumps({"files": paths}, indent=2, default=str)


@mcp.tool()
def pipeline(
    tickers: list[str] | None = None,
    skip_training: bool = False,
    portfolio_name: str = "default",
) -> str:
    """
    Run the full weekly pipeline end-to-end: collect data → compute features →
    train model → generate predictions → run backtest → generate report.
    This is the single command to keep everything up to date.
    """
    from scheduler.jobs import run_pipeline

    results: dict[str, Any] = run_pipeline(
        tickers=[t.upper() for t in tickers] if tickers else None,
        skip_training=skip_training,
        portfolio_name=portfolio_name,
    )
    return json.dumps(results, indent=2, default=str)


@mcp.tool()
def stats() -> str:
    """Show database coverage: number of tickers, price rows, macro series, and date ranges."""
    init_db()
    from db.models import MacroSeriesManager, PriceHistoryManager

    price_mgr = PriceHistoryManager()
    macro_mgr = MacroSeriesManager()
    price_stats: dict[str, dict[str, Any]] = price_mgr.stats()
    macro_stats: dict[str, dict[str, Any]] = macro_mgr.stats()
    return json.dumps(
        {
            "price_history": {
                "tickers": len(price_stats),
                "total_rows": sum(s["count"] for s in price_stats.values()),
                "per_ticker": price_stats,
            },
            "macro_series": {
                "series": len(macro_stats),
                "total_rows": sum(s["count"] for s in macro_stats.values()),
                "per_series": macro_stats,
            },
        },
        indent=2,
        default=str,
    )


@mcp.tool()
def live_price(ticker: str) -> str:
    """Get the current live price, dividend yield, and key stats for a single ETF."""
    from data.yfinance_loader import get_live_price

    data: dict[str, Any] = get_live_price(ticker)
    return json.dumps(data, indent=2, default=str)


if __name__ == "__main__":
    if "--sse" in sys.argv:
        mcp.run(transport="sse")
    else:
        mcp.run()
