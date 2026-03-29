import argparse
import logging
import sys
from collections import Counter
from typing import Any

logger: logging.Logger = logging.getLogger(__name__)


def cmd_portfolio(args: argparse.Namespace) -> None:
    from db.models import PortfolioManager
    from db.schema import init_db

    init_db()
    mgr = PortfolioManager()

    action: str = args.action

    if action == "show":
        positions: list[dict[str, Any]] = mgr.get_positions(args.portfolio)
        if not positions:
            logger.info(
                "Portfolio '%s' is empty. "
                "Add positions with: portfolio add <TICKER> <SHARES>",
                args.portfolio,
            )
            return
        header: str = (
            f"\nPortfolio: {args.portfolio}\n"
            f"{'Ticker':<8} {'Shares':>10} {'Cost Basis':>12} "
            f"{'Name':<40} {'Added'}\n"
            f"{'─' * 90}"
        )
        logger.info(header)
        for p in positions:
            cost: str = f"${p['cost_basis']:.2f}" if p["cost_basis"] else "—"
            name: str = p["name"] or p["ticker"]
            logger.info(
                "%s %s %s %s %s",
                f"{p['ticker']:<8}",
                f"{p['shares']:>10.2f}",
                f"{cost:>12}",
                f"{name:<40}",
                p["added_at"][:10],
            )
        logger.info("\nTotal positions: %d", len(positions))

    elif action == "add":
        if not args.ticker or args.shares is None:
            logger.info("Usage: portfolio add <TICKER> <SHARES> [--cost <COST_BASIS>]")
            return
        mgr.add_position(args.ticker, args.shares, args.cost, args.portfolio)
        logger.info(
            "Added %s shares of %s to '%s'.",
            args.shares,
            args.ticker.upper(),
            args.portfolio,
        )

    elif action == "remove":
        if not args.ticker:
            logger.info("Usage: portfolio remove <TICKER>")
            return
        if mgr.remove_position(args.ticker, args.portfolio):
            logger.info("Removed %s from '%s'.", args.ticker.upper(), args.portfolio)
        else:
            logger.info(
                "%s not found in '%s'.", args.ticker.upper(), args.portfolio
            )

    elif action == "update":
        if not args.ticker or args.shares is None:
            logger.info(
                "Usage: portfolio update <TICKER> <SHARES> [--cost <COST_BASIS>]"
            )
            return
        if mgr.update_position(args.ticker, args.shares, args.cost, args.portfolio):
            logger.info(
                "Updated %s to %s shares.", args.ticker.upper(), args.shares
            )
        else:
            logger.info(
                "%s not found in '%s'.", args.ticker.upper(), args.portfolio
            )


def cmd_universe(args: argparse.Namespace) -> None:
    from config import DIVIDEND_ETFS
    from db.models import MacroSeriesManager, PriceHistoryManager, UniverseManager
    from db.schema import init_db

    init_db()

    action: str = args.action

    if action == "show":
        mgr = UniverseManager()
        universe: list[dict[str, Any]] = mgr.get_universe()
        if not universe:
            logger.info("Universe is empty. Run: universe seed")
            return
        logger.info(
            "\n%s %s %s %s %s\n%s",
            f"{'Ticker':<8}",
            f"{'Name':<45}",
            f"{'Class':<15}",
            f"{'Portfolio':>9}",
            f"{'Peer':>6}",
            "─" * 90,
        )
        for u in universe:
            port: str = "YES" if u["is_portfolio"] else ""
            peer: str = "YES" if u["is_peer"] else ""
            logger.info(
                "%s %s %s %s %s",
                f"{u['ticker']:<8}",
                f"{(u['name'] or ''):<45}",
                f"{(u['asset_class'] or ''):<15}",
                f"{port:>9}",
                f"{peer:>6}",
            )
        logger.info("\nTotal: %d ETFs", len(universe))

    elif action == "seed":
        mgr = UniverseManager()
        count: int = mgr.seed_from_config(DIVIDEND_ETFS)
        logger.info("Seeded %d ETFs from config into universe.", count)

    elif action == "stats":
        price_mgr = PriceHistoryManager()
        macro_mgr = MacroSeriesManager()
        logger.info("\nPrice history coverage:")
        for ticker, info in price_mgr.stats().items():
            logger.info(
                "  %s %s rows  (%s → %s)",
                f"{ticker:<8}",
                f"{info['count']:>5}",
                info["first_date"],
                info["last_date"],
            )
        logger.info("\nMacro series coverage:")
        for series_id, info in macro_mgr.stats().items():
            logger.info(
                "  %s %s rows  (%s → %s)",
                f"{series_id:<20}",
                f"{info['count']:>5}",
                info["first_date"],
                info["last_date"],
            )


def cmd_discover(args: argparse.Namespace) -> None:
    from db.schema import init_db
    from discovery.etf_screener import discover_dividend_etfs
    from discovery.universe import refresh_universe

    init_db()

    etfs: list[dict[str, Any]] = discover_dividend_etfs(min_aum=args.min_aum)
    added: int = refresh_universe(etfs)
    logger.info(
        "\nDiscovered %d dividend ETFs, %d new additions to universe.",
        len(etfs),
        added,
    )

    cats: Counter[str] = Counter(e.get("category", "unknown") for e in etfs)
    lines: list[str] = ["\nBy category:"]
    for cat, count in cats.most_common():
        lines.append(f"  {cat:<25} {count} ETFs")
    logger.info("\n".join(lines))


def cmd_collect(args: argparse.Namespace) -> None:
    from pipeline.collector import collect_all

    tickers: list[str] | None = (
        [t.upper() for t in args.tickers] if args.tickers else None
    )
    collect_all(tickers)
    logger.info("Collection complete.")


def cmd_featurize(args: argparse.Namespace) -> None:
    from pipeline.features import compute_features

    tickers: list[str] | None = (
        [t.upper() for t in args.tickers] if args.tickers else None
    )
    count: int = compute_features(tickers)
    logger.info("Featurization complete: %d feature rows written.", count)


def cmd_train(args: argparse.Namespace) -> None:
    from ml.train import train_model

    result: dict[str, Any] = train_model(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        patience=args.patience,
        resume=args.resume,
    )
    if result["status"] == "completed":
        logger.info(
            "Training complete: %d epochs, best val_loss=%.6f, device=%s",
            result["epochs_run"],
            result["best_val_loss"],
            result["device"],
        )
    else:
        logger.info(
            "Training: %s (%d samples)", result["status"], result.get("samples", 0)
        )


def cmd_predict(args: argparse.Namespace) -> None:
    from ml.predict import generate_predictions

    tickers: list[str] | None = (
        [t.upper() for t in args.tickers] if args.tickers else None
    )
    predictions: list[dict[str, Any]] = generate_predictions(tickers=tickers)
    if predictions:
        lines: list[str] = [
            f"\n{'Rank':<6} {'Ticker':<8} {'Action':<12} "
            f"{'Predicted Return':>16} {'Confidence':>11}",
            "─" * 60,
        ]
        for p in predictions:
            lines.append(
                f"{p['predicted_rank']:<6} {p['ticker']:<8} {p['action']:<12} "
                f"{p['predicted_ret']:>15.4%} {p['confidence']:>10.1%}"
            )
        logger.info("\n".join(lines))


def cmd_report(args: argparse.Namespace) -> None:
    from report.generator import generate_report

    paths: dict[str, str] = generate_report(
        portfolio_name=args.portfolio,
        output_dir=args.output_dir,
    )
    if paths:
        lines: list[str] = ["\nReport files generated:"]
        for fmt, path in paths.items():
            lines.append(f"  {fmt}: {path}")
        logger.info("\n".join(lines))
    else:
        logger.info("No report files generated.")


def cmd_backtest(args: argparse.Namespace) -> None:
    from backtest.engine import run_backtest

    results: dict[str, Any] = run_backtest(
        lookback_weeks=args.weeks,
        top_n=args.top_n,
        portfolio_name=args.portfolio,
    )
    if results.get("status") != "completed":
        logger.info("Backtest: %s", results.get("status", "failed"))
        return

    s: dict[str, float] = results["strategy"]
    b: dict[str, float] = results["baseline"]
    logger.info(
        "\n%s\n"
        "  BACKTEST RESULTS  (%s → %s)\n"
        "  %d weekly periods\n"
        "%s\n\n"
        "%s %s %s\n"
        "%s\n"
        "%s %s %s\n"
        "%s %s %s\n"
        "%s %s %s\n"
        "%s %s %s\n\n"
        "%s %s",
        "=" * 60,
        results["period_start"],
        results["period_end"],
        results["weeks"],
        "=" * 60,
        f"{'Metric':<25}",
        f"{'Strategy':>12}",
        f"{'Baseline':>12}",
        "─" * 52,
        f"{'Cumulative Return':<25}",
        f"{s['cumulative_return']:>11.2%}",
        f"{b['cumulative_return']:>11.2%}",
        f"{'Annualized Return':<25}",
        f"{s['annualized_return']:>11.2%}",
        f"{b['annualized_return']:>11.2%}",
        f"{'Sharpe Ratio':<25}",
        f"{s['sharpe_ratio']:>12.4f}",
        f"{b['sharpe_ratio']:>12.4f}",
        f"{'Max Drawdown':<25}",
        f"{s['max_drawdown']:>11.2%}",
        f"{b['max_drawdown']:>11.2%}",
        f"{'Win Rate (vs baseline)':<25}",
        f"{results['win_rate']:>11.1%}",
    )


def cmd_pipeline(args: argparse.Namespace) -> None:
    from scheduler.jobs import run_pipeline

    tickers: list[str] | None = (
        [t.upper() for t in args.tickers] if args.tickers else None
    )
    run_pipeline(
        tickers=tickers,
        skip_training=args.skip_training,
        portfolio_name=args.portfolio,
    )


def cmd_scheduler(args: argparse.Namespace) -> None:
    from scheduler.jobs import start_scheduler

    start_scheduler()


def cmd_stats(args: argparse.Namespace) -> None:
    try:
        from db.models import MacroSeriesManager, PriceHistoryManager
        from db.schema import init_db

        init_db()
        price_mgr = PriceHistoryManager()
        macro_mgr = MacroSeriesManager()
        price_stats: dict[str, dict[str, Any]] = price_mgr.stats()
        macro_stats: dict[str, dict[str, Any]] = macro_mgr.stats()
        logger.info(
            "\nSQLite:\n"
            "  Price history     : %d tickers, %d total rows\n"
            "  Macro series      : %d series, %d total rows",
            len(price_stats),
            sum(s["count"] for s in price_stats.values()),
            len(macro_stats),
            sum(s["count"] for s in macro_stats.values()),
        )
    except Exception as e:
        logger.error("SQLite: unavailable (%s)", e)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Investment Portfolio Optimizer CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command")

    # portfolio — manage positions
    port_p = sub.add_parser("portfolio", help="Manage your investment portfolio")
    port_p.add_argument(
        "action",
        choices=["show", "add", "remove", "update"],
        help="Portfolio action",
    )
    port_p.add_argument("ticker", nargs="?", help="ETF ticker symbol")
    port_p.add_argument("shares", nargs="?", type=float, help="Number of shares")
    port_p.add_argument("--cost", type=float, default=None, help="Cost basis per share")
    port_p.add_argument(
        "--portfolio",
        default="default",
        help="Portfolio name (default: 'default')",
    )
    port_p.set_defaults(func=cmd_portfolio)

    # universe — manage ETF universe
    uni_p = sub.add_parser("universe", help="Manage the ETF universe")
    uni_p.add_argument(
        "action", choices=["show", "seed", "stats"], help="Universe action"
    )
    uni_p.set_defaults(func=cmd_universe)

    # discover — auto-find peer ETFs
    disc_p = sub.add_parser(
        "discover", help="Discover peer dividend ETFs automatically"
    )
    disc_p.add_argument(
        "--min-aum",
        type=float,
        default=100_000_000,
        help="Minimum AUM filter (default: $100M)",
    )
    disc_p.set_defaults(func=cmd_discover)

    # collect — pull data into SQLite
    coll_p = sub.add_parser("collect", help="Collect data from all sources into SQLite")
    coll_p.add_argument(
        "tickers", nargs="*", help="Specific tickers (default: full universe)"
    )
    coll_p.set_defaults(func=cmd_collect)

    # featurize — compute ML features
    feat_p = sub.add_parser("featurize", help="Compute ML feature vectors")
    feat_p.add_argument(
        "tickers",
        nargs="*",
        help="Specific tickers (default: all with price data)",
    )
    feat_p.set_defaults(func=cmd_featurize)

    # train — train LSTM model
    train_p = sub.add_parser("train", help="Train the LSTM prediction model")
    train_p.add_argument("--epochs", type=int, default=50, help="Max training epochs")
    train_p.add_argument("--batch-size", type=int, default=32)
    train_p.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    train_p.add_argument(
        "--patience", type=int, default=7, help="Early stopping patience"
    )
    train_p.add_argument(
        "--resume", action="store_true", help="Resume from last checkpoint"
    )
    train_p.set_defaults(func=cmd_train)

    # predict — generate ranked predictions
    pred_p = sub.add_parser("predict", help="Generate ranked predictions")
    pred_p.add_argument(
        "tickers", nargs="*", help="Specific tickers (default: full universe)"
    )
    pred_p.set_defaults(func=cmd_predict)

    # report — generate weekly report files
    rep_p = sub.add_parser("report", help="Generate weekly investment report")
    rep_p.add_argument("--portfolio", default="default", help="Portfolio name")
    rep_p.add_argument("--output-dir", default=None, help="Output directory")
    rep_p.set_defaults(func=cmd_report)

    # backtest — walk-forward simulation
    bt_p = sub.add_parser("backtest", help="Run walk-forward backtest")
    bt_p.add_argument(
        "--weeks",
        type=int,
        default=13,
        help="Lookback weeks (default: 13 = ~3 months)",
    )
    bt_p.add_argument("--top-n", type=int, default=5, help="Top N tickers for strategy")
    bt_p.add_argument(
        "--portfolio", default="default", help="Portfolio name for baseline"
    )
    bt_p.set_defaults(func=cmd_backtest)

    # pipeline — full end-to-end run
    pipe_p = sub.add_parser("pipeline", help="Run the full weekly pipeline end-to-end")
    pipe_p.add_argument(
        "tickers", nargs="*", help="Specific tickers (default: full universe)"
    )
    pipe_p.add_argument(
        "--skip-training", action="store_true", help="Skip model training"
    )
    pipe_p.add_argument(
        "--portfolio",
        default="default",
        help="Portfolio name for backtest baseline",
    )
    pipe_p.set_defaults(func=cmd_pipeline)

    # scheduler — start daemon
    sched_p = sub.add_parser("scheduler", help="Start the automated scheduler daemon")
    sched_p.set_defaults(func=cmd_scheduler)

    # stats — show all data stats
    stats_p = sub.add_parser("stats", help="Show database statistics")
    stats_p.set_defaults(func=cmd_stats)

    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = build_parser()
    args: argparse.Namespace = parser.parse_args()

    if not hasattr(args, "func"):
        parser.print_help()
        sys.exit(0)

    args.func(args)


if __name__ == "__main__":
    main()
