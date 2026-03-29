"""
Pipeline orchestration and scheduling.

Pipeline steps:
  1. Collect data from all sources (prices, macro)
  2. Compute ML features
  3. Train/update the model
  4. Generate predictions
  5. Run backtest
  6. Generate report

Scheduling:
  - Daily: data collection (step 1)
  - Weekly: full pipeline (steps 1-6)
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

logger: logging.Logger = logging.getLogger(__name__)


def run_pipeline(
    tickers: list[str] | None = None,
    skip_training: bool = False,
    portfolio_name: str = "default",
) -> dict[str, Any]:
    from db.schema import init_db

    init_db()

    results: dict[str, Any] = {}
    run_date: str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info(
        "\n%s\n  INVESTMENT PIPELINE — %s\n%s",
        "=" * 60,
        run_date,
        "=" * 60,
    )

    # Step 1: Collect data
    logger.info("\n[1/6] Collecting data from all sources...")
    try:
        from pipeline.collector import collect_all

        collect_results: dict[str, dict[str, int]] = collect_all(tickers)
        n_prices: int = sum(collect_results["prices"].values())
        n_macro: int = sum(collect_results["macro"].values())
        results["collect"] = {"prices": n_prices, "macro": n_macro, "status": "ok"}
        logger.info(
            "  Collected %d price rows, %d macro data points.", n_prices, n_macro
        )
    except Exception as e:
        logger.error("  Collection FAILED: %s", e)
        results["collect"] = {"status": "failed", "error": str(e)}

    # Step 2: Compute features
    logger.info("\n[2/6] Computing ML features...")
    try:
        from pipeline.features import compute_features

        n_features: int = compute_features(tickers)
        results["features"] = {"count": n_features, "status": "ok"}
        logger.info("  Computed %d feature rows.", n_features)
    except Exception as e:
        logger.error("  Feature computation FAILED: %s", e)
        results["features"] = {"status": "failed", "error": str(e)}

    # Step 3: Train model
    if not skip_training:
        logger.info("\n[3/6] Training ML model...")
        try:
            from ml.train import train_model

            train_results: dict[str, Any] = train_model(
                tickers=tickers, epochs=50, patience=7
            )
            results["train"] = train_results
            if train_results["status"] == "completed":
                logger.info(
                    "  Trained %d epochs, val_loss=%.6f",
                    train_results["epochs_run"],
                    train_results["best_val_loss"],
                )
            else:
                logger.info("  Skipped: %s", train_results["status"])
        except Exception as e:
            logger.error("  Training FAILED: %s", e)
            results["train"] = {"status": "failed", "error": str(e)}
    else:
        logger.info("\n[3/6] Skipping training (--skip-training)")
        results["train"] = {"status": "skipped"}

    # Step 4: Generate predictions
    logger.info("\n[4/6] Generating predictions...")
    predictions: list[dict[str, Any]] | None = None
    try:
        from ml.predict import generate_predictions

        predictions = generate_predictions(tickers=tickers)
        results["predictions"] = {"count": len(predictions), "status": "ok"}
        logger.info("  Generated predictions for %d tickers.", len(predictions))
    except Exception as e:
        logger.error("  Prediction FAILED: %s", e)
        results["predictions"] = {"status": "failed", "error": str(e)}

    # Step 5: Run backtest
    logger.info("\n[5/6] Running backtest...")
    bt_results: dict[str, Any] | None = None
    try:
        from backtest.engine import run_backtest

        bt_results = run_backtest(portfolio_name=portfolio_name)
        results["backtest"] = bt_results
        if bt_results.get("status") == "completed":
            s: dict[str, float] = bt_results["strategy"]
            b: dict[str, float] = bt_results["baseline"]
            logger.info(
                "  Strategy: %s  Baseline: %s  Win rate: %s",
                f"{s['cumulative_return']:.2%}",
                f"{b['cumulative_return']:.2%}",
                f"{bt_results['win_rate']:.0%}",
            )
        else:
            logger.info("  Status: %s", bt_results.get("status"))
    except Exception as e:
        logger.error("  Backtest FAILED: %s", e)
        results["backtest"] = {"status": "failed", "error": str(e)}

    # Step 6: Generate report
    logger.info("\n[6/6] Generating report...")
    try:
        from report.generator import generate_report

        paths: dict[str, str] = generate_report(
            predictions=predictions,
            backtest_results=bt_results if isinstance(bt_results, dict) else None,
            portfolio_name=portfolio_name,
        )
        results["report"] = {"paths": paths, "status": "ok"}
    except Exception as e:
        logger.error("  Report generation FAILED: %s", e)
        results["report"] = {"status": "failed", "error": str(e)}

    logger.info("\n%s\n  PIPELINE COMPLETE\n%s\n", "=" * 60, "=" * 60)

    return results


def start_scheduler() -> None:
    """
    Start the APScheduler daemon for automated daily/weekly runs.

    Daily job (weekdays 6PM ET): collect data only (skip training).
    Weekly job (Saturday 9AM): full pipeline including training, backtest, report.
    Requires: pip install apscheduler
    """
    try:
        from apscheduler.schedulers.blocking import BlockingScheduler
        from apscheduler.triggers.cron import CronTrigger
    except ImportError:
        logger.error(
            "APScheduler not installed. Install with: pip install apscheduler\n"
            "Alternatively, use system cron to call: python main.py pipeline"
        )
        return

    scheduler = BlockingScheduler()

    scheduler.add_job(
        func=lambda: run_pipeline(skip_training=True),
        trigger=CronTrigger(day_of_week="mon-fri", hour=18, minute=0),
        id="daily_collect",
        name="Daily data collection",
    )

    scheduler.add_job(
        func=run_pipeline,
        trigger=CronTrigger(day_of_week="sat", hour=9, minute=0),
        id="weekly_pipeline",
        name="Weekly full pipeline",
    )

    lines: list[str] = ["Scheduler started. Jobs:"]
    for job in scheduler.get_jobs():
        lines.append(f"  {job.name}: {job.trigger}")
    lines.append("\nPress Ctrl+C to stop.\n")
    logger.info("\n".join(lines))

    try:
        scheduler.start()
    except KeyboardInterrupt:
        scheduler.shutdown()
        logger.info("Scheduler stopped.")
