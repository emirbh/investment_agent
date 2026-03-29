from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Any

import torch

from db.connection import get_connection
from db.schema import init_db
from ml.dataset import InferenceDataset
from ml.model import InvestmentLSTM
from ml.utils import MODEL_DIR, get_device

logger: logging.Logger = logging.getLogger(__name__)


def generate_predictions(
    tickers: list[str] | None = None,
    model_path: str | None = None,
    seq_len: int = 10,
) -> list[dict[str, Any]]:
    """
    Run inference on the latest features for each ticker, rank by predicted
    forward return, assign buy/sell actions based on percentile position,
    and persist results to the predictions table. Returns the full ranked
    prediction list sorted descending by predicted return.
    """
    init_db()
    device: torch.device = get_device()
    model_path = model_path or os.path.join(MODEL_DIR, "best_model.pt")

    if not os.path.exists(model_path):
        logger.warning("[Predict] No trained model found. Run 'train' first.")
        return []

    model: InvestmentLSTM = InvestmentLSTM()
    checkpoint: dict[str, Any] = torch.load(
        model_path, map_location="cpu", weights_only=True
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    inf_data: InferenceDataset = InferenceDataset(tickers=tickers, seq_len=seq_len)

    if not inf_data.sequences:
        logger.warning(
            "[Predict] No tickers with sufficient feature data for inference."
        )
        return []

    predictions: list[dict[str, Any]] = []
    run_date: str = datetime.now().strftime("%Y-%m-%d")

    with torch.no_grad():
        for ticker, seq in inf_data.sequences.items():
            seq_batch: torch.Tensor = seq.unsqueeze(0).to(device)
            pred_return: float = model(seq_batch).item()
            predictions.append(
                {
                    "ticker": ticker,
                    "predicted_ret": pred_return,
                }
            )

    predictions.sort(key=lambda x: x["predicted_ret"], reverse=True)

    n: int = len(predictions)
    for i, p in enumerate(predictions):
        p["predicted_rank"] = i + 1
        percentile: float = i / n if n > 1 else 0.5

        if percentile < 0.2:
            p["action"] = "BUY"
            p["confidence"] = 0.8 - percentile
        elif percentile < 0.4:
            p["action"] = "OVERWEIGHT"
            p["confidence"] = 0.6
        elif percentile < 0.6:
            p["action"] = "HOLD"
            p["confidence"] = 0.5
        elif percentile < 0.8:
            p["action"] = "UNDERWEIGHT"
            p["confidence"] = 0.6
        else:
            p["action"] = "SELL"
            p["confidence"] = 0.8 - (1 - percentile)

    conn = get_connection()
    for p in predictions:
        conn.execute(
            """INSERT INTO predictions
               (run_date, ticker, predicted_rank, predicted_ret, confidence, action)
               VALUES (?, ?, ?, ?, ?, ?)
               ON CONFLICT(run_date, ticker) DO UPDATE SET
                   predicted_rank = excluded.predicted_rank,
                   predicted_ret = excluded.predicted_ret,
                   confidence = excluded.confidence,
                   action = excluded.action""",
            (
                run_date,
                p["ticker"],
                p["predicted_rank"],
                p["predicted_ret"],
                p["confidence"],
                p["action"],
            ),
        )
    conn.commit()
    conn.close()

    logger.info(
        "[Predict] Generated predictions for %d tickers (run_date=%s).",
        len(predictions),
        run_date,
    )
    return predictions
