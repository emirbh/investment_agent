"""
Training loop for the InvestmentLSTM model.
Supports MPS (Apple Silicon), CUDA, and CPU backends.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from ml.dataset import InvestmentDataset
from ml.model import InvestmentLSTM
from ml.utils import MODEL_DIR, get_device, load_checkpoint, save_checkpoint

logger: logging.Logger = logging.getLogger(__name__)


def train_model(
    tickers: list[str] | None = None,
    train_end_date: str | None = None,
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    seq_len: int = 10,
    patience: int = 7,
    val_split: float = 0.2,
    resume: bool = False,
) -> dict[str, Any]:
    device: torch.device = get_device()
    logger.info("[Train] Device: %s", device)

    dataset = InvestmentDataset(
        tickers=tickers,
        end_date=train_end_date,
        seq_len=seq_len,
    )

    if len(dataset) < 50:
        logger.warning(
            "[Train] Only %d samples — need at least 50. Collect more data first.",
            len(dataset),
        )
        return {"status": "insufficient_data", "samples": len(dataset)}

    val_size: int = max(1, int(len(dataset) * val_split))
    train_size: int = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    logger.info(
        "[Train] Samples: %d (train=%d, val=%d)", len(dataset), train_size, val_size
    )

    model: InvestmentLSTM = InvestmentLSTM().to(device)

    if resume:
        ckpt = load_checkpoint(model, path=os.path.join(MODEL_DIR, "best_model.pt"))
        if ckpt:
            logger.info(
                "[Train] Resumed from epoch %d, val_loss=%.6f",
                ckpt["epoch"],
                ckpt["val_loss"],
            )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion: nn.MSELoss = nn.MSELoss()

    best_val_loss: float = float("inf")
    patience_counter: int = 0
    history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
    epoch: int = 0

    for epoch in range(epochs):
        model.train()
        train_loss: float = 0.0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            pred: torch.Tensor = model(batch_x)
            loss: torch.Tensor = criterion(pred, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * batch_x.size(0)

        train_loss /= train_size

        model.eval()
        val_loss: float = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                pred = model(batch_x)
                loss = criterion(pred, batch_y)
                val_loss += loss.item() * batch_x.size(0)

        val_loss /= val_size
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            logger.info(
                "  Epoch %3d/%d  train_loss=%.6f  val_loss=%.6f",
                epoch + 1,
                epochs,
                train_loss,
                val_loss,
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            save_checkpoint(model, optimizer, epoch, val_loss)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(
                    "  Early stopping at epoch %d (patience=%d)", epoch + 1, patience
                )
                break

    logger.info("[Train] Best val_loss: %.6f", best_val_loss)

    return {
        "status": "completed",
        "epochs_run": epoch + 1,
        "best_val_loss": best_val_loss,
        "samples": len(dataset),
        "device": str(device),
        "history": history,
    }
