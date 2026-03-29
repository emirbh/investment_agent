"""
ML utilities — device selection, checkpointing.
"""

from __future__ import annotations

import os
from typing import Any

import torch

MODEL_DIR: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")


def get_device() -> torch.device:
    """Select best available device: MPS > CUDA > CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_loss: float,
    path: str | None = None,
) -> str:
    """Save a training checkpoint to disk."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    path = path or os.path.join(MODEL_DIR, "best_model.pt")
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss,
        },
        path,
    )
    return path


def load_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    path: str | None = None,
) -> dict[str, Any] | None:
    """Load a training checkpoint from disk. Returns None if file doesn't exist."""
    path = path or os.path.join(MODEL_DIR, "best_model.pt")
    if not os.path.exists(path):
        return None
    checkpoint: dict[str, Any] = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint
