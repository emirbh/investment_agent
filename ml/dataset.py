"""
PyTorch Dataset for time-series feature sequences.
Reads from the SQLite features table and constructs windowed sequences.
"""

from __future__ import annotations

import json

import torch
from torch.utils.data import Dataset

from db.connection import get_connection

# Feature keys in order — must be consistent across training and inference
FEATURE_KEYS: list[str] = [
    "ret_1d",
    "ret_5d",
    "ret_10d",
    "ret_21d",
    "ret_63d",
    "vol_10d",
    "vol_21d",
    "sma_ratio_10",
    "sma_ratio_21",
    "sma_ratio_50",
    "rsi_14",
    "volume_ratio",
    "trailing_div_yield",
    "dist_52w_high",
    "dist_52w_low",
    "fed_rate",
    "treasury_10y",
    "treasury_2y",
    "yield_curve",
    "cpi",
    "hy_spread",
    "oil_wti",
    "real_yield",
    "unemployment",
]

NUM_FEATURES: int = len(FEATURE_KEYS)


class InvestmentDataset(Dataset):
    """
    Loads feature sequences from SQLite.
    Each sample: (sequence of `seq_len` feature vectors, target forward return).
    """

    def __init__(
        self,
        tickers: list[str] | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        seq_len: int = 21,
        target_col: str = "target",
    ) -> None:
        self.seq_len: int = seq_len
        self.samples: list[tuple[torch.Tensor, float]] = []

        conn = get_connection()

        if tickers is None:
            rows = conn.execute(
                "SELECT DISTINCT ticker FROM features ORDER BY ticker"
            ).fetchall()
            tickers = [r["ticker"] for r in rows]

        for ticker in tickers:
            query: str = (
                "SELECT date, feature_vec, target FROM features WHERE ticker = ?"
            )
            params: list[str] = [ticker]
            if start_date:
                query += " AND date >= ?"
                params.append(start_date)
            if end_date:
                query += " AND date <= ?"
                params.append(end_date)
            query += " ORDER BY date"

            rows = conn.execute(query, params).fetchall()

            if len(rows) < seq_len + 1:
                continue

            # Parse feature vectors
            parsed: list[tuple[list[float], float | None]] = []
            for r in rows:
                fv: dict[str, float | None] = json.loads(r["feature_vec"])
                vec: list[float] = [fv.get(k, 0.0) or 0.0 for k in FEATURE_KEYS]
                parsed.append((vec, r["target"]))

            # Create sliding windows
            for i in range(seq_len, len(parsed)):
                target: float | None = parsed[i][1]
                if target is None:
                    continue

                seq: torch.Tensor = torch.tensor(
                    [parsed[j][0] for j in range(i - seq_len, i)],
                    dtype=torch.float32,
                )
                self.samples.append((seq, float(target)))

        conn.close()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        seq, target = self.samples[idx]
        return seq, torch.tensor(target, dtype=torch.float32)


class InferenceDataset:
    """
    Build feature sequences for inference (no target needed).
    Returns the latest `seq_len` feature vectors per ticker.
    """

    def __init__(
        self,
        tickers: list[str] | None = None,
        seq_len: int = 21,
    ) -> None:
        self.seq_len: int = seq_len
        self.sequences: dict[str, torch.Tensor] = {}

        conn = get_connection()

        if tickers is None:
            rows = conn.execute(
                "SELECT DISTINCT ticker FROM features ORDER BY ticker"
            ).fetchall()
            tickers = [r["ticker"] for r in rows]

        for ticker in tickers:
            rows = conn.execute(
                "SELECT feature_vec FROM features WHERE ticker = ? "
                "ORDER BY date DESC LIMIT ?",
                (ticker, seq_len),
            ).fetchall()

            if len(rows) < seq_len:
                continue

            # Reverse to chronological order
            rows = list(reversed(rows))
            vecs: list[list[float]] = []
            for r in rows:
                fv: dict[str, float | None] = json.loads(r["feature_vec"])
                vecs.append([fv.get(k, 0.0) or 0.0 for k in FEATURE_KEYS])

            self.sequences[ticker] = torch.tensor(vecs, dtype=torch.float32)

        conn.close()
