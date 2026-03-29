"""
LSTM with Temporal Attention for investment return prediction.

Architecture:
  Input: (batch, seq_len, n_features)
  -> 2-layer LSTM (hidden=64, dropout=0.2)
  -> Temporal attention (learns which days matter)
  -> FC head: 64 -> 32 -> 1 (predicted forward return)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ml.dataset import NUM_FEATURES


class TemporalAttention(nn.Module):
    """Attention over time steps — learns which days in the window matter most."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.attention: nn.Sequential = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, lstm_output: torch.Tensor) -> torch.Tensor:
        """Apply attention over time dimension. Input: (batch, seq_len, hidden)."""
        scores: torch.Tensor = self.attention(lstm_output)  # (batch, seq_len, 1)
        weights: torch.Tensor = torch.softmax(scores, dim=1)  # (batch, seq_len, 1)
        context: torch.Tensor = (lstm_output * weights).sum(dim=1)  # (batch, hidden)
        return context


class InvestmentLSTM(nn.Module):
    """LSTM + Attention model for predicting forward returns."""

    def __init__(
        self,
        input_size: int = NUM_FEATURES,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        self.input_norm: nn.LayerNorm = nn.LayerNorm(input_size)

        self.lstm: nn.LSTM = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        self.attention: TemporalAttention = TemporalAttention(hidden_size)

        self.head: nn.Sequential = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. Input: (batch, seq_len, input_size). Output: (batch,)."""
        x = self.input_norm(x)
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden_size)
        context: torch.Tensor = self.attention(lstm_out)  # (batch, hidden_size)
        return self.head(context).squeeze(-1)  # (batch,)
