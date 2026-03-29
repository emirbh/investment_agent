"""
Tests for the PyTorch model.
"""

from __future__ import annotations

import torch
import pytest

from ml.dataset import NUM_FEATURES
from ml.model import InvestmentLSTM, TemporalAttention
from ml.utils import get_device


class TestInvestmentLSTM:
    def test_forward_pass_shape(self) -> None:
        model: InvestmentLSTM = InvestmentLSTM()
        batch: torch.Tensor = torch.randn(4, 21, NUM_FEATURES)
        out: torch.Tensor = model(batch)
        assert out.shape == (4,)

    def test_single_sample(self) -> None:
        model: InvestmentLSTM = InvestmentLSTM()
        x: torch.Tensor = torch.randn(1, 21, NUM_FEATURES)
        out: torch.Tensor = model(x)
        assert out.shape == (1,)
        assert torch.isfinite(out).all()

    def test_gradient_flow(self) -> None:
        model: InvestmentLSTM = InvestmentLSTM()
        x: torch.Tensor = torch.randn(4, 21, NUM_FEATURES)
        target: torch.Tensor = torch.randn(4)

        out: torch.Tensor = model(x)
        loss: torch.Tensor = torch.nn.MSELoss()(out, target)
        loss.backward()

        # Check gradients exist
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"


class TestTemporalAttention:
    def test_output_shape(self) -> None:
        attn: TemporalAttention = TemporalAttention(64)
        x: torch.Tensor = torch.randn(4, 21, 64)
        out: torch.Tensor = attn(x)
        assert out.shape == (4, 64)

    def test_attention_weights_sum_to_one(self) -> None:
        attn: TemporalAttention = TemporalAttention(64)
        x: torch.Tensor = torch.randn(1, 21, 64)
        scores: torch.Tensor = attn.attention(x)
        weights: torch.Tensor = torch.softmax(scores, dim=1)
        assert weights.sum().item() == pytest.approx(1.0, abs=0.01)


class TestDevice:
    def test_device_available(self) -> None:
        device: torch.device = get_device()
        assert device.type in ("mps", "cuda", "cpu")

    def test_model_on_device(self) -> None:
        device: torch.device = get_device()
        model: InvestmentLSTM = InvestmentLSTM().to(device)
        x: torch.Tensor = torch.randn(2, 21, NUM_FEATURES).to(device)
        out: torch.Tensor = model(x)
        assert out.device.type == device.type
