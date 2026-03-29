"""
Tests for backtest metrics.
"""

from __future__ import annotations

import pytest

from backtest.metrics import cumulative_return, max_drawdown, sharpe_ratio, win_rate


class TestMetrics:
    def test_cumulative_return_positive(self) -> None:
        returns: list[float] = [0.01, 0.02, 0.01]
        cr: float = cumulative_return(returns)
        assert cr > 0
        assert cr == pytest.approx(0.04050, abs=0.001)

    def test_cumulative_return_zero(self) -> None:
        assert cumulative_return([]) == 0.0

    def test_cumulative_return_negative(self) -> None:
        returns: list[float] = [-0.05, -0.03]
        cr: float = cumulative_return(returns)
        assert cr < 0

    def test_sharpe_ratio(self) -> None:
        # Varying positive returns with zero risk-free should give positive Sharpe
        returns: list[float] = [
            0.02,
            0.03,
            0.01,
            0.04,
            0.02,
            0.03,
            0.01,
            0.02,
            0.03,
            0.01,
        ]
        sr: float = sharpe_ratio(returns, risk_free_rate=0.0)
        assert sr > 0

    def test_sharpe_ratio_single_return(self) -> None:
        assert sharpe_ratio([0.01]) == 0.0

    def test_max_drawdown_no_loss(self) -> None:
        returns: list[float] = [0.01, 0.02, 0.03]
        assert max_drawdown(returns) == 0.0

    def test_max_drawdown_with_loss(self) -> None:
        returns: list[float] = [0.10, -0.20, 0.05]
        dd: float = max_drawdown(returns)
        assert dd > 0
        assert dd < 1.0

    def test_win_rate(self) -> None:
        strat: list[float] = [0.05, -0.02, 0.03, 0.01]
        base: list[float] = [0.02, 0.01, 0.02, 0.04]
        wr: float = win_rate(strat, base)
        # strat wins periods 0, 2 (2 out of 4)
        assert wr == 0.5

    def test_win_rate_empty(self) -> None:
        assert win_rate([], []) == 0.0
