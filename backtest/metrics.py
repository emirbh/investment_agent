from __future__ import annotations

import math


def cumulative_return(returns: list[float]) -> float:
    cum: float = 1.0
    for r in returns:
        cum *= 1 + r
    return cum - 1


def annualized_return(
    cum_return: float, periods: int, periods_per_year: int = 52
) -> float:
    if periods <= 0:
        return 0.0
    return (1 + cum_return) ** (periods_per_year / periods) - 1


def sharpe_ratio(
    returns: list[float],
    risk_free_rate: float = 0.045,
    periods_per_year: int = 52,
) -> float:
    if len(returns) < 2:
        return 0.0
    rf_period: float = risk_free_rate / periods_per_year
    excess: list[float] = [r - rf_period for r in returns]
    mean_excess: float = sum(excess) / len(excess)
    std: float = math.sqrt(
        sum((r - mean_excess) ** 2 for r in excess) / (len(excess) - 1)
    )
    if std == 0:
        return 0.0
    return mean_excess / std * math.sqrt(periods_per_year)


def max_drawdown(returns: list[float]) -> float:
    peak: float = 1.0
    cum: float = 1.0
    max_dd: float = 0.0
    for r in returns:
        cum *= 1 + r
        if cum > peak:
            peak = cum
        dd: float = (peak - cum) / peak
        if dd > max_dd:
            max_dd = dd
    return max_dd


def win_rate(strategy_returns: list[float], baseline_returns: list[float]) -> float:
    if not strategy_returns:
        return 0.0
    wins: int = sum(1 for s, b in zip(strategy_returns, baseline_returns) if s > b)
    return wins / len(strategy_returns)
