"""Point-in-time backtesting and walk-forward evaluation."""

from __future__ import annotations

from regimelab.backtest.engine import (
    CASH_COLUMN,
    BacktestResult,
    run_backtest,
    simulate_weights,
    target_weight_frame,
)
from regimelab.backtest.metrics import (
    PerformanceMetrics,
    compute_metrics,
    drawdown_series,
    equity_curve,
    regime_conditional_metrics,
)
from regimelab.backtest.portfolio import NextOpenResult, simulate_next_open
from regimelab.backtest.walkforward import (
    WalkForwardFold,
    WalkForwardResult,
    run_walk_forward,
)

__all__ = [
    "CASH_COLUMN",
    "BacktestResult",
    "NextOpenResult",
    "PerformanceMetrics",
    "WalkForwardFold",
    "WalkForwardResult",
    "compute_metrics",
    "drawdown_series",
    "equity_curve",
    "regime_conditional_metrics",
    "run_backtest",
    "run_walk_forward",
    "simulate_next_open",
    "simulate_weights",
    "target_weight_frame",
]
