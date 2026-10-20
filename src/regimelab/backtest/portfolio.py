"""Opt-in next-open portfolio simulation with an auditable position ledger.

This API does not change the historical close-only overlay. Opening fills
require a caller-supplied exchange schedule and consistently adjusted prices;
the simulator never downloads data or infers an exchange calendar.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from regimelab.backtest._portfolio_inputs import prepare_inputs
from regimelab.backtest._portfolio_ledger import Ledger, run_ledger
from regimelab.backtest.engine import CASH_COLUMN
from regimelab.backtest.metrics import PerformanceMetrics, compute_metrics, drawdown_series

if TYPE_CHECKING:
    from regimelab.settings import Settings

__all__ = ["NextOpenResult", "simulate_next_open"]


@dataclass(frozen=True, slots=True)
class NextOpenResult(Ledger):
    """Actual units and cash, with separate opening and closing weights.

    ``fees`` are currency amounts; ``costs`` use previous closing NAV as
    denominator. ``traded_notional`` is gross risky fill notional divided by
    pre-trade opening NAV. ``execution_times`` are scheduled opening timestamps,
    not a promise that every row contains a fill; inspect ``traded_units``.
    """

    target_weights: pd.DataFrame
    decision_at_open: pd.Series
    execution_times: pd.Series
    net_returns: pd.Series
    costs: pd.Series
    traded_notional: pd.Series
    equity: pd.Series
    metrics: PerformanceMetrics
    benchmark_returns: pd.Series
    benchmark_equity: pd.Series
    benchmark_metrics: PerformanceMetrics

    @property
    def drawdown(self) -> pd.Series:
        """Drawdown including initial capital as a possible peak."""
        return drawdown_series(self.equity, initial=1.0)

    @property
    def benchmark_drawdown(self) -> pd.Series:
        """Benchmark drawdown under the same inception convention."""
        return drawdown_series(self.benchmark_equity, initial=1.0)

    def to_frame(self) -> pd.DataFrame:
        """Session-level audit trail; zero fills mean no hidden rebalance."""
        frame = pd.DataFrame(
            {
                "decision_at_open": self.decision_at_open,
                "scheduled_open": self.execution_times,
                "has_fills": self.traded_units.ne(0.0).any(axis="columns"),
                "nav_before_trade": self.nav_before_trade,
                "nav_after_trade": self.nav_after_trade,
                "nav": self.nav,
                "cash_balance": self.cash_balance,
                "cash_interest": self.cash_interest,
                "fees": self.fees,
                "cost": self.costs,
                "traded_notional": self.traded_notional,
                "net_return": self.net_returns,
                "equity": self.equity,
                "drawdown": self.drawdown,
                "benchmark_return": self.benchmark_returns,
                "benchmark_equity": self.benchmark_equity,
            },
        )
        for values, prefix in (
            (self.positions, "q_"),
            (self.traded_units, "fill_units_"),
            (self.traded_values, "fill_value_"),
            (self.opening_weights, "w_open_"),
            (self.closing_weights, "w_close_"),
        ):
            frame = frame.join(values.add_prefix(prefix))
        return frame


@dataclass(frozen=True, slots=True)
class _Summary:
    returns: pd.Series
    costs: pd.Series
    turnover: pd.Series
    equity: pd.Series
    metrics: PerformanceMetrics


def _summarize(ledger: Ledger, settings: Settings) -> _Summary:
    capital = settings.backtest.initial_capital
    previous = ledger.nav.shift(1, fill_value=capital)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        returns = (ledger.nav / previous - 1.0).rename("net_return")
        costs = (ledger.fees / previous).rename("cost")
        turnover = (
            ledger.traded_values.abs().sum(axis="columns") / ledger.nav_before_trade
        ).rename("traded_notional")
        equity = (ledger.nav / capital).rename("equity")
    if (returns <= -1.0).any() or not all(
        np.isfinite(series.to_numpy()).all() for series in (returns, costs, turnover, equity)
    ):
        raise ValueError("nonfinite or unrepresentable portfolio returns")
    metrics = compute_metrics(
        returns,
        trading_days=settings.features.trading_days_per_year,
        risk_free_rate=settings.backtest.risk_free_rate,
        weights=ledger.opening_weights,
        turnover=turnover,
        costs=costs,
        initial_equity=1.0,
    )
    return _Summary(returns, costs, turnover, equity, metrics)


def simulate_next_open(
    prices: pd.DataFrame,
    target_weights: pd.DataFrame,
    settings: Settings,
    *,
    open_prices: pd.DataFrame,
    sessions: pd.DataFrame,
    published_at: pd.Series | None = None,
    open_observed: pd.DataFrame | None = None,
    close_observed: pd.DataFrame | None = None,
    start: dt.date | None = None,
) -> NextOpenResult:
    """Execute close-of-session targets at an eligible later regular opening.

    Parameters
    ----------
    prices, open_prices
        Consistently adjusted closes/opens on identical session indexes.
        Missing prices are not filled. Fractional adjusted units are allowed.
    target_weights
        Sparse, long-only close-of-session decisions; cash is the residual.
        Identical targets retain units, not constant portfolio weights.
    settings
        Universe, benchmark, initial capital, cash APR and all-in per-risky-leg
        transaction cost in bps. This convention excludes cash from turnover.
        ``signal_lag_days`` must be 1: this is not an integer-lag overlay.
    sessions
        Full exchange schedule with timezone-aware ``market_open`` and
        ``market_close`` columns. The caller is responsible for its accuracy.
    published_at
        Optional timezone-aware availability per decision. Must not precede
        that session's close; an instruction exactly at the open is too late.
        Defaulting to the decision close is a research assumption, not a
        historical publication record or guaranteed auction fill.
    open_observed, close_observed
        Optional boolean provenance masks, on the full price index. False
        forbids using a carried price. Without masks the caller vouches that
        supplied prices are genuinely observed.
    start
        Portfolio inception on/after this date. Otherwise the first session
        after the earliest decision (or first price session for no decisions).

    Returns
    -------
    NextOpenResult
        Self-financing positions, valuations, fills and net metrics. Cash at
        the previous close earns the configured session rate before the open.
        The benchmark buys once at inception using the same ledger and costs.

    Raises
    ------
    ValueError
        On ambiguous input ordering, unavailable used prices, invalid weights,
        costs or schedules, or an unfinanceable/nonfinite portfolio.
    """
    inputs = prepare_inputs(
        prices,
        target_weights,
        settings,
        open_prices=open_prices,
        sessions=sessions,
        published_at=published_at,
        open_observed=open_observed,
        close_observed=close_observed,
        start=start,
    )
    assets = list(inputs.assets)
    ledger = run_ledger(
        inputs.opens.loc[:, assets],
        inputs.closes.loc[:, assets],
        inputs.instructions,
        capital=inputs.capital,
        cost_rate=inputs.cost_rate,
        daily_rf=inputs.daily_rf,
        open_observed=inputs.open_observed.loc[:, assets],
        close_observed=inputs.close_observed.loc[:, assets],
    )
    benchmark = settings.data.benchmark
    benchmark_targets = pd.DataFrame({benchmark: 1.0, CASH_COLUMN: 0.0}, index=inputs.opens.index)
    reference = run_ledger(
        inputs.opens.loc[:, [benchmark]],
        inputs.closes.loc[:, [benchmark]],
        benchmark_targets,
        capital=inputs.capital,
        cost_rate=inputs.cost_rate,
        daily_rf=inputs.daily_rf,
        open_observed=inputs.open_observed.loc[:, [benchmark]],
        close_observed=inputs.close_observed.loc[:, [benchmark]],
    )
    summary = _summarize(ledger, settings)
    comparison = _summarize(reference, settings)
    return NextOpenResult(
        positions=ledger.positions,
        cash_balance=ledger.cash_balance,
        cash_interest=ledger.cash_interest,
        nav_before_trade=ledger.nav_before_trade,
        nav_after_trade=ledger.nav_after_trade,
        nav=ledger.nav,
        opening_weights=ledger.opening_weights,
        closing_weights=ledger.closing_weights,
        traded_units=ledger.traded_units,
        traded_values=ledger.traded_values,
        fees=ledger.fees,
        target_weights=inputs.targets,
        decision_at_open=inputs.decision_at_open,
        execution_times=inputs.execution_times,
        net_returns=summary.returns,
        costs=summary.costs,
        traded_notional=summary.turnover,
        equity=summary.equity,
        metrics=summary.metrics,
        benchmark_returns=comparison.returns.rename("benchmark_return"),
        benchmark_equity=comparison.equity.rename("benchmark_equity"),
        benchmark_metrics=comparison.metrics,
    )
