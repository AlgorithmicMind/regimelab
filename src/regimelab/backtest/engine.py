"""Regime-conditional allocation backtest.

Scope, stated up front
----------------------
This is a **transparent overlay test**, not a trading system. It answers one
question: does conditioning a simple long-only allocation on the inferred regime
improve risk-adjusted outcomes relative to holding the benchmark? Nothing here
is optimised -- the regime weights are fixed by configuration and never fitted.
That is deliberate. Optimised weights would make the results a statement about
the optimiser rather than about the regime signal.

Execution assumptions, all conservative
---------------------------------------
- **Signal lag.** A regime inferred from the close of session ``t`` is acted on
  at the close of ``t + signal_lag_days``, default one session. Trading at the
  same close that produced the signal is the most common way a backtest becomes
  untradeable, and it typically accounts for a large share of the apparent edge
  in published regime strategies.
- **Rebalancing is event-driven, not calendar-driven.** Weights change only when
  the regime changes. This is what a regime overlay actually does, and it avoids
  imposing an arbitrary monthly or weekly grid that would blur the very timing
  the signal is being judged on. It also means turnover is an *output* of the
  classifier's stability rather than a parameter to be tuned -- a noisy
  classifier is charged for its noise.
- **Costs.** ``transaction_cost_bps`` is charged on gross traded notional every
  time weights move. There is no bid-ask model, no market impact and no
  slippage, so realised costs would be *higher* than reported, not lower.
- **Cash.** The unallocated fraction earns the configured risk-free rate. Not
  crediting it would overstate the benefit of de-risking; crediting a rate the
  strategy could not obtain would understate the cost.
- **Confidence floor.** When the modal posterior is below
  ``confidence_floor`` the previous allocation is retained rather than acting on
  an ambiguous read. This suppresses churn around regime boundaries, which is
  where a probabilistic classifier is least reliable and where naive
  argmax-driven overlays lose most of their gross edge to costs.
- **No shorting, no leverage.** Weights are non-negative and sum to at most one.

Known limitations
-----------------
Positions are assumed rebalanced exactly to target whenever the regime changes,
with no intra-period drift between those events. For liquid ETFs held over
multi-week regimes this is a small approximation, but it is not free: it flatters
turnover slightly, because a real portfolio drifts and needs topping up.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import numpy as np
import pandas as pd

from regimelab.backtest.metrics import (
    PerformanceMetrics,
    compute_metrics,
    drawdown_series,
    equity_curve,
    regime_conditional_metrics,
)
from regimelab.models.types import AllocationWeights, MarketRegime, regime_ladder

if TYPE_CHECKING:
    from regimelab.settings import BacktestSettings, Settings

__all__ = ["CASH_COLUMN", "BacktestResult", "run_backtest", "target_weight_frame"]

CASH_COLUMN = "CASH"
"""Column name for the unallocated fraction."""

# Weight changes below this threshold are treated as noise and not traded, so
# floating-point residue in the allocation table cannot manufacture turnover.
_TRADE_THRESHOLD = 1e-9


@dataclass(frozen=True, slots=True)
class BacktestResult:
    """Complete output of a backtest run.

    Attributes
    ----------
    regimes
        Regime in force on each session, as observed at the decision time.
    target_weights
        Weights the regime implies on each session, before lag and schedule.
    held_weights
        Weights actually held during each session, after lag and schedule. These
        are the weights that generated the returns.
    gross_returns
        Portfolio return before transaction costs.
    net_returns
        Portfolio return after transaction costs. All reported statistics use
        this series.
    traded_notional
        Gross notional traded per session, as a fraction of capital.
    costs
        Transaction cost per session, as a fraction of capital.
    equity, benchmark_equity
        Compounded curves, both starting at 1.0.
    metrics, benchmark_metrics
        Statistics for the strategy and for buy-and-hold.
    regime_metrics
        Benchmark return statistics conditioned on regime -- the diagnostic that
        shows whether the classifier separated anything.
    """

    regimes: pd.Series
    target_weights: pd.DataFrame
    held_weights: pd.DataFrame
    gross_returns: pd.Series
    net_returns: pd.Series
    traded_notional: pd.Series
    costs: pd.Series
    equity: pd.Series
    benchmark_returns: pd.Series
    benchmark_equity: pd.Series
    metrics: PerformanceMetrics
    benchmark_metrics: PerformanceMetrics
    regime_metrics: pd.DataFrame

    @property
    def drawdown(self) -> pd.Series:
        """Strategy drawdown from the running peak."""
        return drawdown_series(self.equity)

    @property
    def benchmark_drawdown(self) -> pd.Series:
        """Benchmark drawdown from the running peak."""
        return drawdown_series(self.benchmark_equity)

    @property
    def excess_return(self) -> float:
        """Total return of the strategy minus that of the benchmark."""
        return self.metrics.total_return - self.benchmark_metrics.total_return

    def comparison_table(self) -> pd.DataFrame:
        """Side-by-side strategy versus benchmark statistics."""
        strategy = dict(self.metrics.summary_rows())
        benchmark = dict(self.benchmark_metrics.summary_rows())
        keys = list(strategy)
        return pd.DataFrame(
            {
                "metric": keys,
                "strategy": [strategy[k] for k in keys],
                "benchmark": [benchmark.get(k, "-") for k in keys],
            },
        )

    def to_frame(self) -> pd.DataFrame:
        """Session-level audit trail, suitable for persisting as Parquet."""
        frame = pd.DataFrame(
            {
                "regime": self.regimes.map(lambda r: getattr(r, "value", r)),
                "gross_return": self.gross_returns,
                "net_return": self.net_returns,
                "traded_notional": self.traded_notional,
                "cost": self.costs,
                "equity": self.equity,
                "benchmark_return": self.benchmark_returns,
                "benchmark_equity": self.benchmark_equity,
                "drawdown": self.drawdown,
            },
        )
        weights = self.held_weights.add_prefix("w_")
        return frame.join(weights)


def target_weight_frame(
    regimes: pd.Series,
    settings: BacktestSettings,
    symbols: list[str],
) -> pd.DataFrame:
    """Translate a regime path into target portfolio weights.

    Parameters
    ----------
    regimes
        Regime label per session.
    settings
        Backtest configuration holding the per-regime allocations.
    symbols
        Investable symbols. Any allocation to a symbol outside this list is
        redirected to cash rather than silently dropped, so the weights always
        sum to one.

    Returns
    -------
    pandas.DataFrame
        Weights indexed by session, with one column per symbol plus
        :data:`CASH_COLUMN`.

    Notes
    -----
    The allocation table is normalised through
    :class:`~regimelab.models.types.AllocationWeights`, which caps every symbol
    at 1.0 and non-negativity, and recomputes the residual as cash. This protects
    against human-edited TOML tables that slightly exceed 1.0 due to rounding.
    """
    columns = [*symbols, CASH_COLUMN]
    rows: dict[MarketRegime, dict[str, float]] = {}

    for regime, raw in settings.regime_weights.items():
        allocation = AllocationWeights(weights=raw)
        normalised = allocation.normalise(symbols=symbols)
        weights = dict.fromkeys(columns, 0.0)
        for symbol, weight in normalised.items():
            weights[symbol] = weight
        weights[CASH_COLUMN] = max(0.0, 1.0 - sum(normalised.values()))
        rows[regime] = weights

    # A regime with no configured allocation defaults to cash.
    fallback = dict.fromkeys(columns, 0.0)
    fallback[CASH_COLUMN] = 1.0

    records = [rows.get(regime, fallback) for regime in regimes]
    frame = pd.DataFrame(records, index=regimes.index, columns=columns)
    return frame.astype("float64")


def run_backtest(
    prices: pd.DataFrame,
    regimes: pd.Series,
    settings: Settings,
    *,
    confidence: pd.Series | None = None,
    start: dt.date | None = None,
) -> BacktestResult:
    """Run the regime-conditional allocation backtest.

    Parameters
    ----------
    prices
        Closing prices, dates by symbols. Must include the benchmark.
    regimes
        Regime label per session, as inferred causally. Reindexed onto the price
        calendar and forward-filled: a regime observed at ``t`` remains in force
        until a new observation, which is causal.
    settings
        Resolved configuration.
    confidence
        Modal posterior confidence per session, in ``[0, 1]``. Below
        ``settings.backtest.confidence_floor`` the previous allocation is
        retained.
    start
        Optional first evaluated session, used to exclude the warm-up span.

    Returns
    -------
    BacktestResult
        Full result, including the benchmark comparison.

    Raises
    ------
    ValueError
        If the benchmark is absent or no session survives alignment.
    """

    backtest_cfg = settings.backtest
    benchmark = settings.data.benchmark

    if benchmark not in prices.columns:
        msg = f"benchmark {benchmark!r} absent from the price frame"
        raise ValueError(msg)

    investable = [s for s in settings.data.risk_assets if s in prices.columns]
    asset_returns = prices.loc[:, investable].pct_change()

    aligned_regimes = regimes.reindex(prices.index).ffill()
    usable = aligned_regimes.dropna().index
    if start is not None:
        usable = usable[usable >= pd.Timestamp(start)]

    if len(usable) < 2:
        msg = "backtest requires at least 2 sessions with a known regime"
        raise ValueError(msg)

    aligned_regimes = aligned_regimes.loc[usable]
    asset_returns = asset_returns.loc[usable]

    targets = target_weight_frame(aligned_regimes, backtest_cfg, investable)

    # Event-driven rebalancing: the target changes only when the regime changes.
    # The raw target series would have a new value every session even when the
    # regime has not changed; this mask holds the previous target whenever the
    # inferred regime is unchanged.
    regime_changed = aligned_regimes.ne(aligned_regimes.shift()).fillna(True)
    scheduled = targets.where(regime_changed, other=np.nan).ffill().astype("float64")

    # Apply the confidence floor. When the classifier is unsure, its modal
    # posterior is low and the risk of acting on noise is highest; in those
    # sessions the previous allocation is held.
    if confidence is not None and backtest_cfg.confidence_floor > 0.0:
        aligned_conf = confidence.reindex(prices.index).reindex(usable)
        confident = aligned_conf.ge(backtest_cfg.confidence_floor).fillna(True)
        scheduled = scheduled.where(confident, other=np.nan).ffill().astype("float64")

    # Signal lag. The scheduled weight at t is decided on information as of
    # t - signal_lag_days and is held on t.
    held = scheduled.shift(backtest_cfg.signal_lag_days)
    held = held.ffill(limit=backtest_cfg.signal_lag_days)
    held = held.fillna(0.0)

    if len(held):
        # Before the first executable decision the portfolio is fully in cash.
        lead_in = min(backtest_cfg.signal_lag_days, len(held))
        held.iloc[:lead_in] = 0.0
        cash_idx = cast(int, held.columns.get_loc(CASH_COLUMN))
        held.iloc[:lead_in, cash_idx] = 1.0

    daily_rf = backtest_cfg.risk_free_rate / settings.features.trading_days_per_year
    risk_returns = (held.loc[:, investable] * asset_returns).sum(axis="columns")
    cash_returns = held[CASH_COLUMN] * daily_rf
    gross_returns = (risk_returns + cash_returns).rename("gross_return")

    # Gross notional traded: buys plus sells, as a fraction of capital. Tiny
    # residual changes below _TRADE_THRESHOLD are treated as no trade, so
    # identical consecutive target weights do not bleed transaction costs.
    weight_diff = held.diff().abs()
    weight_diff = weight_diff.where(weight_diff >= _TRADE_THRESHOLD, 0.0)
    traded = weight_diff.sum(axis="columns")
    traded.iloc[0] = held.iloc[0].abs().sum()
    traded = traded.rename("traded_notional")

    costs = (traded * backtest_cfg.transaction_cost_bps / 1e4).rename("cost")
    net_returns = (gross_returns - costs).rename("net_return")

    benchmark_returns = prices[benchmark].pct_change().loc[usable].rename("benchmark_return")

    metrics = compute_metrics(
        net_returns,
        trading_days=settings.features.trading_days_per_year,
        risk_free_rate=backtest_cfg.risk_free_rate,
        weights=held,
        turnover=traded,
        costs=costs,
    )
    benchmark_metrics = compute_metrics(
        benchmark_returns,
        trading_days=settings.features.trading_days_per_year,
        risk_free_rate=backtest_cfg.risk_free_rate,
    )

    conditional = regime_conditional_metrics(
        benchmark_returns,
        aligned_regimes,
        trading_days=settings.features.trading_days_per_year,
        order=list(regime_ladder(settings.model.n_states)),
    )

    return BacktestResult(
        regimes=aligned_regimes.rename("regime"),
        target_weights=targets,
        held_weights=held,
        gross_returns=gross_returns,
        net_returns=net_returns,
        traded_notional=traded,
        costs=costs,
        equity=equity_curve(net_returns).rename("equity"),
        benchmark_returns=benchmark_returns,
        benchmark_equity=equity_curve(benchmark_returns).rename("benchmark_equity"),
        metrics=metrics,
        benchmark_metrics=benchmark_metrics,
        regime_metrics=conditional,
    )
