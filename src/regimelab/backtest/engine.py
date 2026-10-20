"""Regime-conditional allocation backtest.

Scope, stated up front
----------------------
This is a **transparent overlay test**, not a trading system. It answers one
question: does conditioning a simple long-only allocation on the inferred regime
improve risk-adjusted outcomes relative to holding the benchmark? Nothing here
is optimised -- the regime weights are fixed by configuration and never fitted.
That is deliberate. Optimised weights would make the results a statement about
the optimiser rather than about the regime signal.

Legacy overlay assumptions
--------------------------
- **Signal shift, not an execution timestamp.** The weight used for the return
  ending at session ``t`` comes from ``t - signal_lag_days``. Thus shift one
  applies a close-D decision to the return from close D to close D+1. It does
  not establish a realizable fill after observing close D. For an explicit
  next-opening position ledger use ``simulate_next_open`` instead.
- **Event-driven targets.** ``run_backtest`` translates regime changes into
  targets; ``simulate_weights`` adopts changed weight vectors. Neither models
  physical units drifting between rebalance instructions.
- **Costs.** The historical turnover convention includes absolute changes in
  risky weights and the residual CASH column. ``transaction_cost_bps`` is
  multiplied by that convention; it is not an observed broker execution cost.
- **Cash.** The unallocated fraction earns the configured constant cash rate.
  This is an assumption, not evidence of a yield obtainable throughout history.
- **Confidence floor.** When the modal posterior is below
  ``confidence_floor`` the previous allocation is retained rather than acting on
  an ambiguous read. This suppresses churn around regime boundaries, which is
  where a probabilistic classifier is least reliable and where naive
  argmax-driven overlays lose most of their gross edge to costs.
- **No shorting, no leverage.** Weights are non-negative and sum to at most one.

Known limitations
-----------------
Weights remain constant between changes, implying unpriced maintenance of the
mix. This is not a self-financing buy-and-hold ledger and no bound on its
performance/cost approximation is claimed. Valid-input legacy outputs are
retained for comparison. Missing prices needed for an exposed return or the
benchmark are errors; only unused assets and the benchmark's first interval
without a preceding input observation may have undefined returns.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, cast

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

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

__all__ = [
    "CASH_COLUMN",
    "BacktestResult",
    "run_backtest",
    "simulate_weights",
    "target_weight_frame",
]

CASH_COLUMN = "CASH"
"""Column name for the unallocated fraction."""

# Weight changes below this threshold are treated as noise and not traded, so
# floating-point residue in the allocation table cannot manufacture turnover.
_TRADE_THRESHOLD = 1e-9

# Validation tolerance for supplied weights: bookkeeping residue, not
# economics. Anything beyond this is a real allocation error.
_WEIGHT_TOLERANCE = 1e-9


@dataclass(frozen=True, slots=True)
class BacktestResult:
    """Complete output of a backtest run.

    Attributes
    ----------
    regimes
        Regime in force on each session, as observed at the decision time.
        ``None`` when the backtest simulated a supplied weight path rather
        than a regime signal.
    target_weights
        Weights the regime implies on each session, before lag and schedule.
    held_weights
        Lagged overlay weights used to calculate each session's return, not
        physical position units or proof of an executable trade.
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
        shows whether the classifier separated anything. ``None`` without a
        regime path.
    """

    regimes: pd.Series | None
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
    regime_metrics: pd.DataFrame | None

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
        if self.regimes is not None:
            frame.insert(
                0,
                "regime",
                self.regimes.map(lambda r: getattr(r, "value", r)),
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


def _validated_weight_frame(
    target_weights: pd.DataFrame,
    prices: pd.DataFrame,
    settings: Settings,
) -> pd.DataFrame:
    """Validate a decision frame and normalise it to ``[*investable, CASH]``.

    Raises ``ValueError`` on any inconsistency: NaN weights, columns outside
    ``risk_assets`` plus ``CASH``, a positive weight on an unpriced risk asset,
    weights outside ``[0, 1]``, a risk sum above 1, a CASH column that is not
    the residual, or decision dates outside the price calendar. All
    comparisons tolerate ``_WEIGHT_TOLERANCE`` of bookkeeping residue.
    """
    if target_weights.isna().any(axis=None):
        msg = "target_weights contains NaN weights"
        raise ValueError(msg)

    allowed = {str(s) for s in settings.data.risk_assets} | {CASH_COLUMN}
    unknown = [str(c) for c in target_weights.columns if str(c) not in allowed]
    if unknown:
        msg = (
            f"target_weights columns {unknown} are not in risk_assets "
            f"{list(settings.data.risk_assets)} or {CASH_COLUMN}"
        )
        raise ValueError(msg)

    unpriced = [
        c for c in target_weights.columns if str(c) != CASH_COLUMN and c not in prices.columns
    ]
    positive_unpriced = [c for c in unpriced if (target_weights[c] > _WEIGHT_TOLERANCE).any()]
    if positive_unpriced:
        msg = f"target_weights assigns positive weight to {positive_unpriced}, which have no prices"
        raise ValueError(msg)

    extra_dates = target_weights.index.difference(prices.index)
    if len(extra_dates):
        msg = (
            "target_weights has decision dates outside the price calendar, "
            f"e.g. {[d.date().isoformat() for d in extra_dates[:5]]}"
        )
        raise ValueError(msg)

    risk_columns = [c for c in target_weights.columns if str(c) != CASH_COLUMN]
    risk = target_weights.loc[:, risk_columns]
    if (risk < -_WEIGHT_TOLERANCE).any(axis=None) or (risk > 1.0 + _WEIGHT_TOLERANCE).any(
        axis=None
    ):
        msg = "target weights must lie in [0, 1]"
        raise ValueError(msg)
    risk_sum = (
        risk.sum(axis="columns") if risk_columns else pd.Series(0.0, index=target_weights.index)
    )
    if (risk_sum > 1.0 + _WEIGHT_TOLERANCE).any():
        msg = "risk weights sum above 1.0"
        raise ValueError(msg)

    # The investable columns are the configured risk assets that are priced;
    # missing ones mean zero exposure.
    investable = [s for s in settings.data.risk_assets if s in prices.columns]
    aligned = target_weights.reindex(columns=[*investable, CASH_COLUMN])
    if CASH_COLUMN in target_weights.columns:
        residual = 1.0 - risk_sum
        if (aligned[CASH_COLUMN] - residual).abs().gt(_WEIGHT_TOLERANCE).any():
            msg = "CASH column must equal 1 - sum(risk weights)"
            raise ValueError(msg)
    else:
        aligned[CASH_COLUMN] = (1.0 - risk_sum).clip(lower=0.0)
    aligned.loc[:, investable] = aligned.loc[:, investable].fillna(0.0)
    return aligned


def _adopted_schedule(targets: pd.DataFrame) -> pd.DataFrame:
    """Collapse a target path into the sequence actually adopted.

    A new target replaces the adopted one only when it differs by more than
    ``_TRADE_THRESHOLD`` -- measured against the last *adopted* row, not the
    previous raw row, so a sequence of sub-threshold drifts cannot accumulate
    into a real change.
    """
    matrix = targets.to_numpy(dtype="float64")
    adopted_rows = np.empty_like(matrix)
    adopted = matrix[0].copy()
    for i in range(matrix.shape[0]):
        if np.abs(matrix[i] - adopted).max() > _TRADE_THRESHOLD:
            adopted = matrix[i].copy()
        adopted_rows[i] = adopted
    return pd.DataFrame(adopted_rows, index=targets.index, columns=targets.columns)


def _raise_invalid_prices(bad: pd.DataFrame) -> None:
    if bad.any(axis=None):
        rows, columns = np.nonzero(bad.to_numpy(dtype=bool))
        session = pd.Timestamp(bad.index[rows[0]]).date()
        symbol = bad.columns[columns[0]]
        raise ValueError(f"invalid close/return prices for {symbol!r} on {session}")


def _finite_mask(frame: pd.DataFrame) -> pd.DataFrame:
    values = frame.to_numpy(dtype="float64", na_value=np.nan)
    return pd.DataFrame(np.isfinite(values), index=frame.index, columns=frame.columns)


def _checked_returns(
    prices: pd.DataFrame,
    required: pd.DataFrame,
    *,
    allow_initial_missing: bool = False,
) -> pd.DataFrame:
    """Validate both marks of every used close-to-close return, without fills."""
    if not all(is_numeric_dtype(dtype) for dtype in prices.dtypes):
        raise ValueError("close prices must be numeric")
    current = prices.reindex(index=required.index, columns=required.columns)
    previous = prices.shift(1).reindex(index=required.index, columns=required.columns)
    needs_previous = required.copy()
    if allow_initial_missing and prices.index[0] in required.index:
        needs_previous.loc[[prices.index[0]], :] = False
    current_valid = (_finite_mask(current) & current.gt(0.0)).fillna(False)
    previous_valid = (_finite_mask(previous) & previous.gt(0.0)).fillna(False)
    _raise_invalid_prices((required & ~current_valid) | (needs_previous & ~previous_valid))
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        returns = prices.pct_change(fill_method=None).reindex(
            index=required.index, columns=required.columns
        )
    _raise_invalid_prices(needs_previous & ~_finite_mask(returns))
    return returns


def simulate_weights(
    prices: pd.DataFrame,
    target_weights: pd.DataFrame,
    settings: Settings,
    *,
    start: dt.date | None = None,
    regimes: pd.Series | None = None,
) -> BacktestResult:
    """Simulate holding a hand-supplied target-weight schedule.

    This is the engine behind :func:`run_backtest`, exposed so that any
    allocation policy -- not only the regime map -- can be evaluated under the
    same execution assumptions: signal lag, event-driven rebalancing,
    transaction costs, and a cash sleeve earning the risk-free rate.

    Parameters
    ----------
    prices
        Numeric closing prices, dates by symbols. Must include the benchmark.
        Required current/previous marks must be finite and positive. Prices
        are never implicitly filled; unused missing assets are tolerated.
    target_weights
        Decision frame: one row per decision date, one column per risk asset
        plus an optional :data:`CASH_COLUMN`. Columns must be a subset of
        ``settings.data.risk_assets`` plus CASH; missing investable columns
        are filled with zero. A positive weight on a symbol absent from
        ``prices`` is rejected. Every weight must lie in ``[0, 1]`` and the
        risk weights may not sum above 1 (within ``_WEIGHT_TOLERANCE``). A
        CASH column, when present, must equal ``1 - sum(risk weights)``; when
        absent it is computed as ``max(0, 1 - sum)``. Every decision date must
        be a session in ``prices.index``.
    settings
        Resolved configuration.
    start
        Optional first evaluated session, used to exclude the warm-up span.
    regimes
        Optional regime path, aligned the same way as ``target_weights``. When
        given, the result carries it and regime-conditional benchmark metrics.

    Returns
    -------
    BacktestResult
        ``target_weights`` holds the aligned input schedule; ``regimes`` and
        ``regime_metrics`` are ``None`` unless ``regimes`` was supplied.

    Raises
    ------
    ValueError
        On any invalid input described above, or when fewer than two sessions
        survive alignment. Undefined exposed or benchmark returns are errors;
        only the benchmark's first interval may lack a preceding observation
        when evaluation begins at the first input row.
    """
    backtest_cfg = settings.backtest
    benchmark = settings.data.benchmark

    if benchmark not in prices.columns:
        msg = f"benchmark {benchmark!r} absent from the price frame"
        raise ValueError(msg)

    investable = [s for s in settings.data.risk_assets if s in prices.columns]
    aligned = _validated_weight_frame(target_weights, prices, settings)

    # A decision taken at t holds until the next one; that is causal.
    aligned = aligned.reindex(prices.index).ffill()
    usable = aligned.index[aligned.notna().all(axis="columns")]
    if start is not None:
        usable = usable[usable >= pd.Timestamp(start)]
    if len(usable) < 2:
        msg = "backtest requires at least 2 sessions with a target weight"
        raise ValueError(msg)

    aligned = aligned.loc[usable]
    # Event-driven rebalancing: adopt a new target only when it departs from
    # the last adopted one by more than the trade threshold.
    scheduled = _adopted_schedule(aligned)

    # --- From here on, identical mechanics to the historical engine -------
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

    asset_returns = _checked_returns(prices.loc[:, investable], held.loc[:, investable].gt(0.0))
    benchmark_required = pd.DataFrame(True, index=usable, columns=[benchmark])
    benchmark_returns = _checked_returns(
        prices.loc[:, [benchmark]], benchmark_required, allow_initial_missing=True
    )[benchmark].rename("benchmark_return")
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

    aligned_regimes = None
    conditional = None
    if regimes is not None:
        aligned_regimes = regimes.reindex(prices.index).ffill().loc[usable].rename("regime")
        conditional = regime_conditional_metrics(
            benchmark_returns,
            aligned_regimes,
            trading_days=settings.features.trading_days_per_year,
            order=list(regime_ladder(settings.model.n_states)),
        )

    return BacktestResult(
        regimes=aligned_regimes,
        target_weights=aligned,
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

    aligned_regimes = regimes.reindex(prices.index).ffill()
    usable = aligned_regimes.dropna().index
    if start is not None:
        usable = usable[usable >= pd.Timestamp(start)]

    if len(usable) < 2:
        msg = "backtest requires at least 2 sessions with a known regime"
        raise ValueError(msg)

    aligned_regimes = aligned_regimes.loc[usable]

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

    # The confidence floor can blank the first decision rows, leaving a
    # leading all-NaN span. The documented stance for that span is 100% cash,
    # so fill it -- otherwise those sessions would hold zero weights and earn
    # nothing instead of the risk-free rate.
    first_valid = scheduled.first_valid_index()
    leading = (
        scheduled.index if first_valid is None else scheduled.index[scheduled.index < first_valid]
    )
    if len(leading):
        scheduled.loc[leading, :] = 0.0
        scheduled.loc[leading, CASH_COLUMN] = 1.0

    result = simulate_weights(prices, scheduled, settings, regimes=aligned_regimes)
    return replace(result, target_weights=targets)
