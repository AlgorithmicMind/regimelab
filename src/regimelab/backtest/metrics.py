"""Performance and risk statistics.

Conventions applied throughout, stated once so the numbers are unambiguous:

- Returns are **simple** (arithmetic) at the portfolio level. Log returns are
  used for feature construction because they aggregate additively, but a
  portfolio's period return is a weighted sum of simple returns, and mixing the
  two is a common source of small, persistent errors.
- Annualisation uses ``252`` sessions, configurable.
- The Sharpe ratio is computed on **excess** returns over the configured
  risk-free rate. Reporting a Sharpe on raw returns silently credits the cash
  rate to the strategy, which flattered every published backtest of the
  2010s and severely penalises them post-2022.
- Maximum drawdown is measured on the **compounded equity curve**, not on
  cumulative sums.
- Every statistic is computed on the realised, **cost-inclusive** net return
  series. Gross-of-cost figures are not reported anywhere in this project.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = [
    "PerformanceMetrics",
    "compute_metrics",
    "drawdown_series",
    "equity_curve",
    "regime_conditional_metrics",
]

_EPSILON = 1e-12


def equity_curve(returns: pd.Series, initial: float = 1.0) -> pd.Series:
    """Compound a simple return series into an equity curve.

    Parameters
    ----------
    returns
        Simple period returns.
    initial
        Starting capital.

    Returns
    -------
    pandas.Series
        Equity curve. Missing returns are treated as zero: a session with no
        observation is a session with no position change, not a session that
        should terminate the curve.
    """
    clean = returns.fillna(0.0).astype("float64")
    return initial * (1.0 + clean).cumprod()


def drawdown_series(equity: pd.Series, *, initial: float | None = None) -> pd.Series:
    """Drawdown from the running peak, as a non-positive fraction.

    Parameters
    ----------
    equity
        Equity curve.
    initial
        Optional starting capital included in the peak baseline: a curve that
        opens below it already reads as a drawdown. ``None`` (the default)
        keeps the legacy behaviour of measuring peaks from the curve alone.
    """
    peak = equity.cummax()
    if initial is not None:
        if not (np.isfinite(initial) and initial > 0):
            msg = "initial equity must be finite and positive"
            raise ValueError(msg)
        peak = peak.clip(lower=initial)
    return (equity / peak - 1.0).rename("drawdown")


class PerformanceMetrics(BaseModel):
    """Summary statistics for a return series.

    Every field is annualised where applicable and net of transaction costs.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    n_observations: int = Field(ge=0)
    start: str | None = None
    end: str | None = None

    total_return: float
    """Cumulative return over the full period."""

    cagr: float
    """Compound annual growth rate."""

    annual_volatility: float
    """Annualised standard deviation of returns."""

    sharpe: float
    """Annualised Sharpe ratio on excess returns."""

    sortino: float
    """Annualised Sortino ratio: excess return over downside deviation.

    More appropriate than Sharpe for a strategy whose whole purpose is
    asymmetric -- suppressing downside while retaining upside. Penalising a
    defensive overlay for upside volatility, as Sharpe does, misstates it."""

    max_drawdown: float
    """Worst peak-to-trough decline, as a non-positive fraction."""

    calmar: float
    """CAGR divided by the absolute maximum drawdown."""

    hit_rate: float
    """Fraction of sessions with a positive return."""

    skewness: float
    """Sample skewness of returns. Negative skew is the cost of most yield-like
    strategies and is invisible in Sharpe."""

    excess_kurtosis: float
    """Sample excess kurtosis. Large values invalidate the Gaussian assumptions
    underlying the Sharpe ratio's usual interpretation."""

    var_95: float
    """Historical 5th-percentile daily return."""

    cvar_95: float
    """Mean return conditional on breaching :attr:`var_95` -- expected shortfall."""

    average_exposure: float | None = None
    """Mean gross exposure to risk assets, when weights are supplied."""

    annual_turnover: float | None = None
    """Annualised one-way turnover, when weights are supplied."""

    total_cost: float | None = None
    """Cumulative transaction cost as a fraction of capital."""

    def summary_rows(self) -> list[tuple[str, str]]:
        """Human-readable label/value pairs for tables and the TUI."""
        rows = [
            ("Total return", f"{self.total_return:+.1%}"),
            ("CAGR", f"{self.cagr:+.2%}"),
            ("Annual volatility", f"{self.annual_volatility:.2%}"),
            ("Sharpe", f"{self.sharpe:.2f}"),
            ("Sortino", f"{self.sortino:.2f}"),
            ("Max drawdown", f"{self.max_drawdown:.1%}"),
            ("Calmar", f"{self.calmar:.2f}"),
            ("Hit rate", f"{self.hit_rate:.1%}"),
            ("Skewness", f"{self.skewness:+.2f}"),
            ("Excess kurtosis", f"{self.excess_kurtosis:+.2f}"),
            ("VaR 95%", f"{self.var_95:.2%}"),
            ("CVaR 95%", f"{self.cvar_95:.2%}"),
        ]
        if self.average_exposure is not None:
            rows.append(("Average exposure", f"{self.average_exposure:.1%}"))
        if self.annual_turnover is not None:
            rows.append(("Annual turnover", f"{self.annual_turnover:.1f}x"))
        if self.total_cost is not None:
            rows.append(("Cumulative cost", f"{self.total_cost:.2%}"))
        return rows


def compute_metrics(
    returns: pd.Series,
    *,
    trading_days: int = 252,
    risk_free_rate: float = 0.0,
    weights: pd.DataFrame | None = None,
    turnover: pd.Series | None = None,
    costs: pd.Series | None = None,
    initial_equity: float | None = None,
) -> PerformanceMetrics:
    """Compute the full statistics block for a net return series.

    Parameters
    ----------
    returns
        Simple daily portfolio returns, net of costs.
    trading_days
        Sessions per year.
    risk_free_rate
        Annualised risk-free rate used for the excess-return ratios.
    weights
        Portfolio weights, for the exposure statistic.
    turnover
        Per-session one-way turnover, for the turnover statistic.
    costs
        Per-session transaction cost as a fraction of capital.
    initial_equity
        Optional starting capital counted in the drawdown peak baseline, so an
        opening loss reads as a drawdown. ``None`` (the default) keeps the
        legacy metrics unchanged.

    Returns
    -------
    PerformanceMetrics
        Validated statistics block.
    """
    if initial_equity is not None and not (np.isfinite(initial_equity) and initial_equity > 0):
        msg = "initial equity must be finite and positive"
        raise ValueError(msg)

    clean = returns.dropna().astype("float64")

    if clean.empty:
        return PerformanceMetrics(
            n_observations=0,
            total_return=0.0,
            cagr=0.0,
            annual_volatility=0.0,
            sharpe=0.0,
            sortino=0.0,
            max_drawdown=0.0,
            calmar=0.0,
            hit_rate=0.0,
            skewness=0.0,
            excess_kurtosis=0.0,
            var_95=0.0,
            cvar_95=0.0,
        )

    values = clean.to_numpy()
    n = values.size

    equity = equity_curve(clean)
    total_return = float(equity.iloc[-1] - 1.0)

    years = n / trading_days
    cagr = float((1.0 + total_return) ** (1.0 / years) - 1.0) if years > 0 else 0.0

    daily_vol = float(values.std(ddof=1)) if n > 1 else 0.0
    annual_vol = daily_vol * np.sqrt(trading_days)

    daily_rf = risk_free_rate / trading_days
    excess = values - daily_rf
    mean_excess = float(excess.mean())

    sharpe = mean_excess / daily_vol * np.sqrt(trading_days) if daily_vol > _EPSILON else 0.0

    # Downside deviation about the risk-free rate, not about the sample mean:
    # what matters is failing to beat cash, not failing to beat oneself.
    downside = np.minimum(excess, 0.0)
    downside_dev = float(np.sqrt(np.mean(downside**2)))
    sortino = mean_excess / downside_dev * np.sqrt(trading_days) if downside_dev > _EPSILON else 0.0

    drawdown = drawdown_series(equity, initial=initial_equity)
    max_drawdown = float(drawdown.min())
    calmar = cagr / abs(max_drawdown) if abs(max_drawdown) > _EPSILON else 0.0

    var_95 = float(np.quantile(values, 0.05))
    tail = values[values <= var_95]
    cvar_95 = float(tail.mean()) if tail.size else var_95

    average_exposure: float | None = None
    if weights is not None and not weights.empty:
        cash_columns = [c for c in weights.columns if str(c).upper() == "CASH"]
        risk_weights = weights.drop(columns=cash_columns, errors="ignore")
        average_exposure = float(risk_weights.abs().sum(axis="columns").mean())

    annual_turnover: float | None = None
    if turnover is not None and not turnover.dropna().empty:
        annual_turnover = float(turnover.dropna().mean() * trading_days)

    total_cost: float | None = None
    if costs is not None and not costs.dropna().empty:
        total_cost = float(costs.dropna().sum())

    skew_value = cast(float, pd.Series(values, dtype=float).skew()) if n > 2 else 0.0
    kurt_value = cast(float, pd.Series(values, dtype=float).kurtosis()) if n > 3 else 0.0

    return PerformanceMetrics(
        n_observations=n,
        start=str(clean.index[0].date()) if hasattr(clean.index[0], "date") else None,
        end=str(clean.index[-1].date()) if hasattr(clean.index[-1], "date") else None,
        total_return=total_return,
        cagr=cagr,
        annual_volatility=annual_vol,
        sharpe=float(sharpe),
        sortino=float(sortino),
        max_drawdown=max_drawdown,
        calmar=float(calmar),
        hit_rate=float((values > 0.0).mean()),
        skewness=skew_value,
        excess_kurtosis=kurt_value,
        var_95=var_95,
        cvar_95=cvar_95,
        average_exposure=average_exposure,
        annual_turnover=annual_turnover,
        total_cost=total_cost,
    )


def regime_conditional_metrics(
    returns: pd.Series,
    regimes: pd.Series,
    *,
    trading_days: int = 252,
    order: Sequence[object] | None = None,
) -> pd.DataFrame:
    """Return statistics conditioned on the prevailing regime.

    The most direct test of whether the classifier found anything. If the
    conditional mean returns and volatilities do not separate monotonically along
    the risk ladder, the states are not regimes -- whatever the log likelihood
    says.

    Parameters
    ----------
    returns
        Daily returns of the reference asset or the strategy.
    regimes
        Regime label per date.
    trading_days
        Sessions per year.
    order
        Explicit row order. Defaults to first appearance.

    Returns
    -------
    pandas.DataFrame
        One row per regime with occupancy, annualised return, annualised
        volatility, Sharpe, hit rate and worst drawdown within the regime.
    """
    aligned_returns, aligned_regimes = returns.align(regimes, join="inner")
    frame = pd.DataFrame(
        {"ret": aligned_returns.astype("float64"), "regime": aligned_regimes},
    ).dropna()

    if frame.empty:
        return pd.DataFrame()

    keys = list(order) if order is not None else list(dict.fromkeys(frame["regime"]))
    rows: list[dict[str, object]] = []

    for key in keys:
        block = frame.loc[frame["regime"] == key, "ret"]
        if block.empty:
            continue

        equity = equity_curve(block)
        volatility = float(block.std(ddof=1)) if len(block) > 1 else 0.0

        rows.append(
            {
                "regime": getattr(key, "value", key),
                "sessions": len(block),
                "occupancy": len(block) / len(frame),
                "ann_return": float(block.mean() * trading_days),
                "ann_volatility": volatility * float(np.sqrt(trading_days)),
                "sharpe": (
                    float(block.mean() / volatility * np.sqrt(trading_days))
                    if volatility > _EPSILON
                    else 0.0
                ),
                "hit_rate": float((block > 0.0).mean()),
                "worst_drawdown": float(drawdown_series(equity).min()),
            },
        )

    return pd.DataFrame(rows)
