"""Validation and publication ordering for the opt-in portfolio ledger."""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype

from regimelab.backtest.engine import CASH_COLUMN, _validated_weight_frame

if TYPE_CHECKING:
    from regimelab.settings import Settings


@dataclass(frozen=True, slots=True)
class PortfolioInputs:
    """Validated copies; the simulator never repairs or mutates caller data."""

    closes: pd.DataFrame
    opens: pd.DataFrame
    targets: pd.DataFrame
    instructions: pd.DataFrame
    decision_at_open: pd.Series
    execution_times: pd.Series
    open_observed: pd.DataFrame
    close_observed: pd.DataFrame
    assets: tuple[str, ...]
    capital: float
    cost_rate: float
    daily_rf: float


def _check_index(index: pd.Index, name: str) -> None:
    if not isinstance(index, pd.DatetimeIndex):
        raise ValueError(f"{name} requires a DatetimeIndex of session labels")
    if (
        index.tz is not None
        or index.hasnans
        or not index.is_unique
        or not index.is_monotonic_increasing
        or not index.equals(index.normalize())
    ):
        raise ValueError(f"{name} requires unique, ascending, normalized, timezone-naive sessions")


def _check_columns(frame: pd.DataFrame, name: str) -> None:
    if not frame.columns.is_unique:
        raise ValueError(f"{name} has duplicate columns")


def _utc_times(series: pd.Series, name: str) -> pd.Series:
    if not isinstance(series.dtype, pd.DatetimeTZDtype) or series.isna().any():
        raise ValueError(f"{name} requires nonmissing timezone-aware timestamps")
    return pd.Series(
        pd.DatetimeIndex(series).tz_convert("UTC"), index=series.index, name=series.name, copy=True
    )


def _mask(
    frame: pd.DataFrame | None,
    calendar: pd.Index,
    columns: list[str],
    name: str,
) -> pd.DataFrame:
    if frame is None:
        return pd.DataFrame(True, index=calendar, columns=columns)
    _check_columns(frame, name)
    if not frame.index.equals(calendar):
        raise ValueError(f"{name} must use exactly the price session index")
    if frame.isna().any(axis=None) or not all(is_bool_dtype(t) for t in frame.dtypes):
        raise ValueError(f"{name} requires boolean, nonmissing observation flags")
    return frame.reindex(columns=columns, fill_value=False).astype(bool).copy()


def _instructions(
    targets: pd.DataFrame,
    published: pd.Series,
    openings: pd.Series,
) -> tuple[pd.DataFrame, pd.Series]:
    instructions = pd.DataFrame(np.nan, index=openings.index, columns=targets.columns)
    owners: list[pd.Timestamp | None] = []
    for session, opening in openings.items():
        eligible = published.index[published < opening]
        if len(eligible):
            owner = eligible[-1]
            instructions.loc[session] = targets.loc[owner]
            owners.append(owner)
        else:
            owners.append(None)
    return instructions, pd.Series(
        owners, index=openings.index, dtype="datetime64[ns]", name="decision_at_open"
    )


def _market_schedule(
    prices: pd.DataFrame,
    open_prices: pd.DataFrame,
    sessions: pd.DataFrame,
) -> tuple[pd.Series, pd.Series]:
    _check_index(prices.index, "prices")
    for frame, name in ((prices, "prices"), (open_prices, "open_prices"), (sessions, "sessions")):
        _check_columns(frame, name)
        _check_index(frame.index, name)
        if not frame.index.equals(prices.index):
            raise ValueError(f"{name} must use exactly the price session index")
    if len(prices) < 2:
        raise ValueError("next-open simulation requires at least two sessions")
    if not {"market_open", "market_close"}.issubset(sessions.columns):
        raise ValueError("sessions requires market_open and market_close")
    openings = _utc_times(sessions["market_open"], "market_open")
    closings = _utc_times(sessions["market_close"], "market_close")
    if (openings >= closings).any():
        raise ValueError("each market_open must precede its market_close")
    for previous, following in zip(closings.iloc[:-1], openings.iloc[1:], strict=True):
        if previous >= following:
            raise ValueError("session timestamps overlap or are not ascending")
    return openings, closings


def _configuration(
    settings: Settings, prices: pd.DataFrame, open_prices: pd.DataFrame
) -> tuple[float, float, float]:
    assets = tuple(settings.data.risk_assets)
    benchmark = settings.data.benchmark
    if len(set(assets)) != len(assets) or CASH_COLUMN in (*assets, benchmark):
        raise ValueError("risk_assets must be unique and CASH cannot be a risky asset/benchmark")
    if benchmark not in prices.columns or benchmark not in open_prices.columns:
        raise ValueError(f"benchmark {benchmark!r} must have opening and closing price columns")
    if settings.backtest.signal_lag_days != 1:
        raise ValueError("simulate_next_open requires signal_lag_days=1, not a close-only lag")
    capital = settings.backtest.initial_capital
    rate = settings.backtest.risk_free_rate
    cost = settings.backtest.transaction_cost_bps / 10000.0
    if not math.isfinite(capital) or capital <= 0:
        raise ValueError("initial_capital must be finite and positive")
    if not math.isfinite(rate) or rate < 0:
        raise ValueError("risk_free_rate must be finite and nonnegative")
    if not math.isfinite(cost) or not 0 <= cost < 1:
        raise ValueError("per-leg transaction cost rate must be finite and in [0, 1)")
    return capital, cost, rate / settings.features.trading_days_per_year


def _decision_targets(
    target_weights: pd.DataFrame, prices: pd.DataFrame, settings: Settings
) -> pd.DataFrame:
    targets = target_weights.copy()
    if not len(targets):
        targets.index = pd.DatetimeIndex([], name=prices.index.name)
    _check_index(targets.index, "target_weights")
    _check_columns(targets, "target_weights")
    if not np.isfinite(targets.to_numpy(dtype="float64")).all():
        raise ValueError("target_weights must contain finite numeric weights")
    targets = targets.astype("float64")
    for symbol in targets.columns:
        if symbol != CASH_COLUMN and symbol not in prices.columns and (targets[symbol] > 0).any():
            raise ValueError(f"target_weights assigns positive weight to unpriced asset {symbol}")
    targets = _validated_weight_frame(targets, prices, settings)
    investable = tuple(s for s in settings.data.risk_assets if s in prices.columns)
    risk = targets.loc[:, list(investable)].clip(lower=0.0, upper=1.0)
    risk = risk.div(risk.sum(axis="columns").clip(lower=1.0), axis="index")
    targets = risk.copy()
    targets[CASH_COLUMN] = (1.0 - risk.sum(axis="columns")).clip(lower=0.0)
    return targets


def _evaluation_index(calendar: pd.Index, targets: pd.DataFrame, start: dt.date | None) -> pd.Index:
    if start is not None:
        cutoff = pd.Timestamp(start)
        if cutoff.tz is not None or cutoff != cutoff.normalize():
            raise ValueError("start must be a timezone-naive session date")
        evaluated = calendar[calendar >= cutoff]
    elif len(targets):
        evaluated = calendar[calendar > targets.index[0]]
    else:
        evaluated = calendar
    if len(evaluated) < 2:
        raise ValueError("next-open simulation requires at least two evaluated sessions")
    return evaluated


def prepare_inputs(
    prices: pd.DataFrame,
    target_weights: pd.DataFrame,
    settings: Settings,
    *,
    open_prices: pd.DataFrame,
    sessions: pd.DataFrame,
    published_at: pd.Series | None,
    open_observed: pd.DataFrame | None,
    close_observed: pd.DataFrame | None,
    start: dt.date | None,
) -> PortfolioInputs:
    """Reject ambiguous inputs and select instructions without looking ahead."""
    openings, closings = _market_schedule(prices, open_prices, sessions)
    capital, cost, daily_rf = _configuration(settings, prices, open_prices)
    targets = _decision_targets(target_weights, prices, settings)
    if published_at is None:
        published = closings.reindex(targets.index).copy()
    else:
        if not published_at.index.equals(targets.index):
            raise ValueError("published_at must be indexed exactly like target_weights")
        published = _utc_times(published_at, "published_at")
    if (published < closings.reindex(targets.index)).any():
        raise ValueError("published_at precedes the decision session close")
    calendar = prices.index
    evaluated = _evaluation_index(calendar, targets, start)
    investable = tuple(str(s) for s in targets.columns if s != CASH_COLUMN)
    benchmark = settings.data.benchmark
    columns = list(dict.fromkeys([*investable, benchmark]))
    observed_open = _mask(open_observed, calendar, columns, "open_observed")
    observed_close = _mask(close_observed, calendar, columns, "close_observed")
    instructions, owners = _instructions(targets, published, openings.loc[evaluated])
    return PortfolioInputs(
        closes=prices.reindex(columns=columns).loc[evaluated].astype("float64").copy(),
        opens=open_prices.reindex(columns=columns).loc[evaluated].astype("float64").copy(),
        targets=targets,
        instructions=instructions,
        decision_at_open=owners,
        execution_times=openings.loc[evaluated].rename("execution_time"),
        open_observed=observed_open.loc[evaluated],
        close_observed=observed_close.loc[evaluated],
        assets=investable,
        capital=capital,
        cost_rate=cost,
        daily_rf=daily_rf,
    )
