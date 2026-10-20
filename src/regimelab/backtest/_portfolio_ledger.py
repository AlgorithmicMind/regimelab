"""One self-financing ledger shared by the strategy and its benchmark."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from regimelab.backtest.engine import _TRADE_THRESHOLD, CASH_COLUMN

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class Ledger:
    """Currency and adjusted-unit accounting, without a synthetic gross curve."""

    positions: pd.DataFrame
    cash_balance: pd.Series
    cash_interest: pd.Series
    nav_before_trade: pd.Series
    nav_after_trade: pd.Series
    nav: pd.Series
    opening_weights: pd.DataFrame
    closing_weights: pd.DataFrame
    traded_units: pd.DataFrame
    traded_values: pd.DataFrame
    fees: pd.Series


def _require_prices(
    prices: FloatArray,
    observed: BoolArray,
    required: BoolArray,
    assets: list[str],
    session: pd.Timestamp,
    *,
    mark: str,
) -> None:
    bad = required & (~np.isfinite(prices) | (prices <= 0.0) | ~observed)
    if bad.any():
        names = [assets[i] for i in np.flatnonzero(bad)]
        raise ValueError(f"invalid/unobserved {mark} prices for {names} on {session.date()}")


def _values(units: FloatArray, prices: FloatArray) -> FloatArray:
    values = np.zeros_like(units)
    active = units > 0.0
    with np.errstate(over="ignore", invalid="ignore"):
        values[active] = units[active] * prices[active]
    if not np.isfinite(values).all():
        raise ValueError("nonfinite position valuation")
    return values


def _positive_nav(value: float) -> float:
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("portfolio NAV must remain finite and positive")
    return value


def _post_fee_capital(
    nav: float, old_values: FloatArray, weights: FloatArray, cost_rate: float
) -> float:
    if cost_rate == 0.0:
        return nav
    # N + c * sum(abs(w_i * N - B_i)) = pre-trade NAV. Keep the
    # lower bracket so solver residue cannot authorize an unfunded buy.
    lower, upper = 0.0, nav
    for _ in range(80):
        middle = lower + (upper - lower) / 2.0
        traded = float(np.abs(weights * middle - old_values).sum())
        if middle + cost_rate * traded <= nav:
            lower = middle
        else:
            upper = middle
    return lower


def _rebalance(
    units: FloatArray,
    old_values: FloatArray,
    *,
    cash: float,
    nav: float,
    weights: FloatArray,
    prices: FloatArray,
    cost_rate: float,
) -> tuple[FloatArray, float, FloatArray, FloatArray, float]:
    capital = _post_fee_capital(nav, old_values, weights, cost_rate)
    new_units = np.zeros_like(units)
    active = weights > 0.0
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        new_units[active] = weights[active] * capital / prices[active]
    if not np.isfinite(new_units).all():
        raise ValueError("nonfinite executed position units")
    changes = new_units - units
    traded_values = np.zeros_like(units)
    changed = changes != 0.0
    with np.errstate(over="ignore", invalid="ignore"):
        traded_values[changed] = changes[changed] * prices[changed]
    fee = cost_rate * float(np.abs(traded_values).sum())
    remaining = cash - float(traded_values.sum()) - fee
    tolerance = 1e-12 * max(1.0, nav)
    if (
        not np.isfinite(traded_values).all()
        or not np.isfinite(remaining)
        or not np.isfinite(fee)
        or remaining < -tolerance
    ):
        raise ValueError("rebalance is not finite and self-financing")
    remaining = max(0.0, remaining)
    actual_nav = _positive_nav(float(_values(new_units, prices).sum()) + remaining)
    if abs(actual_nav + fee - nav) > tolerance:
        raise ValueError("rebalance NAV does not reconcile with fees")
    return new_units, remaining, changes, traded_values, fee


def run_ledger(
    opens: pd.DataFrame,
    closes: pd.DataFrame,
    instructions: pd.DataFrame,
    *,
    capital: float,
    cost_rate: float,
    daily_rf: float,
    open_observed: pd.DataFrame,
    close_observed: pd.DataFrame,
) -> Ledger:
    """Apply opening instructions and then mark unchanged units at the close."""
    assets = list(opens.columns)
    rows, width = opens.shape
    px_open = opens.to_numpy(dtype="float64")
    px_close = closes.to_numpy(dtype="float64")
    fresh_open = open_observed.to_numpy(dtype=bool)
    fresh_close = close_observed.to_numpy(dtype=bool)
    targets = instructions.loc[:, [*assets, CASH_COLUMN]].to_numpy(dtype="float64")
    position_frames = {
        name: np.zeros((rows, width)) for name in ("positions", "traded_units", "traded_values")
    }
    weight_frames = {
        name: np.zeros((rows, width + 1)) for name in ("opening_weights", "closing_weights")
    }
    amounts = {
        name: np.zeros(rows)
        for name in (
            "cash_balance",
            "cash_interest",
            "nav_before_trade",
            "nav_after_trade",
            "nav",
            "fees",
        )
    }
    units = np.zeros(width)
    cash = capital
    adopted = np.zeros(width + 1)
    adopted[-1] = 1.0
    for row, session in enumerate(opens.index):
        interest = cash * daily_rf
        cash += interest
        _require_prices(px_open[row], fresh_open[row], units > 0.0, assets, session, mark="opening")
        old_values = _values(units, px_open[row])
        before = _positive_nav(cash + float(old_values.sum()))
        target = targets[row]
        if np.isfinite(target).all() and np.max(np.abs(target - adopted)) > _TRADE_THRESHOLD:
            required = (units > 0.0) | (target[:-1] > 0.0)
            _require_prices(
                px_open[row], fresh_open[row], required, assets, session, mark="opening"
            )
            units, cash, changes, traded, fee = _rebalance(
                units,
                old_values,
                cash=cash,
                nav=before,
                weights=target[:-1],
                prices=px_open[row],
                cost_rate=cost_rate,
            )
            position_frames["traded_units"][row] = changes
            position_frames["traded_values"][row] = traded
            amounts["fees"][row] = fee
            adopted = target.copy()
        opened = _values(units, px_open[row])
        after = _positive_nav(cash + float(opened.sum()))
        _require_prices(
            px_close[row], fresh_close[row], units > 0.0, assets, session, mark="closing"
        )
        closed = _values(units, px_close[row])
        closing_nav = _positive_nav(cash + float(closed.sum()))
        position_frames["positions"][row] = units
        weight_frames["opening_weights"][row] = np.append(opened, cash) / after
        weight_frames["closing_weights"][row] = np.append(closed, cash) / closing_nav
        for key, value in (
            ("cash_balance", cash),
            ("cash_interest", interest),
            ("nav_before_trade", before),
            ("nav_after_trade", after),
            ("nav", closing_nav),
        ):
            amounts[key][row] = value

    def frame(name: str) -> pd.DataFrame:
        return pd.DataFrame(position_frames[name], index=opens.index, columns=assets)

    def weights(name: str) -> pd.DataFrame:
        return pd.DataFrame(weight_frames[name], index=opens.index, columns=[*assets, CASH_COLUMN])

    def series(name: str) -> pd.Series:
        return pd.Series(amounts[name], index=opens.index, name=name)

    return Ledger(
        positions=frame("positions"),
        cash_balance=series("cash_balance"),
        cash_interest=series("cash_interest"),
        nav_before_trade=series("nav_before_trade"),
        nav_after_trade=series("nav_after_trade"),
        nav=series("nav"),
        opening_weights=weights("opening_weights"),
        closing_weights=weights("closing_weights"),
        traded_units=frame("traded_units"),
        traded_values=frame("traded_values"),
        fees=series("fees"),
    )
