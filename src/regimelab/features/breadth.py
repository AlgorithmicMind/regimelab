"""Market breadth: participation and its rate of change.

Breadth answers a question the index level cannot: is the advance broad, or is it
a handful of names carrying a deteriorating market? Narrowing breadth during a
rising index is one of the more reliable pre-drawdown configurations.

.. warning::

   **Known limitation, stated plainly.** Textbook breadth is computed over index
   *constituents*. Doing that correctly requires point-in-time index membership,
   because using today's membership over history is survivorship bias: the
   constituents of the S&P 500 as of today are, by selection, the survivors, and
   their historical breadth is upward-biased.

   No free data source provides point-in-time membership. This module therefore
   computes breadth over a **fixed cross-section of sector ETFs**
   (``DataSettings.cross_section``). That cross-section has no survivorship bias
   -- the sector sleeves have continuous history and none were selected on
   performance -- at the cost of granularity: nine sector ETFs cannot express the
   dispersion that 500 constituents can, so the signal is muted relative to a
   true constituent-level breadth series.

   This is a deliberate trade of sensitivity for honesty. Swapping in a
   membership-aware provider would upgrade the signal without changing this
   module's interface.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "breadth_features",
    "breadth_momentum_series",
    "percent_above_sma",
]


def percent_above_sma(
    prices: pd.DataFrame,
    *,
    window: int = 200,
    min_assets: int = 3,
) -> pd.Series:
    """Share of the cross-section trading above its own trailing SMA.

    Parameters
    ----------
    prices
        Closing prices, dates by symbols.
    window
        SMA length in sessions.
    min_assets
        Minimum number of assets with a defined SMA required to report a value.
        Below this the reading is ``NaN``: breadth over two assets is not breadth.

    Returns
    -------
    pandas.Series
        Fraction in ``[0, 1]``, named ``breadth``.

    Notes
    -----
    The comparison is strict (``price > sma``). An asset exactly at its SMA is
    counted as *not* participating, which biases the measure marginally bearish
    at the boundary -- the correct direction for a risk indicator, and irrelevant
    for continuous prices.

    Each asset's SMA uses ``min_periods=window``, so an asset contributes only
    once its own SMA is fully warm. That prevents a newly-listed sleeve from
    entering the ratio on a two-day average.
    """
    if prices.empty:
        return pd.Series(dtype="float64", index=prices.index, name="breadth")

    sma = prices.rolling(window, min_periods=window).mean()

    # An asset is eligible only once both its price and its own SMA are known.
    eligible_mask = sma.notna() & prices.notna()
    participating_mask = (prices > sma) & eligible_mask

    eligible = eligible_mask.sum(axis="columns")
    participating = participating_mask.sum(axis="columns")

    # Mask the denominator *before* dividing. Masking afterwards still evaluates
    # the division, and integer division by zero raises in pandas 3.
    denominator = eligible.where(eligible >= min_assets).astype("float64")
    ratio = participating.astype("float64") / denominator
    return ratio.rename("breadth")


def breadth_momentum_series(
    breadth: pd.Series,
    *,
    window: int = 21,
) -> pd.Series:
    """Change in breadth over a trailing window.

    A deterioration oscillator. The *level* of breadth is regime-dependent and
    slow; the change is what flags a turn. Expressed as a simple difference
    (bounded in ``[-1, 1]``) rather than a percentage change, because breadth can
    legitimately approach zero and a ratio would explode there.

    Parameters
    ----------
    breadth
        Output of :func:`percent_above_sma`.
    window
        Trailing lookback in sessions.

    Returns
    -------
    pandas.Series
        Difference, named ``breadth_momentum``.
    """
    return breadth.diff(window).rename("breadth_momentum")


def breadth_thrust(
    breadth: pd.Series,
    *,
    fast: int = 10,
    slow: int = 63,
) -> pd.Series:
    """Fast-minus-slow breadth oscillator.

    Complements :func:`breadth_momentum_series`: a fixed-lag difference is
    sensitive to the single value ``window`` sessions ago, whereas a moving-average
    crossover is robust to one anomalous session. Both trailing.

    Parameters
    ----------
    breadth
        Breadth series.
    fast, slow
        Trailing window lengths; ``fast`` must be the shorter.

    Returns
    -------
    pandas.Series
        Oscillator, named ``breadth_thrust``.
    """
    if fast >= slow:
        msg = f"fast window {fast} must be shorter than slow window {slow}"
        raise ValueError(msg)

    fast_ma = breadth.rolling(fast, min_periods=fast).mean()
    slow_ma = breadth.rolling(slow, min_periods=slow).mean()
    return (fast_ma - slow_ma).rename("breadth_thrust")


def breadth_features(
    prices: pd.DataFrame,
    *,
    sma_window: int = 200,
    momentum_window: int = 21,
    min_assets: int = 3,
) -> pd.DataFrame:
    """Compute the breadth feature block.

    Parameters
    ----------
    prices
        Closing prices of the cross-section, dates by symbols.
    sma_window
        SMA length for the participation ratio.
    momentum_window
        Lookback for the momentum oscillator.
    min_assets
        Minimum eligible assets per session.

    Returns
    -------
    pandas.DataFrame
        Columns ``breadth`` and ``breadth_momentum``.
    """
    if prices.shape[1] < min_assets:
        empty = pd.Series(np.nan, index=prices.index, dtype="float64")
        return pd.DataFrame(
            {
                "breadth": empty,
                "breadth_momentum": empty,
            },
        )

    breadth = percent_above_sma(prices, window=sma_window, min_assets=min_assets)
    momentum = breadth_momentum_series(breadth, window=momentum_window)
    return pd.concat([breadth, momentum], axis="columns", sort=False)
