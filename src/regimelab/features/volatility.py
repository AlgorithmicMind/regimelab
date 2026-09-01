"""Volatility features: range-based realised estimators and VIX term structure.

Range-based estimators
----------------------
Close-to-close volatility discards the intraday path and is consequently a noisy
estimator. Using the high-low range recovers much of that information:

- **Parkinson (1980)** uses the high-low range only. Roughly five times more
  efficient than close-to-close, but it cannot see overnight gaps and therefore
  *understates* volatility in gap-driven markets -- precisely the regimes this
  engine cares about.
- **Garman-Klass (1980)** adds the open-close move, reaching about eight times
  the efficiency of close-to-close. This is the default.

Both assume zero drift and continuous observation. Neither assumption holds
intraday, so both are biased low; the bias is stable enough that the *relative*
signal across regimes remains reliable, which is what the classifier uses.

VIX term structure
------------------
The VIX3M/VIX ratio is one of the cleanest available risk-appetite reads.

- **Contango** (VIX3M > VIX, positive spread): the normal state. Near-term
  implied volatility is below longer-dated, consistent with calm markets.
- **Backwardation** (VIX3M < VIX, negative spread): acute stress. Traders are
  paying more for immediate protection than for three-month protection.

The transition into backwardation is fast, typically days, which makes it a
genuine leading indicator rather than a confirmation of a drawdown already under
way. It is included as a *ratio* rather than a level difference so it is
comparable across volatility environments: a 5-point spread means something very
different at VIX 12 than at VIX 60.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd

from regimelab.features._rolling import annualisation_factor, require_series

__all__ = [
    "close_to_close_volatility",
    "garman_klass_volatility",
    "parkinson_volatility",
    "trailing_log_return",
    "vix_term_spread",
]

# 1 / (4 ln 2), the Parkinson normalisation constant.
_PARKINSON_SCALE = 1.0 / (4.0 * np.log(2.0))
# 2 ln 2 - 1, the Garman-Klass open-close correction coefficient.
_GK_CORRECTION = 2.0 * np.log(2.0) - 1.0


def _sanitised_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Log ratio guarding against non-positive prices.

    Non-positive prices are a data error, not a market event. Masking them to
    ``NaN`` keeps a single bad tick from poisoning an entire rolling window with
    ``-inf``.
    """
    safe_num = numerator.where(numerator > 0.0)
    safe_den = denominator.where(denominator > 0.0)
    return cast(pd.Series, np.log(safe_num / safe_den))


def parkinson_volatility(
    high: pd.Series,
    low: pd.Series,
    *,
    window: int = 21,
    trading_days: int = 252,
) -> pd.Series:
    """Annualised Parkinson range volatility.

    .. math::

        \\sigma^2 = \\frac{1}{4 \\ln 2} \\,
                    \\mathrm{mean}\\!\\left[\\ln(H_t / L_t)^2\\right]

    Parameters
    ----------
    high, low
        Session high and low prices.
    window
        Trailing window in sessions.
    trading_days
        Sessions per year, for annualisation.

    Returns
    -------
    pandas.Series
        Annualised volatility, named ``parkinson_vol``.
    """
    log_range = _sanitised_ratio(require_series(high, "high"), require_series(low, "low"))
    variance = (log_range**2).rolling(window, min_periods=window).mean() * _PARKINSON_SCALE
    result = cast(pd.Series, np.sqrt(variance)) * annualisation_factor(trading_days)
    return result.rename("parkinson_vol")


def garman_klass_volatility(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    *,
    window: int = 21,
    trading_days: int = 252,
) -> pd.Series:
    """Annualised Garman-Klass range volatility.

    .. math::

        \\sigma^2 = \\mathrm{mean}\\!\\left[
            \\tfrac{1}{2}\\ln(H/L)^2
            - (2\\ln 2 - 1)\\ln(C/O)^2
        \\right]

    Parameters
    ----------
    open_, high, low, close
        Session OHLC prices.
    window
        Trailing window in sessions.
    trading_days
        Sessions per year, for annualisation.

    Returns
    -------
    pandas.Series
        Annualised volatility, named ``realised_vol``.

    Notes
    -----
    The estimator can return a small negative variance when the open-close move
    dominates the range, which happens on gap days where ``O`` or ``C`` sits at
    the range boundary. Negative variances are clipped to zero rather than
    propagated as ``NaN``: dropping the observation would bias the window towards
    calm sessions, which is the opposite of what a risk system should do.
    """
    log_hl = _sanitised_ratio(require_series(high, "high"), require_series(low, "low"))
    log_co = _sanitised_ratio(require_series(close, "close"), require_series(open_, "open"))

    daily_variance = 0.5 * log_hl**2 - _GK_CORRECTION * log_co**2
    variance = daily_variance.rolling(window, min_periods=window).mean().clip(lower=0.0)
    result = cast(pd.Series, np.sqrt(variance)) * annualisation_factor(trading_days)
    return result.rename("realised_vol")


def close_to_close_volatility(
    close: pd.Series,
    *,
    window: int = 21,
    trading_days: int = 252,
) -> pd.Series:
    """Annualised close-to-close volatility of log returns.

    Retained as a baseline: the range estimators should agree with it on average
    and beat it on variance. When they disagree systematically, the high/low data
    is suspect.
    """
    log_returns = cast(pd.Series, np.log(require_series(close, "close"))).diff()
    daily = log_returns.rolling(window, min_periods=window).std(ddof=1)
    return (daily * annualisation_factor(trading_days)).rename("close_to_close_vol")


def vix_term_spread(vix: pd.Series, vix3m: pd.Series) -> pd.Series:
    """VIX term-structure slope as a ratio.

    .. math::

        \\text{spread} = \\frac{VIX3M}{VIX} - 1

    Positive is contango (calm), negative is backwardation (stress).

    Parameters
    ----------
    vix
        30-day implied volatility index.
    vix3m
        93-day implied volatility index.

    Returns
    -------
    pandas.Series
        Term spread, named ``vix_term_spread``.
    """
    near = require_series(vix, "vix")
    far = require_series(vix3m, "vix3m")
    safe_near = near.where(near > 0.0)
    return (far / safe_near - 1.0).rename("vix_term_spread")


def trailing_log_return(
    close: pd.Series,
    *,
    window: int = 21,
) -> pd.Series:
    """Cumulative log return over a trailing window.

    Uses ``diff(window)`` on log prices rather than ``pct_change``, so the value
    is additive across time and directly comparable to the volatility features
    the classifier sees alongside it.
    """
    log_price = cast(pd.Series, np.log(require_series(close, "close").where(lambda s: s > 0.0)))
    return log_price.diff(window).rename("trailing_return")


def volatility_features(
    ohlc: pd.DataFrame,
    vix: pd.Series | None = None,
    vix3m: pd.Series | None = None,
    *,
    window: int = 21,
    return_window: int = 21,
    trading_days: int = 252,
) -> pd.DataFrame:
    """Compute the volatility feature block for the risk asset.

    Parameters
    ----------
    ohlc
        Frame with ``open``, ``high``, ``low``, ``close`` columns for a single
        instrument.
    vix, vix3m
        Implied volatility indices. When either is absent the term spread is
        emitted as all-``NaN`` rather than omitted, so the feature matrix keeps a
        stable shape.
    window
        Trailing window for the realised-volatility estimators.
    return_window
        Trailing window for the cumulative return.
    trading_days
        Sessions per year.

    Returns
    -------
    pandas.DataFrame
        Columns ``realised_vol``, ``parkinson_vol``, ``trailing_return`` and
        ``vix_term_spread``.
    """
    missing = {"open", "high", "low", "close"} - set(ohlc.columns)
    if missing:
        msg = f"ohlc frame is missing columns: {sorted(missing)}"
        raise ValueError(msg)

    blocks = [
        garman_klass_volatility(
            ohlc["open"],
            ohlc["high"],
            ohlc["low"],
            ohlc["close"],
            window=window,
            trading_days=trading_days,
        ),
        parkinson_volatility(
            ohlc["high"],
            ohlc["low"],
            window=window,
            trading_days=trading_days,
        ),
        trailing_log_return(ohlc["close"], window=return_window),
    ]

    if vix is not None and vix3m is not None:
        blocks.append(vix_term_spread(vix, vix3m))
    else:
        blocks.append(
            pd.Series(np.nan, index=ohlc.index, name="vix_term_spread", dtype="float64"),
        )

    return pd.concat(blocks, axis="columns", sort=False)
