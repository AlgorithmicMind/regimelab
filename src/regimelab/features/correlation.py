"""Dynamic cross-asset correlation.

Correlation is not a constant to be estimated once; it is a state variable. In
calm markets equities, duration and gold behave as three distinct risk sources.
In a liquidation event they behave as one, because the marginal seller is
liquidating everything simultaneously. Diversification therefore fails exactly
when it is needed, and a static covariance assumption understates tail risk by a
wide margin.

Two complementary readings are produced:

- **Mean pairwise correlation** over a trailing window: the headline coupling
  measure, directly interpretable.
- **Correlation of the defensive sleeve to the risk asset**: the specific
  question a multi-asset allocator needs answered. When bond-equity correlation
  turns positive -- 2022 being the canonical episode -- a duration hedge stops
  hedging, and the correct defensive posture becomes cash. The backtest's
  ``RISK_OFF`` weights encode that.

Why not DCC-GARCH
-----------------
A DCC model would give a better-conditioned, faster-reacting correlation
estimate. It is deliberately out of scope: it adds a second layer of estimated
parameters that must themselves be refit inside the walk-forward loop, and the
resulting sensitivity to that refit schedule is difficult to disentangle from
regime signal. A trailing sample correlation is cruder, fully transparent, and
its failure mode -- slow reaction -- is known and bounded.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from regimelab.features._rolling import mean_offdiagonal, rolling_matrix_reduce

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = [
    "correlation_features",
    "mean_pairwise_correlation",
    "pairwise_correlation",
    "rolling_correlation_matrix",
]


def _correlation_reducer(block: np.ndarray) -> float:
    """Mean off-diagonal correlation of a window.

    Columns with zero variance inside the window are dropped first: a constant
    series has undefined correlation, and ``numpy.corrcoef`` would emit ``NaN``
    for the entire row and column, contaminating the mean.
    """
    variances = block.var(axis=0, ddof=1)
    keep = variances > 0.0
    if int(keep.sum()) < 2:
        return float("nan")

    matrix = np.corrcoef(block[:, keep], rowvar=False)
    if not np.isfinite(matrix).all():
        return float("nan")
    return mean_offdiagonal(matrix)


def mean_pairwise_correlation(
    returns: pd.DataFrame,
    *,
    window: int = 63,
    min_assets: int = 3,
) -> pd.Series:
    """Rolling mean of the off-diagonal correlation matrix entries.

    Parameters
    ----------
    returns
        Daily log returns, dates by symbols.
    window
        Trailing window in sessions. 63 is one quarter: long enough to estimate
        a correlation matrix of this width, short enough to register a regime
        change within weeks.
    min_assets
        Minimum complete columns per window.

    Returns
    -------
    pandas.Series
        Mean pairwise correlation in ``[-1, 1]``, named ``mean_correlation``.
    """
    return rolling_matrix_reduce(
        returns,
        window=window,
        reducer=_correlation_reducer,
        min_columns=min_assets,
        name="mean_correlation",
    )


def pairwise_correlation(
    left: pd.Series,
    right: pd.Series,
    *,
    window: int = 63,
) -> pd.Series:
    """Rolling correlation between two return series.

    Parameters
    ----------
    left, right
        Daily return series, aligned on a common index.
    window
        Trailing window in sessions.

    Returns
    -------
    pandas.Series
        Correlation, named ``pair_correlation``.
    """
    aligned_left, aligned_right = left.align(right, join="inner")
    correlation = aligned_left.rolling(window, min_periods=window).corr(aligned_right)
    return correlation.rename("pair_correlation")


def rolling_correlation_matrix(
    returns: pd.DataFrame,
    asof: pd.Timestamp,
    *,
    window: int = 63,
) -> pd.DataFrame:
    """Correlation matrix estimated on the window ending at ``asof``.

    Returned for reporting, not for the feature vector: the dashboard renders it
    as a heat map, which communicates a correlation regime far better than a
    single averaged number.

    Parameters
    ----------
    returns
        Daily log returns, dates by symbols.
    asof
        Right edge of the window, inclusive.
    window
        Window length in sessions.

    Returns
    -------
    pandas.DataFrame
        Symmetric correlation matrix over the symbols with complete data in the
        window. Empty if the window is not full.
    """
    block = returns.loc[:asof].iloc[-window:]
    if len(block) < window:
        return pd.DataFrame()

    complete = block.dropna(axis="columns", how="any")
    varying = complete.loc[:, complete.var(ddof=1) > 0.0]
    if varying.shape[1] < 2:
        return pd.DataFrame()

    return varying.corr()


def correlation_features(
    returns: pd.DataFrame,
    *,
    window: int = 63,
    risk_asset: str | None = None,
    defensive_assets: Sequence[str] = (),
    min_assets: int = 3,
) -> pd.DataFrame:
    """Compute the correlation feature block.

    Parameters
    ----------
    returns
        Daily log returns, dates by symbols.
    window
        Trailing window in sessions.
    risk_asset
        Symbol of the primary risk asset, for the defensive-correlation reading.
    defensive_assets
        Symbols whose correlation to ``risk_asset`` is averaged into
        ``defensive_correlation``.
    min_assets
        Minimum complete columns for the mean pairwise measure.

    Returns
    -------
    pandas.DataFrame
        Columns ``mean_correlation`` and, when computable,
        ``defensive_correlation``.
    """
    blocks: list[pd.Series] = [
        mean_pairwise_correlation(returns, window=window, min_assets=min_assets),
    ]

    usable_defensive = [
        symbol
        for symbol in defensive_assets
        if risk_asset is not None and symbol in returns.columns and symbol != risk_asset
    ]

    if risk_asset in returns.columns and usable_defensive:
        pairs = [
            pairwise_correlation(returns[symbol], returns[risk_asset], window=window)
            for symbol in usable_defensive
        ]
        defensive = pd.concat(pairs, axis="columns", sort=False).mean(axis="columns", skipna=True)
        blocks.append(defensive.rename("defensive_correlation"))
    else:
        blocks.append(
            pd.Series(
                np.nan,
                index=returns.index,
                name="defensive_correlation",
                dtype="float64",
            ),
        )

    return pd.concat(blocks, axis="columns", sort=False)
