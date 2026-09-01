"""Trailing-window machinery shared by the feature families.

Every reduction here reads ``values[t - window + 1 : t + 1]`` and writes to
position ``t``. That asymmetry is the causality guarantee: a feature physically
cannot see beyond its own timestamp, so look-ahead is not a discipline that has
to be maintained, it is a property of the loop bounds.

The property-based tests in ``tests/test_causality.py`` verify this empirically
by mutating the tail of the input and asserting the head of the output is
unchanged.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

__all__ = ["rolling_matrix_reduce", "usable_columns"]


def usable_columns(
    frame: pd.DataFrame,
    *,
    min_coverage: float = 0.95,
) -> list[str]:
    """Columns with enough non-null coverage to enter a matrix computation.

    A column that is mostly missing does not merely add noise to a covariance
    estimate, it changes the estimator: pairwise-complete covariance matrices
    built from unequal samples are not guaranteed positive semi-definite, and
    the eigenvalues then stop being interpretable as variance shares.

    Parameters
    ----------
    frame
        Candidate matrix, dates by symbols.
    min_coverage
        Minimum fraction of non-null observations required.

    Returns
    -------
    list of str
        Qualifying column names, in the frame's original order.
    """
    if frame.empty:
        return []
    coverage = frame.notna().mean(axis="index")
    return [str(c) for c in frame.columns[coverage >= min_coverage]]


def rolling_matrix_reduce(
    frame: pd.DataFrame,
    window: int,
    reducer: Callable[[np.ndarray], float],
    *,
    min_columns: int = 2,
    name: str | None = None,
) -> pd.Series:
    """Apply a matrix-valued reduction over trailing windows.

    ``pandas.rolling`` reduces column-wise and so cannot express a computation
    that needs the whole window as a matrix -- eigen-decompositions and
    correlation matrices among them. This fills that gap while preserving the
    trailing-window semantics.

    Parameters
    ----------
    frame
        Dates by symbols. Column order is preserved and matters: it determines
        the ordering of the matrix passed to ``reducer``.
    window
        Number of trailing rows per window, inclusive of the current row.
    reducer
        Maps a ``(window, n_columns)`` float array with no missing values to a
        scalar. Called once per output position.
    min_columns
        Minimum number of complete columns required. Windows with fewer produce
        ``NaN`` rather than a degenerate result.
    name
        Name of the returned series.

    Returns
    -------
    pandas.Series
        Float series on ``frame.index``. The first ``window - 1`` entries are
        ``NaN``: there is no history to reduce, and fabricating a partial-window
        value would silently change the estimator.
    """
    if window < 2:
        msg = f"window must be at least 2, got {window}"
        raise ValueError(msg)

    index = frame.index
    result = np.full(len(index), np.nan, dtype="float64")

    if len(index) < window or frame.shape[1] < min_columns:
        return pd.Series(result, index=index, name=name)

    values = frame.to_numpy(dtype="float64", copy=False)

    for position in range(window - 1, len(index)):
        block = values[position - window + 1 : position + 1]

        # Drop columns with any gap inside this window. Done per window, not
        # globally, so a symbol that starts mid-sample contributes as soon as it
        # has a full window instead of being excluded outright.
        complete = ~np.isnan(block).any(axis=0)
        if int(complete.sum()) < min_columns:
            continue

        selected = block[:, complete]
        if not np.isfinite(selected).all():
            continue

        result[position] = reducer(selected)

    return pd.Series(result, index=index, name=name)


def mean_offdiagonal(matrix: np.ndarray) -> float:
    """Mean of the strictly off-diagonal entries of a square matrix.

    Parameters
    ----------
    matrix
        Square array, typically a correlation matrix.

    Returns
    -------
    float
        Mean off-diagonal value, or ``nan`` for a 1x1 matrix where no pair exists.
    """
    n = int(matrix.shape[0])
    if n < 2:
        return float("nan")
    total = float(matrix.sum() - np.trace(matrix))
    return float(total / (n * (n - 1)))


def standardised_shift(
    series: pd.Series,
    short_window: int,
    long_window: int,
    *,
    name: str | None = None,
) -> pd.Series:
    """Standardised difference between a short and a long trailing mean.

    Implements the transformation Kritzman & Li (2010) apply to the absorption
    ratio::

        delta = (mean_short - mean_long) / std_long

    Both windows and the scaling standard deviation are trailing, so the whole
    quantity is observable at ``t``. Using a full-sample standard deviation here
    -- a common shortcut -- would leak the future variance of the signal into
    every historical value.

    Parameters
    ----------
    series
        Input signal.
    short_window, long_window
        Trailing window lengths. ``short_window`` must be the shorter of the two.
    name
        Name of the returned series.

    Returns
    -------
    pandas.Series
        Standardised shift, ``NaN`` where either window is not yet full or where
        the trailing standard deviation is zero.
    """
    if short_window >= long_window:
        msg = f"short_window {short_window} must be less than long_window {long_window}"
        raise ValueError(msg)

    short_mean = series.rolling(short_window, min_periods=short_window).mean()
    long_mean = series.rolling(long_window, min_periods=long_window).mean()
    long_std = series.rolling(long_window, min_periods=long_window).std(ddof=1)

    # A zero trailing standard deviation makes the ratio meaningless rather than
    # infinite; report absence.
    denominator = long_std.where(long_std > 0.0)
    shift = (short_mean - long_mean) / denominator
    return shift.rename(name)


def require_series(values: pd.Series | pd.DataFrame, label: str) -> pd.Series:
    """Coerce a one-column frame to a series, rejecting anything wider.

    Guards the feature entry points against a caller passing ``frame[["SPY"]]``
    where ``frame["SPY"]`` was meant -- an error that otherwise surfaces much
    later as a confusing broadcast.
    """
    if isinstance(values, pd.Series):
        return values
    if values.shape[1] == 1:
        return values.iloc[:, 0]
    msg = f"{label} expects a single series, got {values.shape[1]} columns"
    raise ValueError(msg)


def annualisation_factor(trading_days: int) -> float:
    """Square-root-of-time scaler from daily to annual volatility."""
    return float(np.sqrt(trading_days))


def select_cross_section(
    frame: pd.DataFrame,
    symbols: Sequence[str] | None,
) -> pd.DataFrame:
    """Restrict a frame to ``symbols``, preserving the requested order.

    Deterministic column ordering is a correctness requirement, not a style
    preference: eigen-decompositions and covariance parameterisations are
    sensitive to it.
    """
    if symbols is None:
        return frame.loc[:, sorted(str(c) for c in frame.columns)]
    wanted = [s for s in symbols if s in frame.columns]
    return frame.loc[:, wanted]
