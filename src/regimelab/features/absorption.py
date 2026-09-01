"""Absorption ratio: PCA-based measure of systemic fragility.

Kritzman, Li, Page & Rigobon (2010), *Principal Components as a Measure of
Systemic Risk*, Journal of Portfolio Management.

Definition
----------
For a cross-section of :math:`N` assets with sample covariance matrix
:math:`\\Sigma` and eigenvalues :math:`\\lambda_1 \\ge \\dots \\ge \\lambda_N`,
the absorption ratio over the leading :math:`n` eigenvectors is

.. math::

    AR_n = \\frac{\\sum_{i=1}^{n} \\lambda_i}{\\sum_{j=1}^{N} \\lambda_j}

The denominator is the trace of :math:`\\Sigma`, i.e. the total variance of the
cross-section. :math:`AR` is therefore the fraction of total variance absorbed
by the leading components, bounded in :math:`[n/N, 1]`.

Interpretation
--------------
A high absorption ratio means the cross-section is tightly coupled: few
independent sources of risk, so a shock propagates instead of being diversified
away. It is a *fragility* measure, not a direction forecast. Markets can sit at
a high AR for months while continuing to rally.

The tradable signal is the **shift**, not the level. Kritzman & Li standardise a
15-day mean of AR against its trailing one-year mean and standard deviation; a
shift above roughly one standard deviation has historically preceded drawdowns,
while a sharp negative shift has preceded rallies. Both windows here are
trailing, so the standardisation itself introduces no look-ahead -- see
:func:`regimelab.features._rolling.standardised_shift`.

Implementation notes
--------------------
- Eigenvalues come from :func:`numpy.linalg.eigvalsh`, which exploits symmetry
  and returns real, ascending values. Using the general ``eigvals`` on a
  covariance matrix invites tiny imaginary residuals that then need cleaning up.
- The covariance matrix is computed on log returns, not prices.
- Columns are *not* standardised before decomposition. Kritzman & Li decompose
  the covariance matrix, so a high-variance asset legitimately dominates. Passing
  a correlation matrix instead is a different (also defensible) statistic, but it
  is not the published absorption ratio, and mixing them up makes results
  incomparable to the literature.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from regimelab.features._rolling import (
    rolling_matrix_reduce,
    standardised_shift,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = [
    "absorption_delta_series",
    "absorption_ratio_series",
    "leading_eigenvector_count",
]


def leading_eigenvector_count(n_assets: int, fraction: float = 0.2) -> int:
    """Number of eigenvectors treated as "leading".

    Kritzman & Li use :math:`n = N/5`. Rounded up and clamped to
    :math:`[1, N-1]`: ``n = N`` would make the ratio identically one and destroy
    the signal, which is an easy configuration mistake to make with a small
    cross-section.

    Parameters
    ----------
    n_assets
        Size of the cross-section.
    fraction
        Share of eigenvectors to retain.

    Returns
    -------
    int
        Eigenvector count, at least 1 and at most ``n_assets - 1``.
    """
    if n_assets < 2:
        msg = f"absorption ratio needs at least 2 assets, got {n_assets}"
        raise ValueError(msg)
    if not 0.0 < fraction <= 1.0:
        msg = f"fraction must lie in (0, 1], got {fraction}"
        raise ValueError(msg)

    count = math.ceil(fraction * n_assets)
    return int(min(max(count, 1), n_assets - 1))


def _absorption_reducer(fraction: float) -> object:
    """Build a window reducer computing the absorption ratio.

    Closes over ``fraction`` so the eigenvector count adapts to however many
    columns survived the window's completeness filter.
    """

    def reduce_window(block: np.ndarray) -> float:
        n_assets = block.shape[1]
        if n_assets < 2:
            return float("nan")

        # rowvar=False: rows are observations, columns are assets.
        covariance = np.cov(block, rowvar=False, ddof=1)
        total_variance = float(np.trace(covariance))
        if not np.isfinite(total_variance) or total_variance <= 0.0:
            return float("nan")

        eigenvalues = np.linalg.eigvalsh(covariance)
        # eigvalsh returns ascending order; clip the small negative values that
        # floating-point error produces on near-singular matrices.
        eigenvalues = np.clip(eigenvalues[::-1], 0.0, None)

        n_leading = leading_eigenvector_count(n_assets, fraction)
        absorbed = float(eigenvalues[:n_leading].sum())
        return absorbed / total_variance

    return reduce_window


def absorption_ratio_series(
    returns: pd.DataFrame,
    *,
    window: int = 500,
    eigen_fraction: float = 0.2,
    min_assets: int = 3,
) -> pd.Series:
    """Rolling absorption ratio of a return cross-section.

    Parameters
    ----------
    returns
        Daily log returns, dates by symbols. Column order is preserved.
    window
        Trailing covariance estimation window in sessions. 500 follows the paper.
    eigen_fraction
        Share of eigenvectors treated as leading.
    min_assets
        Minimum complete columns per window; below this the window yields ``NaN``.

    Returns
    -------
    pandas.Series
        Absorption ratio in ``[0, 1]``, named ``absorption_ratio``. ``NaN`` for
        the first ``window - 1`` sessions.
    """
    if returns.shape[1] < min_assets:
        return pd.Series(
            np.nan,
            index=returns.index,
            name="absorption_ratio",
            dtype="float64",
        )

    return rolling_matrix_reduce(
        returns,
        window=window,
        reducer=_absorption_reducer(eigen_fraction),  # type: ignore[arg-type]
        min_columns=min_assets,
        name="absorption_ratio",
    )


def absorption_delta_series(
    absorption_ratio: pd.Series,
    *,
    short_window: int = 15,
    long_window: int = 252,
) -> pd.Series:
    """Standardised shift in the absorption ratio.

    .. math::

        \\Delta AR = \\frac{\\overline{AR}_{15} - \\overline{AR}_{252}}
                          {\\sigma(AR_{252})}

    This, rather than the level, is what the paper trades. A reading above
    :math:`+1` indicates risk is concentrating faster than its recent norm.

    Parameters
    ----------
    absorption_ratio
        Output of :func:`absorption_ratio_series`.
    short_window, long_window
        Trailing windows for the fast mean and the slow mean/standard deviation.

    Returns
    -------
    pandas.Series
        Standardised shift, named ``absorption_delta``.
    """
    return standardised_shift(
        absorption_ratio,
        short_window=short_window,
        long_window=long_window,
        name="absorption_delta",
    )


def absorption_features(
    returns: pd.DataFrame,
    *,
    window: int = 500,
    eigen_fraction: float = 0.2,
    short_window: int = 15,
    long_window: int = 252,
    symbols: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Compute both absorption features in one pass.

    Parameters
    ----------
    returns
        Daily log returns, dates by symbols.
    window, eigen_fraction
        Passed to :func:`absorption_ratio_series`.
    short_window, long_window
        Passed to :func:`absorption_delta_series`.
    symbols
        Optional explicit cross-section, in order.

    Returns
    -------
    pandas.DataFrame
        Columns ``absorption_ratio`` and ``absorption_delta``.
    """
    cross_section = returns if symbols is None else returns.loc[:, list(symbols)]
    ratio = absorption_ratio_series(
        cross_section,
        window=window,
        eigen_fraction=eigen_fraction,
    )
    delta = absorption_delta_series(
        ratio,
        short_window=short_window,
        long_window=long_window,
    )
    return pd.concat([ratio, delta], axis="columns", sort=False)
