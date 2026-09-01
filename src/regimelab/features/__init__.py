"""Feature families and the causal feature matrix that assembles them.

All four families are computed as full time series with trailing windows, then
sliced by the as-of cursor -- rather than recomputed inside a per-date loop. The
two are numerically identical because every window is trailing, and the
vectorised form is orders of magnitude faster, which is what makes a multi-decade
walk-forward tractable.

That equivalence is not taken on trust. ``tests/test_causality.py`` asserts it by
mutating the tail of the panel and requiring the head of the feature matrix to be
bit-identical.
"""

from __future__ import annotations

import datetime as dt
from typing import TYPE_CHECKING

import pandas as pd

from regimelab.features.absorption import (
    absorption_delta_series,
    absorption_features,
    absorption_ratio_series,
)
from regimelab.features.breadth import (
    breadth_features,
    breadth_momentum_series,
    percent_above_sma,
)
from regimelab.features.correlation import (
    correlation_features,
    mean_pairwise_correlation,
    rolling_correlation_matrix,
)
from regimelab.features.volatility import (
    close_to_close_volatility,
    garman_klass_volatility,
    parkinson_volatility,
    trailing_log_return,
    vix_term_spread,
    volatility_features,
)
from regimelab.models.types import FeatureSnapshot

if TYPE_CHECKING:
    from regimelab.data.alignment import AlignedPanel
    from regimelab.settings import Settings

__all__ = [
    "FEATURE_COLUMNS",
    "absorption_delta_series",
    "absorption_features",
    "absorption_ratio_series",
    "breadth_features",
    "breadth_momentum_series",
    "build_feature_matrix",
    "close_to_close_volatility",
    "correlation_features",
    "garman_klass_volatility",
    "mean_pairwise_correlation",
    "parkinson_volatility",
    "percent_above_sma",
    "rolling_correlation_matrix",
    "snapshot_at",
    "trailing_log_return",
    "vix_term_spread",
    "volatility_features",
]

FEATURE_COLUMNS: tuple[str, ...] = tuple(FeatureSnapshot.model_fields)
"""Canonical feature-matrix columns, in the order declared on the snapshot."""

_OHLC_FIELDS = ("open", "high", "low", "close")


def build_feature_matrix(panel: AlignedPanel, settings: Settings) -> pd.DataFrame:
    """Compute the full causal feature matrix for an aligned panel.

    Parameters
    ----------
    panel
        Point-in-time aligned panel.
    settings
        Resolved configuration supplying window lengths and the universe.

    Returns
    -------
    pandas.DataFrame
        Indexed by the panel's master calendar, with the columns of
        :data:`FEATURE_COLUMNS` plus the diagnostic columns ``parkinson_vol`` and
        ``defensive_correlation``. Leading rows are ``NaN`` until each feature's
        window is warm; they are not dropped here, because where the usable
        sample begins is the pipeline's decision, not the feature layer's.
    """
    feature_cfg = settings.features
    data_cfg = settings.data
    available = set(panel.symbols)

    blocks: list[pd.DataFrame] = []

    # --- Systemic fragility ------------------------------------------------
    cross_section = [s for s in data_cfg.cross_section if s in available]
    if len(cross_section) >= 3:
        cross_returns = panel.log_returns(cross_section)
        blocks.append(
            absorption_features(
                cross_returns,
                window=feature_cfg.absorption_window,
                eigen_fraction=feature_cfg.absorption_eigen_fraction,
                short_window=feature_cfg.absorption_delta_short,
                long_window=feature_cfg.absorption_delta_long,
            ),
        )

        # --- Participation -------------------------------------------------
        blocks.append(
            breadth_features(
                panel.close(cross_section),
                sma_window=feature_cfg.breadth_sma,
                momentum_window=feature_cfg.breadth_momentum_window,
            ),
        )

    # --- Volatility and term structure -------------------------------------
    risk_asset = data_cfg.benchmark
    if risk_asset in available:
        ohlc = pd.DataFrame(
            {field: panel.field(field, [risk_asset])[risk_asset] for field in _OHLC_FIELDS},
        )
        vix = (
            panel.close([data_cfg.vix_symbol])[data_cfg.vix_symbol]
            if data_cfg.vix_symbol in available
            else None
        )
        vix3m = (
            panel.close([data_cfg.vix3m_symbol])[data_cfg.vix3m_symbol]
            if data_cfg.vix3m_symbol in available
            else None
        )
        blocks.append(
            volatility_features(
                ohlc,
                vix,
                vix3m,
                window=feature_cfg.volatility_window,
                return_window=feature_cfg.trailing_return_window,
                trading_days=feature_cfg.trading_days_per_year,
            ),
        )

    # --- Cross-asset coupling ----------------------------------------------
    allocation_universe = [s for s in data_cfg.risk_assets if s in available]
    if len(allocation_universe) >= 2:
        blocks.append(
            correlation_features(
                panel.log_returns(allocation_universe),
                window=feature_cfg.correlation_window,
                risk_asset=risk_asset,
                defensive_assets=[s for s in allocation_universe if s != risk_asset],
                min_assets=min(3, len(allocation_universe)),
            ),
        )

    if not blocks:
        matrix = pd.DataFrame(index=panel.calendar, columns=list(FEATURE_COLUMNS), dtype="float64")
    else:
        matrix = pd.concat(blocks, axis="columns", sort=False)

    # Reindex onto the master calendar: log_returns drops the first session, and
    # every feature must share one index for the model layer to stack them.
    matrix = matrix.reindex(index=panel.calendar)

    for column in FEATURE_COLUMNS:
        if column not in matrix.columns:
            matrix[column] = pd.Series(pd.NA, index=matrix.index, dtype="float64")

    ordered = [*FEATURE_COLUMNS, *sorted(set(matrix.columns) - set(FEATURE_COLUMNS))]
    matrix = matrix.loc[:, ordered].astype("float64")
    matrix.index.name = "date"
    return matrix


def snapshot_at(matrix: pd.DataFrame, asof: dt.date | pd.Timestamp) -> FeatureSnapshot:
    """Extract the validated feature snapshot observable at ``asof``.

    Parameters
    ----------
    matrix
        Output of :func:`build_feature_matrix`.
    asof
        Decision date. The most recent row at or before this date is used.

    Returns
    -------
    FeatureSnapshot
        Validated snapshot. Fields whose window is not yet warm are ``None``.

    Raises
    ------
    KeyError
        If the matrix has no row at or before ``asof``.
    """
    truncated = matrix.loc[: pd.Timestamp(asof)]
    if truncated.empty:
        msg = f"feature matrix has no observations at or before {asof}"
        raise KeyError(msg)

    row = truncated.iloc[-1]
    values: dict[str, float | None] = {}
    for column in FEATURE_COLUMNS:
        value = row.get(column)
        # NaN means "window not warm" and must reach the schema as None; a
        # silently coerced 0.0 would be indistinguishable from a real reading.
        values[column] = None if value is None or pd.isna(value) else float(value)

    return FeatureSnapshot(**values)
