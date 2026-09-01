"""Walk-forward evaluation.

Point-in-time walk-forward means the model is refit on a training window ending
at :math:`t_k` and used to classify sessions *after* :math:`t_k` but *before* the
next refit. Crucially, the parameters estimated at :math:`t_k` are **frozen**
while the filter is advanced: the model cannot see any observation used in its
out-of-sample classification.

The loop also reuses the filtered posterior at the end of one training window as
the prior for the next segment, so the state continuity is preserved across
refits rather than resetting to the stationary distribution each time. This
matters for HMM-style classifiers: an abrupt reset would make the first few
out-of-sample days of every fold behave like a cold start, biasing turnover and
drawdown estimates.

The default is an **expanding window**: each refit uses all data from the start
of the panel up to the refit date. A fixed rolling window would discard
recession data during calm periods, which is exactly when the model most needs
to remember what tail risk looks like.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from regimelab.backtest.engine import run_backtest as run_backtest_engine
from regimelab.models.registry import build_classifier
from regimelab.models.types import MarketRegime, ModelMetadata, RegimeProbabilities

if TYPE_CHECKING:
    from regimelab.backtest.engine import BacktestResult
    from regimelab.data.alignment import AlignedPanel
    from regimelab.models.base import FittedRegime
    from regimelab.settings import Settings

__all__ = ["WalkForwardFold", "WalkForwardResult", "run_walk_forward"]


@dataclass(frozen=True, slots=True)
class WalkForwardFold:
    """Metadata for one walk-forward fold."""

    fold_index: int
    train_start: dt.date
    train_end: dt.date
    test_start: dt.date
    test_end: dt.date
    n_train: int
    n_test: int
    model: ModelMetadata
    assignment: pd.DataFrame | None = None


@dataclass(frozen=True, slots=True)
class WalkForwardResult:
    """Combined walk-forward output.

    Attributes
    ----------
    labels
        Modal regime per session across all folds.
    probabilities
        Filtered posterior per session across all folds.
    confidence
        Modal posterior confidence per session.
    folds
        Per-fold metadata.
    features
        Feature matrix used by the classifiers.
    backtest
        Optional backtest result when prices are supplied.
    """

    labels: pd.Series
    probabilities: pd.DataFrame
    confidence: pd.Series
    folds: tuple[WalkForwardFold, ...]
    features: pd.DataFrame
    backtest: BacktestResult | None = None

    @property
    def regimes(self) -> pd.Series:
        """Alias for ``labels``."""
        return self.labels

    def regime_probabilities(self, asof: dt.date | pd.Timestamp) -> RegimeProbabilities:
        """Validated posterior at the most recent date at or before ``asof``."""
        truncated = self.probabilities.loc[: pd.Timestamp(asof)]
        if truncated.empty:
            msg = f"no walk-forward probability at or before {asof}"
            raise KeyError(msg)
        row = truncated.iloc[-1]
        return RegimeProbabilities(
            probabilities={MarketRegime(str(name)): float(value) for name, value in row.items()},
        )


def _refit_dates(features: pd.DataFrame, settings: Settings) -> list[pd.Timestamp]:
    """Training window end dates for each fold.

    The first fold uses the earliest feature row that satisfies
    ``min_train_observations`` and the feature warmup. Subsequent folds advance by
    ``model.refit_frequency_days``.
    """
    full = features.dropna(how="any")
    if len(full) < settings.model.min_train_observations:
        msg = (
            f"feature matrix has only {len(full)} complete rows, "
            f"below min_train_observations {settings.model.min_train_observations}"
        )
        raise ValueError(msg)

    first = settings.model.min_train_observations - 1
    starts = full.index[first :: settings.backtest.refit_frequency_days]

    # Always include the final observation as a refit point so the last fold
    # reaches the end of the sample.
    if full.index[-1] not in starts:
        starts = starts.append(pd.DatetimeIndex([full.index[-1]]))

    return list(starts)


def _filter_segment(
    fit: FittedRegime,
    features: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    initial: np.ndarray | None,
) -> pd.DataFrame:
    """Filter a span from ``start`` to ``end`` with the provided initial state.

    The initial distribution is the filtered posterior at the end of the
    preceding training or test segment, preserving Markov continuity.
    """
    segment = features.loc[start:end]
    return fit.probabilities(segment, initial=initial)


def run_walk_forward(
    panel: AlignedPanel,
    features: pd.DataFrame,
    settings: Settings,
    *,
    run_backtest: bool = True,
) -> WalkForwardResult:
    """Run the walk-forward regime classification loop.

    Parameters
    ----------
    panel
        Aligned price panel.
    features
        Full causal feature matrix.
    settings
        Resolved configuration.
    run_backtest
        Whether to run the regime-conditional backtest on the resulting path.

    Returns
    -------
    WalkForwardResult
        Combined out-of-sample regime path and optional backtest.
    """

    benchmark = settings.data.benchmark
    reference_returns = panel.log_returns([benchmark])[benchmark]
    model = build_classifier(settings.model)

    full = features.dropna(how="any")
    refit_dates = _refit_dates(features, settings)

    if len(refit_dates) < 2:
        msg = (
            "walk-forward requires at least two refit points: one to fit and one "
            "to evaluate. Increase the date range or lower min_train_observations."
        )
        raise ValueError(msg)

    all_probabilities: list[pd.DataFrame] = []
    all_folds: list[WalkForwardFold] = []

    # Start the filter from the stationary distribution on the first fold; for
    # subsequent folds use the posterior at the previous refit date.
    initial: np.ndarray | None = None

    for fold_index, train_end in enumerate(refit_dates):
        is_last = fold_index == len(refit_dates) - 1

        # Out-of-sample segment is (prev_train_end, current_train_end], except
        # for the final fold, which is (prev_train_end, last_feature_date].
        test_start = full.index[0] if fold_index == 0 else refit_dates[fold_index - 1]

        test_end = full.index[-1] if is_last else refit_dates[fold_index + 1]

        # Training always starts from the earliest complete feature row.
        train_start = full.index[0]

        fit = model.fit(features, reference_returns, train_end=train_end)

        # Filter the segment. If this is not the last fold, it is exactly one
        # refit step long; if it is the last, it runs to the end of the data.
        probs = _filter_segment(fit, features, test_start, test_end, initial=initial)

        # Save the posterior at the end of this segment as the next prior.
        if not probs.empty:
            initial = probs.iloc[-1].to_numpy(dtype="float64")
            # First fold only: we keep the entire segment. Subsequent folds
            # overlap at test_start with the previous fold's end; drop the first
            # row to avoid duplicating the refit boundary.
            if fold_index > 0:
                probs = probs.iloc[1:]
            all_probabilities.append(probs)

        all_folds.append(
            WalkForwardFold(
                fold_index=fold_index,
                train_start=train_start.date(),
                train_end=train_end.date(),
                test_start=test_start.date(),
                test_end=test_end.date(),
                n_train=features.loc[train_start:train_end].dropna(how="any").shape[0],
                n_test=probs.shape[0],
                model=fit.metadata,
                assignment=fit.assignment.describe() if not probs.empty else None,
            ),
        )

    if not all_probabilities:
        msg = "walk-forward produced no probability path"
        raise ValueError(msg)

    probabilities = pd.concat(all_probabilities, sort=False)
    probabilities = probabilities[~probabilities.index.duplicated(keep="last")].sort_index()

    # Modal regime with the canonical risk-off tie-break.
    ranked = sorted(probabilities.columns, key=lambda c: MarketRegime(str(c)).risk_rank)
    modal = probabilities.loc[:, ranked[::-1]].idxmax(axis="columns")
    labels = pd.Series(
        [MarketRegime(str(value)) for value in modal],
        index=modal.index,
        dtype="object",
        name="regime",
    )
    confidence = probabilities.max(axis="columns").rename("confidence")

    result = WalkForwardResult(
        labels=labels,
        probabilities=probabilities,
        confidence=confidence,
        folds=tuple(all_folds),
        features=features,
        backtest=None,
    )

    if run_backtest:
        result = replace(
            result,
            backtest=run_backtest_engine(
                panel.close(),
                result.labels,
                settings,
                confidence=result.confidence,
                start=refit_dates[0].date(),
            ),
        )

    return result
