"""Walk-forward evaluation.

Point-in-time walk-forward means the model is refit on a training window ending
at :math:`t_k` and used to classify sessions *after* :math:`t_k` but *before* the
next refit. Crucially, the parameters estimated at :math:`t_k` are **frozen**
while the filter is advanced: the model cannot see any observation used in its
out-of-sample classification.

Each fold model is solely responsible for the interval :math:`(t_k, t_{k+1}]`:
the filter is run over the whole history from the stationary distribution with
the parameters frozen at :math:`t_k` -- causal, and independent of the previous
fold's model -- and only that interval is kept. Dates at or before the first
refit are never labelled, so no session is classified by a model that trained
on it.

The default is an **expanding window**: each refit uses all data from the start
of the panel up to the refit date. A fixed rolling window would discard
recession data during calm periods, which is exactly when the model most needs
to remember what tail risk looks like.

The fold machinery is exposed as :func:`refit_schedule`, :func:`oos_train_end`
and :func:`filter_frozen` so that
:func:`regimelab.pipeline.run_single_asof` can reproduce the walk-forward
signal for a single date without reimplementing any of it.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import pandas as pd

from regimelab.backtest.engine import run_backtest as run_backtest_engine
from regimelab.models.base import InsufficientDataError, probabilities_from_row
from regimelab.models.registry import build_classifier
from regimelab.models.types import MarketRegime, ModelMetadata, RegimeProbabilities

if TYPE_CHECKING:
    from regimelab.backtest.engine import BacktestResult
    from regimelab.data.alignment import AlignedPanel
    from regimelab.models.base import FittedRegime
    from regimelab.settings import Settings

__all__ = [
    "WalkForwardFold",
    "WalkForwardResult",
    "filter_frozen",
    "oos_train_end",
    "refit_schedule",
    "run_walk_forward",
]


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

    def posterior_at(
        self,
        asof: dt.date | pd.Timestamp,
    ) -> tuple[RegimeProbabilities, pd.Timestamp]:
        """Validated posterior at the most recent labelled date at or before ``asof``.

        Also returns the date that produced it, which is earlier than ``asof``
        whenever the newest sessions have no complete feature row.
        """
        truncated = self.probabilities.loc[: pd.Timestamp(asof)]
        if truncated.empty:
            msg = f"no walk-forward probability at or before {asof}"
            raise KeyError(msg)
        return probabilities_from_row(truncated.iloc[-1]), truncated.index[-1]

    def regime_probabilities(self, asof: dt.date | pd.Timestamp) -> RegimeProbabilities:
        """Validated posterior at the most recent date at or before ``asof``."""
        return self.posterior_at(asof)[0]


def refit_schedule(features: pd.DataFrame, settings: Settings) -> list[pd.Timestamp]:
    """Scheduled training-window end dates for each fold.

    The first fold uses the earliest feature row that satisfies
    ``min_train_observations`` and the feature warmup. Subsequent folds
    advance by ``backtest.refit_frequency_days``. The stepped schedule is
    prefix-stable: truncating ``features`` and recomputing yields the same
    dates up to the truncation point.
    """
    full = features.dropna(how="any")
    if len(full) < settings.model.min_train_observations:
        msg = (
            f"feature matrix has only {len(full)} complete rows, "
            f"below min_train_observations {settings.model.min_train_observations}"
        )
        raise ValueError(msg)

    first = settings.model.min_train_observations - 1
    return list(full.index[first :: settings.backtest.refit_frequency_days])


def _refit_dates(features: pd.DataFrame, settings: Settings) -> list[pd.Timestamp]:
    """Scheduled refits plus the final observation as an interval sentinel.

    The appended last date terminates the final fold's interval; it is never a
    valid ``train_end`` -- a model trained through the last row would be
    labelling data it has seen. :func:`oos_train_end` therefore reads only
    :func:`refit_schedule`.
    """
    starts = refit_schedule(features, settings)
    full = features.dropna(how="any")

    # Always include the final observation so the last fold reaches the end
    # of the sample.
    if full.index[-1] not in starts:
        starts.append(full.index[-1])

    return starts


def oos_train_end(
    features: pd.DataFrame,
    settings: Settings,
    date: pd.Timestamp,
) -> pd.Timestamp:
    """The ``train_end`` of the walk-forward fold that labels ``date``.

    The last scheduled refit strictly before ``date``. Truncating the feature
    matrix at ``date`` first keeps the answer causal, and the stepped
    schedule's prefix-stability makes it equal the full-run schedule.

    Raises
    ------
    InsufficientDataError
        If ``date`` is at or before the first refit, so no out-of-sample
        walk-forward model exists for it.
    """
    try:
        schedule = refit_schedule(features.loc[:date], settings)
    except ValueError:
        schedule = []

    eligible = [d for d in schedule if d < date]
    if eligible:
        return eligible[-1]

    full = features.dropna(how="any")
    first_refit = (
        full.index[settings.model.min_train_observations - 1]
        if len(full) >= settings.model.min_train_observations
        else None
    )
    detail = f" (first scheduled refit: {first_refit.date()})" if first_refit is not None else ""
    msg = f"no out-of-sample walk-forward model exists for {date.date()}{detail}"
    raise InsufficientDataError(msg)


def filter_frozen(
    fit: FittedRegime,
    features: pd.DataFrame,
    end: dt.date | pd.Timestamp,
) -> pd.DataFrame:
    """Filter the whole history from the stationary distribution.

    The parameters frozen into ``fit`` are applied causally over
    ``features`` up to ``end``. The filter starts at the first *all-column*
    complete row rather than ``features.index[0]``: a row whose model features
    are complete while a diagnostic column is NaN would otherwise change the
    filtered path.
    """
    full = features.dropna(how="any")
    if full.empty:
        msg = "feature matrix has no complete rows"
        raise InsufficientDataError(msg)
    return fit.probabilities(features.loc[full.index[0] : end], initial=None)


def run_walk_forward(
    panel: AlignedPanel,
    features: pd.DataFrame,
    settings: Settings,
    *,
    run_backtest: bool = True,
    progress_callback: Callable[[int, int, str], None] | None = None,
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
    progress_callback
        Optional callback invoked at the start of each test fold with
        ``(current_fold, total_folds, test_date)`` where ``test_date`` is an
        ISO-formatted string for the fold's test period end.

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

    # Fold k labels the interval (r_k, r_{k+1}]; the last refit point would
    # cover (r_last, last feature date], which is empty because _refit_dates
    # always appends the final observation. Empty intervals are skipped before
    # fitting so no model is trained for a fold that produces no labels.
    intervals: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    for fold_index, train_end in enumerate(refit_dates):
        is_last = fold_index + 1 == len(refit_dates)
        test_end = full.index[-1] if is_last else refit_dates[fold_index + 1]
        if full.loc[train_end:test_end].shape[0] <= 1:
            continue
        intervals.append((train_end, test_end))

    # Training always starts from the earliest complete feature row.
    train_start = full.index[0]

    total_folds = len(intervals)
    for fold_index, (train_end, test_end) in enumerate(intervals):
        if progress_callback is not None:
            progress_callback(fold_index + 1, total_folds, test_end.date().isoformat())

        fit = model.fit(features, reference_returns, train_end=train_end)

        # Filter the whole history from the stationary distribution with the
        # parameters frozen at train_end, then keep only this fold's interval
        # (train_end, test_end].
        hist = filter_frozen(fit, features, test_end)
        probs = hist.loc[hist.index > train_end]

        all_probabilities.append(probs)
        all_folds.append(
            WalkForwardFold(
                fold_index=fold_index,
                train_start=train_start.date(),
                train_end=train_end.date(),
                test_start=probs.index[0].date(),
                test_end=test_end.date(),
                n_train=features.loc[train_start:train_end].dropna(how="any").shape[0],
                n_test=probs.shape[0],
                model=fit.metadata,
                assignment=fit.assignment.describe(),
            ),
        )

    if not all_probabilities:
        msg = "walk-forward produced no probability path"
        raise ValueError(msg)

    # Fold intervals are disjoint by construction, so concatenation is exact.
    probabilities = pd.concat(all_probabilities, sort=False).sort_index()
    duplicated = probabilities.index[probabilities.index.duplicated()]
    if len(duplicated):
        msg = (
            f"walk-forward fold intervals overlap on {len(duplicated)} dates, "
            f"e.g. {[d.date().isoformat() for d in duplicated[:5]]}"
        )
        raise ValueError(msg)

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
