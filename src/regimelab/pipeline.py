"""End-to-end orchestration.

The pipeline is the only place where the data, feature, model, backtest and
reporting layers are allowed to touch. It is deliberately thin: each stage
returns plain Pydantic or pandas objects, and the pipeline only wires them.

Causality is guaranteed by construction:

1. The data layer aligns every symbol to the master calendar.
2. Feature windows are trailing, so a row at ``t`` only uses data up to ``t``.
3. The walk-forward loop refits at ``t_k`` on history ending at ``t_k``.
4. The backtest simulates trades on day ``t`` using the regime available at the
   close of day ``t - signal_lag_days``.

The telemetry payload is the boundary object: anything the UI or report needs
must fit inside a :class:`~regimelab.models.types.TelemetryPayload`.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd

from regimelab.backtest.walkforward import (
    WalkForwardResult,
    filter_frozen,
    oos_train_end,
    run_walk_forward,
)
from regimelab.data.fetcher import load_aligned_panel
from regimelab.features import build_feature_matrix, snapshot_at
from regimelab.models.base import InsufficientDataError, probabilities_from_row
from regimelab.models.registry import build_classifier
from regimelab.models.types import AllocationWeights, TelemetryPayload
from regimelab.store import RunManifest, RunStore, digest_frame

if TYPE_CHECKING:
    from regimelab.backtest.engine import BacktestResult
    from regimelab.data.alignment import AlignedPanel
    from regimelab.models.base import FittedRegime
    from regimelab.settings import Settings

__all__ = ["PipelineResult", "RunCommand", "latest_telemetry", "run_pipeline"]


@dataclass(frozen=True, slots=True)
class PipelineResult:
    """Output of a complete pipeline run.

    Attributes
    ----------
    settings
        Resolved configuration that produced the run.
    panel
        Aligned, point-in-time price panel.
    features
        Causal feature matrix.
    walkforward
        Walk-forward regime path and backtest.
    manifest
        Run provenance, including git state and input digest.
    store
        Filesystem location for artefacts.
    """

    settings: Settings
    panel: AlignedPanel
    features: pd.DataFrame
    walkforward: WalkForwardResult
    manifest: RunManifest
    store: RunStore

    @property
    def backtest(self) -> BacktestResult | None:
        """Convenience accessor for the backtest result."""
        return self.walkforward.backtest


def _fetch_and_align(settings: Settings, *, force_refresh: bool = False) -> AlignedPanel:
    """Fetch data through the configured provider and enforce PIT alignment.

    Every pipeline entry point goes through
    :func:`~regimelab.data.fetcher.load_aligned_panel`: alignment anchors on
    ``data.calendar_anchor``, the panel cache lives in ``data.cache_dir`` and
    is reused within ``data.cache_ttl_hours``. ``force_refresh`` bypasses the
    cache and always refetches.
    """
    return load_aligned_panel(settings, force_refresh=force_refresh)


def _fit_single_model(
    features: pd.DataFrame,
    returns: pd.Series,
    settings: Settings,
    train_end: pd.Timestamp,
) -> FittedRegime:
    """Fit the classifier on the full available history."""
    model = build_classifier(settings.model)
    return model.fit(features, returns, train_end=train_end)


def _data_quality_warnings(
    *,
    asof: dt.date,
    session_date: dt.date,
    observation_date: dt.date,
    features: pd.DataFrame,
    feature_names: Sequence[str],
    panel: AlignedPanel,
) -> tuple[str, ...]:
    """Describe, never repair, the gap between the request and the data.

    Degradations are reported in a fixed order: a requested date absent from
    the supplied panel, a posterior computed on an older supplied session, model
    features unavailable at the as-of date, inputs carried forward by the
    staleness budget, and symbols dropped for having no usable history at all.
    """
    warnings: list[str] = []

    if session_date < asof:
        message = (
            f"asof_not_in_panel: requested asof {asof.isoformat()} is not present "
            f"in the provided panel; last available session {session_date.isoformat()}; "
            "verify an exchange calendar to distinguish market closure from unavailable data"
        )
        warnings.append(message)

    if observation_date < session_date:
        message = (
            f"stale_observation: regime computed on {observation_date.isoformat()}, "
            f"last session {session_date.isoformat()}"
        )
        warnings.append(message)

    session = features.loc[: pd.Timestamp(asof)]
    row = session.iloc[-1] if not session.empty else None
    warnings.extend(
        f"missing_feature: {name} unavailable at asof"
        for name in feature_names
        if name not in features.columns or row is None or pd.isna(row[name])
    )

    staleness = panel.staleness.loc[: pd.Timestamp(asof)]
    if not staleness.empty:
        latest = staleness.iloc[-1]
        for symbol in sorted(str(s) for s in latest.index):
            value = latest[symbol]
            if np.isfinite(value) and value > 0:
                warnings.append(
                    f"stale_input: {symbol} last observed {int(value)} sessions earlier"
                )

    warnings.extend(f"dropped_symbol: {symbol}" for symbol in panel.dropped_symbols)
    return tuple(warnings)


def run_pipeline(
    settings: Settings,
    *,
    command: str = "run",
    run_backtest: bool = True,
    progress_callback: Callable[[int, int, str], None] | None = None,
    status_callback: Callable[[str], None] | None = None,
    force_refresh: bool = False,
) -> PipelineResult:
    """Run the full point-in-time pipeline.

    Parameters
    ----------
    settings
        Resolved configuration.
    command
        CLI command name to record in the manifest.
    run_backtest
        Whether to run the walk-forward backtest.
    progress_callback
        Optional callback forwarded to :func:`~regimelab.backtest.walkforward.run_walk_forward`
        with ``(current_fold, total_folds, test_date)``.
    status_callback
        Optional callback invoked with human-readable phase descriptions.
    force_refresh
        Bypass the panel cache and refetch, ignoring ``cache_ttl_hours``.

    Returns
    -------
    PipelineResult
        Full result with a writable :class:`~regimelab.store.RunStore`.
    """
    if status_callback is not None:
        status_callback("Fetching & aligning PIT market data...")
    panel = _fetch_and_align(settings, force_refresh=force_refresh)

    if status_callback is not None:
        status_callback("Computing systemic risk feature matrix...")
    features = build_feature_matrix(panel, settings)

    manifest = RunManifest.capture(
        command=command,
        settings=settings,
        input_digest=digest_frame(panel.close()),
    )
    store = RunStore(settings.artifacts_dir, manifest)

    if status_callback is not None:
        status_callback("Running walk-forward regime estimation...")
    walkforward = run_walk_forward(
        panel,
        features,
        settings,
        run_backtest=run_backtest,
        progress_callback=progress_callback,
    )

    store.write_manifest()
    store.write_frame(features, "features")
    store.write_frame(walkforward.probabilities, "regime_probabilities")
    store.write_frame(walkforward.labels.to_frame("regime"), "regime_labels")
    if walkforward.backtest is not None:
        store.write_frame(walkforward.backtest.equity.to_frame("equity"), "equity")

    return PipelineResult(
        settings=settings,
        panel=panel,
        features=features,
        walkforward=walkforward,
        manifest=manifest,
        store=store,
    )


def latest_telemetry(result: PipelineResult) -> TelemetryPayload:
    """Build the telemetry payload for the last master-calendar session.

    The payload is the canonical boundary object rendered by the TUI and HTML
    report. It contains the modal regime, the full posterior, the feature
    snapshot and the target allocation at the final as-of date.

    When the newest sessions are not filterable -- inputs past their staleness
    budget, say -- the payload reports the gap in ``session_date``,
    ``observation_date`` and ``warnings`` rather than filling it. Whether a
    stale read is publishable is the consumer's decision, not the engine's.
    """
    # ``asof`` is already the last master-calendar session, so it is also the
    # session the payload reports against.
    asof = result.features.index[-1]
    features = result.features
    snapshot = snapshot_at(features, asof)
    probabilities, observed = result.walkforward.posterior_at(asof)
    regime = probabilities.argmax

    target_weights: AllocationWeights | None = None
    if regime in result.settings.backtest.regime_weights:
        target_weights = AllocationWeights(
            weights=result.settings.backtest.regime_weights[regime],
        )

    # The payload is signed by the model of the fold that labelled
    # ``observed`` -- the interval is inclusive on both ends, matching the
    # walk-forward convention that fold k owns (r_k, r_{k+1}] plus its
    # first labelled session.
    observation_date = observed.date()
    fold = next(
        (
            candidate
            for candidate in result.walkforward.folds
            if candidate.test_start <= observation_date <= candidate.test_end
        ),
        None,
    )
    if fold is None:
        msg = f"no walk-forward fold covers observation_date {observation_date.isoformat()}"
        raise ValueError(msg)
    metadata = fold.model

    return TelemetryPayload(
        asof=asof.date(),
        session_date=asof.date(),
        observation_date=observation_date,
        regime=regime,
        probabilities=probabilities,
        features=snapshot,
        model=metadata,
        state_statistics=fold.statistics,
        target_weights=target_weights,
        warnings=_data_quality_warnings(
            asof=asof.date(),
            session_date=asof.date(),
            observation_date=observed.date(),
            features=features,
            feature_names=metadata.feature_names or result.settings.model.features,
            panel=result.panel,
        ),
    )


class RunCommand:
    """Thin callable wrapper so the CLI and test suite invoke the same entry point.

    Also the hook point for any future instrumentation (timing, structured
    logging, metric emission) that must wrap every user-facing run.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def __call__(self, *, command: str = "run") -> PipelineResult:
        return run_pipeline(self._settings, command=command)

    @property
    def settings(self) -> Settings:
        return self._settings


def run_single_asof(
    settings: Settings,
    asof: dt.date,
    *,
    command: str = "asof",
    fitted: FittedRegime | None = None,
    refit_policy: Literal["walkforward", "daily"] = "walkforward",
    force_refresh: bool = False,
) -> TelemetryPayload:
    """Produce a telemetry payload for a specific historical as-of date.

    With the default ``refit_policy="walkforward"`` the payload reproduces the
    walk-forward signal exactly: the model that owns ``asof``'s last usable
    observation is the one trained at the last scheduled refit *strictly
    before* it, and the filter runs with frozen parameters from the stationary
    distribution. ``refit_policy="daily"`` keeps the previous behaviour of
    fitting through ``asof`` itself. This is the function used by the ``asof``
    CLI command and by the research notebook examples.

    When the sessions closest to ``asof`` are not filterable -- or ``asof``
    itself is not a trading session -- the payload reports the gap in
    ``session_date``, ``observation_date`` and ``warnings`` rather than
    filling it; publishing a stale read is the consumer's decision.

    If ``fitted`` is provided, the model is *not* retrained and the refit
    policy does not apply.  Instead the supplied
    :class:`~regimelab.models.base.FittedRegime` is filtered causally through
    ``asof`` using its frozen parameters and standardiser.

    ``force_refresh`` bypasses the panel cache and refetches, ignoring
    ``cache_ttl_hours``.
    """
    panel = _fetch_and_align(settings, force_refresh=force_refresh)
    features = build_feature_matrix(panel, settings)
    returns = panel.log_returns([settings.data.benchmark])[settings.data.benchmark]

    if fitted is not None:
        fit = fitted
        probabilities, observed = fit.posterior_at(features, asof)
    elif refit_policy == "daily":
        fit = _fit_single_model(features, returns, settings, train_end=pd.Timestamp(asof))
        probabilities, observed = fit.posterior_at(features, asof)
    else:
        # The observation date anchors the fold lookup: when asof's own row
        # is incomplete, a model trained "strictly before asof" could be
        # trained on `observed` itself -- the walk-forward fold that labels
        # `observed` is the one trained strictly before `observed`.
        usable = features.loc[: pd.Timestamp(asof), list(settings.model.features)].dropna(how="any")
        if usable.empty:
            msg = f"no complete feature row at or before {asof}"
            raise InsufficientDataError(msg)
        observed = usable.index[-1]

        train_end = oos_train_end(features.loc[: pd.Timestamp(asof)], settings, observed)
        fit = build_classifier(settings.model).fit(features, returns, train_end=train_end)

        hist = filter_frozen(fit, features, pd.Timestamp(asof))
        if hist.empty:
            msg = f"no filterable observation at or before {asof}"
            raise InsufficientDataError(msg)
        probabilities = probabilities_from_row(hist.iloc[-1])
        if hist.index[-1] != observed:
            msg = (
                f"walk-forward filter observed {hist.index[-1].date()} for asof "
                f"{asof}, expected {observed.date()}"
            )
            raise RuntimeError(msg)

    regime = probabilities.argmax
    snapshot = snapshot_at(features, asof)
    # ``features`` is indexed on the master calendar; the last session at or
    # before ``asof`` exists because the posterior and snapshot already raised
    # otherwise.
    session_date = features.index[features.index <= pd.Timestamp(asof)][-1].date()

    target_weights: AllocationWeights | None = None
    if regime in settings.backtest.regime_weights:
        target_weights = AllocationWeights(weights=settings.backtest.regime_weights[regime])

    manifest = RunManifest.capture(
        command=command,
        settings=settings,
        asof=asof,
        input_digest=digest_frame(panel.close()),
    )
    store = RunStore(settings.artifacts_dir, manifest)
    store.write_manifest()
    store.write_json(manifest, "manifest")

    return TelemetryPayload(
        asof=asof,
        session_date=session_date,
        observation_date=observed.date(),
        regime=regime,
        probabilities=probabilities,
        features=snapshot,
        model=fit.metadata,
        state_statistics=fit.assignment.to_statistics(),
        target_weights=target_weights,
        warnings=_data_quality_warnings(
            asof=asof,
            session_date=session_date,
            observation_date=observed.date(),
            features=features,
            feature_names=fit.standardiser.columns,
            panel=panel,
        ),
    )
