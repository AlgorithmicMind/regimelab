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
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pandas as pd

from regimelab.backtest.walkforward import WalkForwardResult, run_walk_forward
from regimelab.data.alignment import align_panel
from regimelab.data.fetcher import PanelFetcher
from regimelab.data.providers import resolve_provider
from regimelab.features import build_feature_matrix, snapshot_at
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


def _fetch_and_align(settings: Settings) -> AlignedPanel:
    """Fetch data through the configured provider and enforce PIT alignment."""
    provider = resolve_provider(settings)
    cache_dir = settings.artifacts_dir / "cache"
    fetcher = PanelFetcher(provider, cache_dir, ttl_hours=0)
    raw = fetcher.fetch(
        settings.data.all_symbols,
        settings.data.start,
        settings.data.end,
    )
    return align_panel(raw, anchor=settings.data.benchmark)


def _fit_single_model(
    features: pd.DataFrame,
    returns: pd.Series,
    settings: Settings,
    train_end: pd.Timestamp,
) -> FittedRegime:
    """Fit the classifier on the full available history."""
    model = build_classifier(settings.model)
    return model.fit(features, returns, train_end=train_end)


def run_pipeline(
    settings: Settings,
    *,
    command: str = "run",
    run_backtest: bool = True,
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

    Returns
    -------
    PipelineResult
        Full result with a writable :class:`~regimelab.store.RunStore`.
    """
    panel = _fetch_and_align(settings)
    features = build_feature_matrix(panel, settings)

    manifest = RunManifest.capture(
        command=command,
        settings=settings,
        input_digest=digest_frame(panel.close()),
    )
    store = RunStore(settings.artifacts_dir, manifest)

    walkforward = run_walk_forward(
        panel,
        features,
        settings,
        run_backtest=run_backtest,
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
    """Build the telemetry payload for the last observable date.

    The payload is the canonical boundary object rendered by the TUI and HTML
    report. It contains the modal regime, the full posterior, the feature
    snapshot and the target allocation at the final as-of date.
    """
    asof = result.walkforward.labels.index[-1]
    features = result.features
    snapshot = snapshot_at(features, asof)
    probabilities = result.walkforward.regime_probabilities(asof)
    regime = probabilities.argmax

    target_weights: AllocationWeights | None = None
    if regime in result.settings.backtest.regime_weights:
        target_weights = AllocationWeights(
            weights=result.settings.backtest.regime_weights[regime],
        )

    last_fold = result.walkforward.folds[-1]
    metadata = last_fold.model

    return TelemetryPayload(
        asof=asof.date(),
        regime=regime,
        probabilities=probabilities,
        features=snapshot,
        model=metadata,
        target_weights=target_weights,
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
) -> TelemetryPayload:
    """Produce a telemetry payload for a specific historical as-of date.

    The model is fit through ``asof`` and the filter is run up to ``asof``,
    then the payload is built. This is the function used by the ``asof`` CLI
    command and by the research notebook examples.
    """
    panel = _fetch_and_align(settings)
    features = build_feature_matrix(panel, settings)
    returns = panel.log_returns([settings.data.benchmark])[settings.data.benchmark]

    fit = _fit_single_model(features, returns, settings, train_end=pd.Timestamp(asof))
    probabilities = fit.probabilities_at(features, asof)
    regime = probabilities.argmax
    snapshot = snapshot_at(features, asof)

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
        regime=regime,
        probabilities=probabilities,
        features=snapshot,
        model=fit.metadata,
        target_weights=target_weights,
    )
