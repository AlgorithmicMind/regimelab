"""Tests for FittedRegime persistence and causal inference reuse."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from regimelab.models.base import FittedRegime
from regimelab.models.registry import build_classifier
from regimelab.pipeline import PipelineResult, run_single_asof


def test_fitted_regime_bundle_roundtrip(pipeline_result: PipelineResult) -> None:
    """to_bundle -> from_bundle must reproduce identical probabilities."""
    fit = _fit_model(pipeline_result)
    features = _usable_features(pipeline_result)

    expected = fit.probabilities(features).to_numpy()

    bundle = fit.to_bundle()
    restored = FittedRegime.from_bundle(bundle)
    actual = restored.probabilities(features).to_numpy()

    assert restored.classifier == fit.classifier
    assert restored.assignment.method == fit.assignment.method
    assert np.allclose(actual, expected)


def test_fitted_regime_save_load_roundtrip(pipeline_result: PipelineResult, tmp_path: Path) -> None:
    """save -> load must reproduce identical probabilities."""
    fit = _fit_model(pipeline_result)
    features = _usable_features(pipeline_result)

    expected = fit.probabilities(features).to_numpy()

    save_dir = tmp_path / "regime_bundle"
    fit.save(save_dir)
    loaded = FittedRegime.load(save_dir)
    actual = loaded.probabilities(features).to_numpy()

    assert loaded.classifier == fit.classifier
    assert loaded.metadata.n_states == fit.metadata.n_states
    assert np.allclose(actual, expected)


def test_run_single_asof_uses_fitted_model_without_refit(pipeline_result: PipelineResult) -> None:
    """run_single_asof(fitted=...) must reuse a persisted model."""
    settings = pipeline_result.settings
    fit = _fit_model(pipeline_result)
    asof = fit.metadata.train_end

    # Reproduce the same payload with and without the pre-trained model. The
    # daily policy fits through ``asof``, which is what ``_fit_model`` did, so
    # the comparison stays model-for-model.
    cold = run_single_asof(settings, asof, refit_policy="daily")
    with_fitted = run_single_asof(settings, asof, fitted=fit)

    assert cold.regime == with_fitted.regime
    assert np.allclose(
        list(cold.probabilities.probabilities.values()),
        list(with_fitted.probabilities.probabilities.values()),
    )


def _fit_model(result: PipelineResult) -> FittedRegime:
    """Fit the configured classifier on the full usable feature history."""
    settings = result.settings
    features = _usable_features(result)
    returns = result.panel.log_returns([settings.data.benchmark])[settings.data.benchmark]
    model = build_classifier(settings.model)
    return model.fit(features, returns, train_end=features.index[-1])


def _usable_features(result: PipelineResult) -> pd.DataFrame:
    """Return the feature matrix used by the classifier (the columns it was fit on)."""
    features = result.features
    model_features = list(result.settings.model.features)
    return features.loc[:, model_features].dropna(how="any")
