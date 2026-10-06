"""STRESS labelling against the synthetic provider's known latent path.

The synthetic generator simulates a three-state regime path whose stress state
(state 2) is observable via ``SyntheticProvider.latent_states``. That ground
truth lets these tests verify that the classifier recovers the stress regime,
not merely that it emits a well-formed mapping.
"""

from __future__ import annotations

import pandas as pd
import pytest

from regimelab.data.alignment import align_panel
from regimelab.data.fetcher import PanelFetcher
from regimelab.data.providers import SyntheticProvider, resolve_provider
from regimelab.features import build_feature_matrix
from regimelab.models.registry import build_classifier
from regimelab.models.types import regime_ladder
from regimelab.settings import ModelSettings, Settings


@pytest.fixture(scope="module")
def synthetic_sample(test_settings: Settings) -> tuple[pd.DataFrame, pd.Series]:
    """Synthetic feature matrix and benchmark returns, built once per module."""
    provider = resolve_provider(test_settings)
    fetcher = PanelFetcher(provider, test_settings.artifacts_dir / "cache")
    raw = fetcher.fetch(
        test_settings.data.all_symbols,
        test_settings.data.start,
        test_settings.data.end,
    )
    panel = align_panel(raw, anchor=test_settings.data.benchmark)
    features = build_feature_matrix(panel, test_settings)
    returns = panel.log_returns([test_settings.data.benchmark])[test_settings.data.benchmark]
    return features, returns


def test_stress_labels_recover_latent_stress_state(
    test_settings: Settings,
    synthetic_sample: tuple[pd.DataFrame, pd.Series],
) -> None:
    """The most risk-off label must overlap the synthetic stress state.

    Both directions are asserted: most of the dates labelled most-risk-off
    should have latent state 2, and most latent-state-2 dates should carry the
    most risk-off label.
    """
    features, returns = synthetic_sample
    fit = build_classifier(test_settings.model).fit(features, returns, train_end=None)
    labels = fit.labels(features)

    provider = SyntheticProvider(seed=test_settings.model.seed)
    latent = provider.latent_states(test_settings.data.start, test_settings.data.end)

    common = labels.index.intersection(latent.index)
    aligned_labels = labels.loc[common]
    aligned_latent = latent.loc[common]

    most_risk_off = regime_ladder(test_settings.model.n_states)[-1]
    labelled_stress = aligned_labels == most_risk_off
    latent_stress = aligned_latent == 2

    share_latent_in_stress = float((aligned_latent[labelled_stress] == 2).mean())
    share_stress_in_latent = float((aligned_labels[latent_stress] == most_risk_off).mean())

    assert share_latent_in_stress > 0.5
    assert share_stress_in_latent > 0.5


@pytest.mark.parametrize("n_states", [2, 3, 4])
def test_stress_labelling_is_bijective_and_every_label_emitted(
    test_settings: Settings,
    synthetic_sample: tuple[pd.DataFrame, pd.Series],
    n_states: int,
) -> None:
    """The mapping covers the full ladder and every regime is emitted."""
    features, returns = synthetic_sample
    model_settings = ModelSettings.model_validate(
        {**test_settings.model.model_dump(), "n_states": n_states}
    )
    fit = build_classifier(model_settings).fit(features, returns, train_end=None)

    ladder = regime_ladder(n_states)
    assert set(fit.assignment.mapping.values()) == set(ladder)

    labels = fit.labels(features)
    assert labels.nunique() == n_states
