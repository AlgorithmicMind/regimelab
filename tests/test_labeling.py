"""Regression tests for canonical state labelling."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from regimelab.data.alignment import align_panel
from regimelab.data.fetcher import PanelFetcher
from regimelab.data.providers import resolve_provider
from regimelab.features import build_feature_matrix
from regimelab.models.labeling import (
    STRESS_SIGNS,
    RankingMethod,
    StateStatistics,
    assign_labels,
    relabel_probabilities,
    stress_scores,
)
from regimelab.models.registry import CLASSIFIERS, build_classifier
from regimelab.models.types import (
    DEFAULT_RANKING_METHODS,
    MarketRegime,
    regime_ladder,
)
from regimelab.models.types import STRESS_SIGNS as TYPES_STRESS_SIGNS
from regimelab.settings import ModelSettings, Settings


def _make_stats(state: int, mean: float, vol: float, occupancy: float = 0.25) -> StateStatistics:
    return StateStatistics(
        state=state,
        occupancy=occupancy,
        mean_return=mean,
        volatility=vol,
        n_observations=100,
    )


def test_positive_drift_never_high_vol_bear() -> None:
    """A positive-drift cluster with moderate Sharpe must not be HIGH_VOL_BEAR.

    The pathological case: 4-state fit where the third state by Sharpe has
    +10.38% annual return and Sharpe 0.53.  Without the guardrail it receives
    HIGH_VOL_BEAR only because it is the 3rd ranked state.  With the guardrail
    it must be capped at NEUTRAL_TRANSITION (or higher if the non-bear slot is
    already taken by a better positive state).
    """
    # State 0: strongest Sharpe, positive drift.
    s0 = _make_stats(0, mean=0.15, vol=0.20)
    # State 1: second Sharpe, positive drift.
    s1 = _make_stats(1, mean=0.12, vol=0.18)
    # State 2: moderate Sharpe, positive drift -> the bug case.
    s2 = _make_stats(2, mean=0.1038, vol=0.1958)
    # State 3: negative drift, lowest Sharpe.
    s3 = _make_stats(3, mean=-0.05, vol=0.25)

    assignment = assign_labels((s0, s1, s2, s3), method=RankingMethod.SHARPE)

    assert assignment.mapping[2] != MarketRegime.HIGH_VOL_BEAR
    assert assignment.mapping[2] != MarketRegime.RISK_OFF
    assert assignment.mapping[3] in (MarketRegime.HIGH_VOL_BEAR, MarketRegime.RISK_OFF)


def test_positive_drift_capped_at_neutral_for_four_states() -> None:
    """Four positive-drift states must be capped to non-bear labels only."""
    stats = (
        _make_stats(0, mean=0.20, vol=0.15),
        _make_stats(1, mean=0.10, vol=0.15),
        _make_stats(2, mean=0.05, vol=0.15),
        _make_stats(3, mean=0.02, vol=0.15),
    )

    assignment = assign_labels(stats, method=RankingMethod.SHARPE)

    for label in assignment.mapping.values():
        assert label in (MarketRegime.BULL_TREND, MarketRegime.NEUTRAL_TRANSITION)


def test_negative_drift_can_be_bearish() -> None:
    """Negative-drift states must still be allowed to receive bearish labels."""
    stats = (
        _make_stats(0, mean=0.15, vol=0.20),
        _make_stats(1, mean=0.05, vol=0.18),
        _make_stats(2, mean=-0.10, vol=0.25),
        _make_stats(3, mean=-0.20, vol=0.30),
    )

    assignment = assign_labels(stats, method=RankingMethod.SHARPE)

    bearish = (MarketRegime.HIGH_VOL_BEAR, MarketRegime.RISK_OFF)
    assert any(label in bearish for label in assignment.mapping.values())


def test_relabel_probabilities_aggregates_duplicate_regimes() -> None:
    """When two raw states map to the same regime, probabilities must sum."""
    stats = (
        _make_stats(0, mean=0.20, vol=0.15),
        _make_stats(1, mean=0.10, vol=0.15),
        _make_stats(2, mean=0.05, vol=0.15),
        _make_stats(3, mean=0.02, vol=0.15),
    )

    assignment = assign_labels(stats, method=RankingMethod.SHARPE)

    # Build a fake posterior frame with one row and integer state columns.
    posteriors = pd.DataFrame(
        {
            0: [0.10],
            1: [0.20],
            2: [0.30],
            3: [0.40],
        },
        index=pd.DatetimeIndex(["2024-01-01"], name="date"),
    )

    relabeled = relabel_probabilities(posteriors, assignment)

    ladder = regime_ladder(4)
    assert list(relabeled.columns) == [regime.value for regime in ladder]
    # All probabilities must still sum to one after aggregation.
    assert np.allclose(relabeled.to_numpy().sum(axis=1), [1.0])


def test_stress_ranking_orders_by_signed_emission_means() -> None:
    """STRESS maps states onto the ladder by signed stress features only.

    The raw state indices are deliberately not in stress order, and the
    unsigned ``mean_correlation`` column carries extreme values that would
    dominate the ordering if it were not ignored.
    """
    feature_names = ("trailing_return", "realised_vol", "vix_term_spread", "mean_correlation")
    emission_means = np.array(
        [
            [-1.0, 1.5, -1.2, -9.0],  # state 0: crisis profile
            [0.5, -0.8, 0.7, 9.0],  # state 1: calm profile
            [0.0, 0.2, 0.0, 0.0],  # state 2: transition profile
        ]
    )
    stats = tuple(_make_stats(state, mean=0.05, vol=0.20) for state in range(3))

    assignment = assign_labels(
        stats,
        method=RankingMethod.STRESS,
        emission_means=emission_means,
        feature_names=feature_names,
    )

    ladder = regime_ladder(3)
    assert assignment.mapping == {1: ladder[0], 2: ladder[1], 0: ladder[2]}
    expected = [
        (-0.5 - 0.8 - 0.7) / 3,  # state 1
        (0.0 + 0.2 + 0.0) / 3,  # state 2
        (1.0 + 1.5 + 1.2) / 3,  # state 0
    ]
    assert [stat.state for stat in assignment.statistics] == [1, 2, 0]
    for stat, want in zip(assignment.statistics, expected, strict=True):
        assert stat.stress_score == pytest.approx(want)


def test_stress_ranking_is_bijective_with_all_positive_drift() -> None:
    """Under STRESS every ladder label is emitted even with positive drift.

    With SHARPE these four positive-drift states would all be capped onto
    non-bear labels; STRESS has no such guardrail, so the mapping stays a
    bijection onto the ladder and follows the stress ordering.
    """
    feature_names = ("trailing_return", "realised_vol")
    emission_means = np.array(
        [
            [0.8, -0.5],
            [0.2, 0.1],
            [-0.3, 0.6],
            [-1.2, 1.4],
        ]
    )
    stats = tuple(_make_stats(state, mean=0.10, vol=0.15) for state in range(4))

    assignment = assign_labels(
        stats,
        method=RankingMethod.STRESS,
        emission_means=emission_means,
        feature_names=feature_names,
    )

    ladder = regime_ladder(4)
    assert set(assignment.mapping.values()) == set(ladder)
    assert [assignment.mapping[state] for state in range(4)] == list(ladder)


def test_stress_ignores_positive_drift_guardrail() -> None:
    """A high-stress state with positive drift is RISK_OFF under STRESS only.

    Under SHARPE the same state ranks last by score and lands in the RISK_OFF
    slot, but its positive mean return triggers the guardrail cap. Under STRESS
    there is no cap, so the stress signature labels it RISK_OFF.
    """
    feature_names = ("trailing_return", "realised_vol", "vix_term_spread")
    emission_means = np.array(
        [
            [0.8, -0.6, 0.8],  # state 0: calm
            [-0.2, 1.5, -1.0],  # state 1: stressed
        ]
    )
    stats = (
        _make_stats(0, mean=0.20, vol=0.10),
        _make_stats(1, mean=0.15, vol=0.45),
    )

    stress = assign_labels(
        stats,
        method=RankingMethod.STRESS,
        emission_means=emission_means,
        feature_names=feature_names,
    )
    sharpe = assign_labels(stats, method=RankingMethod.SHARPE)

    assert stress.mapping[1] is MarketRegime.RISK_OFF
    assert sharpe.mapping[1] is not MarketRegime.RISK_OFF


def test_stress_tie_breaks_volatility_then_index() -> None:
    """Equal stress scores resolve by volatility, then by raw state index."""
    feature_names = ("trailing_return", "realised_vol")
    emission_means = np.array([[0.5, 0.5]] * 3)
    stats = (
        _make_stats(0, mean=0.10, vol=0.30),
        _make_stats(1, mean=0.10, vol=0.20),
        _make_stats(2, mean=0.10, vol=0.20),
    )

    assignment = assign_labels(
        stats,
        method=RankingMethod.STRESS,
        emission_means=emission_means,
        feature_names=feature_names,
    )

    assert [stat.state for stat in assignment.statistics] == [1, 2, 0]


def test_stress_requires_emission_inputs() -> None:
    """STRESS raises when its emission inputs are missing or inconsistent."""
    stats = (
        _make_stats(0, mean=0.10, vol=0.20),
        _make_stats(1, mean=0.10, vol=0.20),
    )
    feature_names = ("trailing_return", "realised_vol")

    with pytest.raises(ValueError, match="emission_means"):
        assign_labels(stats, method=RankingMethod.STRESS)

    with pytest.raises(ValueError, match="emission_means"):
        assign_labels(
            stats,
            method=RankingMethod.STRESS,
            emission_means=np.zeros((2, 2)),
        )

    with pytest.raises(ValueError, match="does not match"):
        assign_labels(
            stats,
            method=RankingMethod.STRESS,
            emission_means=np.zeros((2, 3)),
            feature_names=feature_names,
        )

    with pytest.raises(ValueError, match="no signed stress features"):
        assign_labels(
            stats,
            method=RankingMethod.STRESS,
            emission_means=np.zeros((2, 1)),
            feature_names=("mean_correlation",),
        )


def test_stress_signs_custom_mapping_changes_order() -> None:
    """A custom ``signs`` mapping reorders states and can invert the ladder."""
    feature_names = ("trailing_return", "realised_vol", "vix_term_spread")
    emission_means = np.array(
        [
            [0.5, -0.8, 0.7],  # state 0: calm profile
            [-1.0, 1.5, -1.2],  # state 1: crisis profile
            [0.0, 0.2, 0.0],  # state 2: transition profile
        ]
    )
    stats = tuple(_make_stats(state, mean=0.05, vol=0.20) for state in range(3))

    default = assign_labels(
        stats,
        method=RankingMethod.STRESS,
        emission_means=emission_means,
        feature_names=feature_names,
    )
    custom = assign_labels(
        stats,
        method=RankingMethod.STRESS,
        emission_means=emission_means,
        feature_names=feature_names,
        signs={"realised_vol": -1},
    )

    ladder = regime_ladder(3)
    # Default signs: state 1 the most stressed, state 0 the calmest.
    assert default.mapping == {0: ladder[0], 2: ladder[1], 1: ladder[2]}
    # Only realised_vol, signed -1: high realised vol now reads as calm.
    assert custom.mapping == {1: ladder[0], 2: ladder[1], 0: ladder[2]}


def test_stress_signs_empty_signs_errors() -> None:
    """A ``signs`` mapping covering no feature fails with its keys listed."""
    stats = (
        _make_stats(0, mean=0.10, vol=0.20),
        _make_stats(1, mean=0.10, vol=0.20),
    )
    with pytest.raises(ValueError, match="foo_sign"):
        assign_labels(
            stats,
            method=RankingMethod.STRESS,
            emission_means=np.zeros((2, 2)),
            feature_names=("trailing_return", "realised_vol"),
            signs={"foo_sign": 1},
        )


def test_describe_reports_stress_score() -> None:
    """``describe()`` exposes the attached stress score per state."""
    feature_names = ("trailing_return", "realised_vol")
    emission_means = np.array([[0.5, -0.5], [-0.5, 0.5]])
    stats = (
        _make_stats(0, mean=0.10, vol=0.20),
        _make_stats(1, mean=0.10, vol=0.20),
    )

    assignment = assign_labels(
        stats,
        method=RankingMethod.STRESS,
        emission_means=emission_means,
        feature_names=feature_names,
    )
    frame = assignment.describe()

    assert "stress_score" in frame.columns
    assert frame["stress_score"].notna().all()


def test_to_statistics_is_ladder_ordered() -> None:
    """``to_statistics()`` returns one entry per state, sorted by risk_rank."""
    feature_names = ("trailing_return", "realised_vol", "vix_term_spread")
    emission_means = np.array(
        [
            [0.5, -0.8, 0.7],  # state 0: calm profile
            [-1.0, 1.5, -1.2],  # state 1: crisis profile
            [0.0, 0.2, 0.0],  # state 2: transition profile
        ]
    )
    stats = (
        _make_stats(0, mean=0.10, vol=0.10, occupancy=0.5),
        _make_stats(1, mean=-0.20, vol=0.40, occupancy=0.2),
        _make_stats(2, mean=0.02, vol=0.15, occupancy=0.3),
    )

    assignment = assign_labels(
        stats,
        method=RankingMethod.STRESS,
        emission_means=emission_means,
        feature_names=feature_names,
    )
    entries = assignment.to_statistics()

    # Mapping is bijective, so every ladder regime appears exactly once, in
    # ladder order -- not in raw-state order (calm=0, transition=2, crisis=1).
    assert [entry.regime for entry in entries] == list(regime_ladder(3))
    assert [entry.raw_state for entry in entries] == [0, 2, 1]

    crisis = entries[-1]
    assert crisis.regime == MarketRegime.RISK_OFF
    assert crisis.occupancy == pytest.approx(0.2)
    assert crisis.ann_return == pytest.approx(-0.20)
    assert crisis.ann_volatility == pytest.approx(0.40)
    assert crisis.n_observations == 100
    assert crisis.stress_score is not None

    # describe() is built from to_statistics(): same rows, documented columns.
    frame = assignment.describe()
    assert list(frame.columns) == [
        "regime",
        "raw_state",
        "occupancy",
        "ann_return",
        "ann_volatility",
        "sharpe",
        "stress_score",
        "observations",
    ]
    assert list(frame["regime"]) == [r.value for r in regime_ladder(3)]
    assert list(frame["raw_state"]) == [0, 2, 1]


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


def test_ranking_method_setting_overrides_classifier_default(
    test_settings: Settings,
    synthetic_sample: tuple[pd.DataFrame, pd.Series],
) -> None:
    """``model.ranking_method`` is honoured over the classifier's default."""
    features, returns = synthetic_sample

    default_fit = build_classifier(test_settings.model).fit(features, returns, train_end=None)
    assert default_fit.assignment.method is RankingMethod.STRESS

    sharpe_model = ModelSettings.model_validate(
        {**test_settings.model.model_dump(), "ranking_method": "sharpe"}
    )
    sharpe_fit = build_classifier(sharpe_model).fit(features, returns, train_end=None)
    assert sharpe_fit.assignment.method is RankingMethod.SHARPE


def test_fit_uses_configured_stress_signs(
    test_settings: Settings,
    synthetic_sample: tuple[pd.DataFrame, pd.Series],
) -> None:
    """``model.stress_signs`` flows into the assignment and changes it."""
    features, returns = synthetic_sample

    default_fit = build_classifier(test_settings.model).fit(features, returns, train_end=None)

    custom_signs = {"realised_vol": -1, "vix_term_spread": 1}
    custom_model = ModelSettings.model_validate(
        {**test_settings.model.model_dump(), "stress_signs": custom_signs}
    )
    custom_fit = build_classifier(custom_model).fit(features, returns, train_end=None)

    # The attached stress scores equal the custom-sign computation exactly.
    expected = stress_scores(
        custom_fit.estimate.emission.means,
        custom_fit.standardiser.columns,
        signs=custom_signs,
    )
    for stat in custom_fit.assignment.statistics:
        assert stat.stress_score == pytest.approx(expected[stat.state])

    # Same fit, different signs: the label assignment must change.
    assert custom_fit.assignment.mapping != default_fit.assignment.mapping


def test_stress_signs_none_means_defaults(
    test_settings: Settings,
    synthetic_sample: tuple[pd.DataFrame, pd.Series],
) -> None:
    """``stress_signs=None`` resolves to ``STRESS_SIGNS``, never merged."""
    assert ModelSettings().stress_signs is None
    assert ModelSettings().effective_stress_signs is STRESS_SIGNS

    # An explicit copy of the defaults must reproduce the default assignment.
    features, returns = synthetic_sample
    explicit = ModelSettings.model_validate(
        {**test_settings.model.model_dump(), "stress_signs": dict(STRESS_SIGNS)}
    )
    default_fit = build_classifier(test_settings.model).fit(features, returns, train_end=None)
    explicit_fit = build_classifier(explicit).fit(features, returns, train_end=None)
    assert explicit_fit.assignment.mapping == default_fit.assignment.mapping


def test_stress_signs_coverage_validated_when_stress_in_effect() -> None:
    """The coverage check fires only when STRESS ranking is in effect."""
    # hmm defaults to STRESS: an uncovered dict is rejected.
    with pytest.raises(ValidationError, match="stress sign"):
        ModelSettings(stress_signs={"foo": 1})

    # The same failure surfaces through the root Settings.
    with pytest.raises(ValidationError, match="stress sign"):
        Settings.model_validate({"model": {"stress_signs": {"foo": 1}}})

    # gmm defaults to VOLATILITY: coverage is irrelevant and accepted.
    relaxed = ModelSettings(classifier="gmm", stress_signs={"foo": 1})
    assert relaxed.stress_signs == {"foo": 1}

    # An explicit non-STRESS method skips the check too.
    sharpe = ModelSettings(ranking_method=RankingMethod.SHARPE, stress_signs={"foo": 1})
    assert sharpe.stress_signs == {"foo": 1}

    # Forcing STRESS on gmm re-applies the coverage requirement.
    with pytest.raises(ValidationError, match="stress sign"):
        ModelSettings(
            classifier="gmm",
            ranking_method=RankingMethod.STRESS,
            stress_signs={"foo": 1},
        )

    # Values are restricted to +/-1.
    with pytest.raises(ValidationError):
        ModelSettings.model_validate({"stress_signs": {"realised_vol": 2}})


def test_classifier_defaults_match_ranking_table() -> None:
    """Every registry classifier's ClassVar reads DEFAULT_RANKING_METHODS."""
    for name, cls in CLASSIFIERS.items():
        assert cls.ranking_method is DEFAULT_RANKING_METHODS[name]


def test_stress_signs_mapping_is_immutable() -> None:
    """``STRESS_SIGNS`` is a read-only view, identical across re-exports."""
    assert STRESS_SIGNS is TYPES_STRESS_SIGNS
    with pytest.raises(TypeError):
        STRESS_SIGNS["realised_vol"] = -1  # type: ignore[index]
