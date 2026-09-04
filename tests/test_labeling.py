"""Regression tests for canonical state labelling."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from regimelab.models.labeling import (
    RankingMethod,
    StateStatistics,
    assign_labels,
    relabel_probabilities,
)
from regimelab.models.types import MarketRegime, regime_ladder


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
