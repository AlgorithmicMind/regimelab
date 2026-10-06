"""Canonical state labelling: the fix for label switching.

The problem
-----------
The likelihood of a mixture or hidden-Markov model is invariant under
permutation of its states. ``state_0`` is whatever EM happened to converge on
first; it carries no meaning. Consequences, all of which have shipped to
production in real systems:

- Two fits differing only in seed swap the labels, so ``regime == BULL_TREND``
  flips while the model's actual belief is unchanged.
- Inside a walk-forward loop, each refit can permute the states, injecting
  phantom regime transitions on refit boundaries. Turnover explodes and the
  backtest attributes the resulting cost to a strategy that never traded.
- A determinism test comparing two identical runs passes, because both runs share
  a seed, while the labelling remains meaningless. The test provides false
  assurance.

Writing ``regimes[state_index]`` is therefore always a bug.

The fix
-------
Discard the model's state indices and derive an ordering from the *economic
content* of each state: its conditional mean return and conditional volatility,
or -- under :attr:`~regimelab.models.types.RankingMethod.STRESS` -- the signed
means of its emission distribution over the features with a known stress sign
(:data:`STRESS_SIGNS`). Sort by that, then map onto the risk-ordered ladder in
:data:`regimelab.models.types.REGIME_LADDER`. The mapping becomes a
deterministic function of the fitted parameters, invariant to permutation, and
stable across refits.

Ties are broken by volatility and finally by the raw state index, so the ordering
is total. Total ordering matters more than the ideal tie-break rule: a
non-deterministic comparator would reintroduce exactly the instability this
module exists to remove.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, cast

import numpy as np
import pandas as pd

from regimelab.models.types import (
    STRESS_SIGNS,
    MarketRegime,
    RankingMethod,
    regime_ladder,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

__all__ = [
    "STRESS_SIGNS",
    "LabelAssignment",
    "OccupancyError",
    "RankingMethod",
    "StateStatistics",
    "assign_labels",
    "relabel_probabilities",
    "stress_scores",
    "summarise_states",
]


class OccupancyError(ValueError):
    """Raised when a fitted state captures too little of the sample.

    A state holding 1% of observations is not a market regime, it is EM fitting
    a handful of outliers. Its mean and covariance are estimated from almost no
    data, so its position on the risk ladder is arbitrary -- and because the
    ladder is a sort, one badly-estimated state corrupts the labelling of every
    other state. Failing loudly beats emitting a confidently mislabelled regime.
    """


@dataclass(frozen=True, slots=True)
class StateStatistics:
    """Economic summary of one fitted latent state.

    Attributes
    ----------
    state
        Raw index as produced by the estimator. Meaningless on its own; retained
        for traceability back to the underlying model object.
    occupancy
        Share of the training sample assigned to this state.
    mean_return
        Conditional mean of the reference return series, annualised.
    volatility
        Conditional standard deviation of the reference return series, annualised.
    n_observations
        Effective number of observations assigned to the state.
    stress_score
        Mean signed standardised emission mean over the signed stress
        features, attached only when ranked under
        :attr:`~regimelab.models.types.RankingMethod.STRESS`. Higher means more
        risk-off.
    """

    state: int
    occupancy: float
    mean_return: float
    volatility: float
    n_observations: int
    stress_score: float | None = None

    @property
    def sharpe(self) -> float:
        """Conditional return-to-risk ratio.

        Returns ``0.0`` for a degenerate zero-volatility state instead of
        ``inf``, so the sort stays well-defined.
        """
        if not np.isfinite(self.volatility) or self.volatility <= 0.0:
            return 0.0
        return self.mean_return / self.volatility

    def score(self, method: RankingMethod) -> float:
        """Risk-on score under ``method``. Higher means more risk-on."""
        if method is RankingMethod.SHARPE:
            return self.sharpe
        if method is RankingMethod.MEAN_RETURN:
            return self.mean_return
        if method is RankingMethod.STRESS:
            return -self.stress_score if self.stress_score is not None else float("nan")
        return -self.volatility


@dataclass(frozen=True, slots=True)
class LabelAssignment:
    """A permutation-invariant mapping from raw states to named regimes.

    Attributes
    ----------
    mapping
        Raw state index to :class:`~regimelab.models.types.MarketRegime`.
    statistics
        Per-state summary, ordered risk-on to risk-off.
    method
        Ranking method used.
    """

    mapping: dict[int, MarketRegime]
    statistics: tuple[StateStatistics, ...]
    method: RankingMethod

    @property
    def inverse(self) -> dict[MarketRegime, int]:
        """Regime to raw state index."""
        return {regime: state for state, regime in self.mapping.items()}

    def regime_of(self, state: int) -> MarketRegime:
        """Named regime for a raw state index."""
        try:
            return self.mapping[state]
        except KeyError:
            msg = f"state {state} was not present in the fit that produced this assignment"
            raise KeyError(msg) from None

    def describe(self) -> pd.DataFrame:
        """Tabular summary for reports and the dashboard."""
        return pd.DataFrame(
            [
                {
                    "regime": self.mapping[stat.state].value,
                    "raw_state": stat.state,
                    "occupancy": stat.occupancy,
                    "ann_return": stat.mean_return,
                    "ann_volatility": stat.volatility,
                    "sharpe": stat.sharpe,
                    "stress_score": stat.stress_score,
                    "observations": stat.n_observations,
                }
                for stat in self.statistics
            ],
        )


def summarise_states(
    reference_returns: pd.Series,
    posteriors: pd.DataFrame,
    *,
    trading_days: int = 252,
) -> tuple[StateStatistics, ...]:
    """Summarise each state's conditional return distribution.

    Uses *posterior-weighted* moments rather than hard assignments. Hard
    assignment throws away the model's uncertainty and, for a state that is
    rarely the modal choice, can leave too few observations to estimate a mean
    at all -- producing exactly the unstable statistics that corrupt the ladder.

    Parameters
    ----------
    reference_returns
        Daily returns of the reference risk asset, aligned to ``posteriors``.
    posteriors
        ``(n_observations, n_states)`` posterior probabilities, integer-named
        columns matching the raw state indices.
    trading_days
        Sessions per year, for annualisation.

    Returns
    -------
    tuple of StateStatistics
        One entry per state, in raw-state order.
    """
    aligned_returns, aligned_posteriors = reference_returns.align(posteriors, join="inner", axis=0)
    returns = aligned_returns.to_numpy(dtype="float64")
    weights_matrix = aligned_posteriors.to_numpy(dtype="float64")

    finite = np.isfinite(returns)
    returns = returns[finite]
    weights_matrix = weights_matrix[finite]

    if returns.size == 0:
        msg = "no overlapping finite observations between returns and posteriors"
        raise ValueError(msg)

    total_weight = weights_matrix.sum()
    statistics: list[StateStatistics] = []

    for position, column in enumerate(aligned_posteriors.columns):
        weights = weights_matrix[:, position]
        weight_sum = float(weights.sum())

        if weight_sum <= 0.0:
            statistics.append(
                StateStatistics(
                    state=int(column),
                    occupancy=0.0,
                    mean_return=float("nan"),
                    volatility=float("nan"),
                    n_observations=0,
                ),
            )
            continue

        mean = float(np.dot(weights, returns) / weight_sum)
        variance = float(np.dot(weights, (returns - mean) ** 2) / weight_sum)

        statistics.append(
            StateStatistics(
                state=int(column),
                occupancy=weight_sum / float(total_weight),
                mean_return=mean * trading_days,
                volatility=float(np.sqrt(max(variance, 0.0)) * np.sqrt(trading_days)),
                n_observations=round(weight_sum),
            ),
        )

    return tuple(statistics)


def stress_scores(
    emission_means: np.ndarray,
    feature_names: Sequence[str],
    *,
    signs: Mapping[str, int] = STRESS_SIGNS,
) -> np.ndarray:
    """Per-state stress score from signed standardised emission means.

    Parameters
    ----------
    emission_means
        ``(n_states, n_features)`` matrix of per-state emission means in
        standardised feature space, columns ordered as ``feature_names``.
    feature_names
        Feature name for each column of ``emission_means``.
    signs
        Stress sign per feature name: ``+1`` means a higher standardised
        emission mean indicates more stress, ``-1`` the opposite. Defaults to
        :data:`STRESS_SIGNS`; a custom mapping replaces it entirely and is
        never merged.

    Returns
    -------
    numpy.ndarray
        ``(n_states,)`` mean of ``sign * mean`` over the features that carry a
        stress sign in ``signs``. Higher means more risk-off.

    Raises
    ------
    ValueError
        If ``emission_means`` does not have one column per feature name, or if
        none of the features carries a stress sign.
    """
    names = [str(name) for name in feature_names]
    means = np.asarray(emission_means, dtype="float64")
    if means.ndim != 2 or means.shape[1] != len(names):
        msg = f"emission_means with shape {means.shape} does not match {len(names)} feature names"
        raise ValueError(msg)

    signed = [(i, signs[name]) for i, name in enumerate(names) if name in signs]
    if not signed:
        valid = ", ".join(sorted(signs))
        msg = f"no signed stress features among {names}; valid keys: {valid}"
        raise ValueError(msg)

    columns = np.array([i for i, _ in signed])
    sign_values = np.array([sign for _, sign in signed], dtype="float64")
    return cast(np.ndarray, (means[:, columns] * sign_values).mean(axis=1))


def assign_labels(
    statistics: Sequence[StateStatistics],
    *,
    method: RankingMethod = RankingMethod.SHARPE,
    min_occupancy: float = 0.0,
    emission_means: np.ndarray | None = None,
    feature_names: Sequence[str] | None = None,
    signs: Mapping[str, int] = STRESS_SIGNS,
) -> LabelAssignment:
    """Map raw states onto the canonical regime ladder.

    Parameters
    ----------
    statistics
        Per-state summaries from :func:`summarise_states`.
    method
        Ranking strategy.
    min_occupancy
        Reject the fit if any state's occupancy falls below this share.
    emission_means
        ``(n_states, n_features)`` standardised emission means, required when
        ``method`` is :attr:`~regimelab.models.types.RankingMethod.STRESS`.
        Row ``s`` must correspond to raw state ``s``.
    feature_names
        Feature name for each column of ``emission_means``; required together
        with ``emission_means`` under STRESS.
    signs
        Stress sign per feature name, forwarded to :func:`stress_scores` under
        STRESS. Defaults to :data:`STRESS_SIGNS` and is ignored by the other
        methods.

    Returns
    -------
    LabelAssignment
        Permutation-invariant state-to-regime mapping.

    Raises
    ------
    OccupancyError
        If a state is below ``min_occupancy``.
    ValueError
        If the state count has no defined ladder, if STRESS ranking is missing
        its emission inputs, or if a state index is out of range for
        ``emission_means``.
    """
    if not statistics:
        msg = "cannot assign labels without state statistics"
        raise ValueError(msg)

    ladder = regime_ladder(len(statistics))

    starved = [s for s in statistics if s.occupancy < min_occupancy]
    if starved:
        details = ", ".join(f"state {s.state}: {s.occupancy:.2%}" for s in starved)
        msg = (
            f"fitted states below the {min_occupancy:.2%} occupancy floor ({details}); "
            "the fit is capturing outliers rather than regimes -- reduce n_states, "
            "lengthen the training window, or relax min_state_occupancy"
        )
        raise OccupancyError(msg)

    if method is RankingMethod.STRESS:
        if emission_means is None or feature_names is None:
            msg = (
                "STRESS ranking requires both emission_means and feature_names: "
                "the score is computed from standardised emission means"
            )
            raise ValueError(msg)
        scores = stress_scores(emission_means, feature_names, signs=signs)
        if any(not 0 <= stat.state < scores.shape[0] for stat in statistics):
            msg = (
                f"state index out of range for {scores.shape[0]} emission rows: "
                f"{[stat.state for stat in statistics]}"
            )
            raise ValueError(msg)
        statistics = tuple(
            replace(stat, stress_score=float(scores[stat.state])) for stat in statistics
        )

    # Total ordering: score descending, then volatility ascending, then raw index.
    # The final key guarantees determinism even for exactly-tied states.
    def sort_key(stat: StateStatistics) -> tuple[float, float, int]:
        score = stat.score(method)
        # NaN would make the comparator non-transitive; sink such states to the
        # risk-off end, which is the conservative placement.
        safe_score = score if np.isfinite(score) else -np.inf
        safe_vol = stat.volatility if np.isfinite(stat.volatility) else np.inf
        return (-safe_score, safe_vol, stat.state)

    ordered = tuple(sorted(statistics, key=sort_key))

    # Economic guardrail: positive-drift states can never be mapped onto bearish
    # canonical regimes.  If the strict Sharpe ladder would place a positive-mean
    # state in HIGH_VOL_BEAR or RISK_OFF, cap it at the most risk-off non-bear
    # label available in the ladder (NEUTRAL_TRANSITION for 3- and 4-state models,
    # BULL_TREND for 2-state models).  When several positive states compete for
    # the same non-bear label, the later duplicates are resolved by
    # :func:`relabel_probabilities`, which aggregates raw-state probabilities per
    # canonical regime.  The guardrail does not apply under STRESS, where the
    # signed stress score is the sole ordering criterion.
    non_bear_labels = [
        label
        for label in ladder
        if label not in (MarketRegime.HIGH_VOL_BEAR, MarketRegime.RISK_OFF)
    ]
    cap_label = non_bear_labels[-1] if non_bear_labels else ladder[0]

    mapping: dict[int, MarketRegime] = {}
    for rank, stat in enumerate(ordered):
        label = ladder[rank]
        if (
            method is not RankingMethod.STRESS
            and label in (MarketRegime.HIGH_VOL_BEAR, MarketRegime.RISK_OFF)
            and np.isfinite(stat.mean_return)
            and stat.mean_return >= 0.0
        ):
            label = cap_label
        mapping[stat.state] = label

    return LabelAssignment(mapping=mapping, statistics=ordered, method=method)


def relabel_probabilities(
    posteriors: pd.DataFrame,
    assignment: LabelAssignment,
) -> pd.DataFrame:
    """Rename posterior columns from raw states to named regimes.

    Parameters
    ----------
    posteriors
        ``(n_observations, n_states)`` frame with integer state columns.
    assignment
        Mapping produced by :func:`assign_labels`.

    Returns
    -------
    pandas.DataFrame
        Columns are :class:`~regimelab.models.types.MarketRegime` values ordered
        by the risk ladder, so downstream consumers can rely on column order as
        well as on names.

    Raises
    ------
    KeyError
        If a posterior column has no assignment.
    """
    unmapped = [c for c in posteriors.columns if int(c) not in assignment.mapping]
    if unmapped:
        msg = f"posterior columns without a label assignment: {unmapped}"
        raise KeyError(msg)

    renamed = posteriors.rename(
        columns={c: assignment.mapping[int(c)].value for c in posteriors.columns},
    )

    # The economic guardrail in :func:`assign_labels` may map several raw states
    # onto the same canonical regime (e.g. multiple positive-drift states capped
    # at NEUTRAL_TRANSITION).  Sum their probabilities so each canonical regime
    # appears exactly once.
    if len(set(renamed.columns)) < len(renamed.columns):
        renamed = renamed.T.groupby(level=0).sum().T

    ladder = regime_ladder(len(assignment.mapping))
    regime_names = [regime.value for regime in ladder]
    return renamed.reindex(columns=regime_names, fill_value=0.0)
