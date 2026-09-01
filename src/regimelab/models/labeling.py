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
content* of each state: its conditional mean return and conditional volatility.
Sort by that, then map onto the risk-ordered ladder in
:data:`regimelab.models.types.REGIME_LADDER`. The mapping becomes a deterministic
function of the fitted parameters, invariant to permutation, and stable across
refits.

Ties are broken by volatility and finally by the raw state index, so the ordering
is total. Total ordering matters more than the ideal tie-break rule: a
non-deterministic comparator would reintroduce exactly the instability this
module exists to remove.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from regimelab.models.types import MarketRegime, regime_ladder

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = [
    "LabelAssignment",
    "OccupancyError",
    "RankingMethod",
    "StateStatistics",
    "assign_labels",
    "relabel_probabilities",
    "summarise_states",
]


class RankingMethod(StrEnum):
    """Strategy for ordering latent states from risk-on to risk-off."""

    SHARPE = "sharpe"
    """Rank by ``mean_return / volatility``. The default: it blends both
    dimensions, so a state with modest drift and very low volatility correctly
    ranks above one with similar drift and crisis-level volatility."""

    MEAN_RETURN = "mean_return"
    """Rank by conditional mean return alone. Cleaner to explain, but it cannot
    distinguish a calm grind higher from a violent bear-market rally."""

    VOLATILITY = "volatility"
    """Rank by ascending conditional volatility. Appropriate when the model is
    fitted purely on volatility features and drift is not identified."""


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
    """

    state: int
    occupancy: float
    mean_return: float
    volatility: float
    n_observations: int

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


def assign_labels(
    statistics: Sequence[StateStatistics],
    *,
    method: RankingMethod = RankingMethod.SHARPE,
    min_occupancy: float = 0.0,
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

    Returns
    -------
    LabelAssignment
        Permutation-invariant state-to-regime mapping.

    Raises
    ------
    OccupancyError
        If a state is below ``min_occupancy``.
    ValueError
        If the state count has no defined ladder.
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
    mapping = {stat.state: ladder[rank] for rank, stat in enumerate(ordered)}

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

    ladder = regime_ladder(len(assignment.mapping))
    return renamed.loc[:, [regime.value for regime in ladder]]
