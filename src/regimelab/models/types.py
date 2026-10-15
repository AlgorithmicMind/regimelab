"""Enums and Pydantic schemas shared across the engine.

Design notes
------------
Latent-state models (HMM, GMM, Markov-switching) return *unordered* integer
states: the label attached to a state is an artefact of the EM initialisation,
not a property of the market. Two refits on the same data with different seeds
can swap ``state_0`` and ``state_2``. Mapping raw state indices onto the
semantic regimes defined here is therefore never a matter of ``regimes[state]``
-- it requires the canonical ordering implemented in
:mod:`regimelab.models.labeling`.

The ladder defined by :data:`REGIME_LADDER` is the single source of truth for
that ordering: index 0 is the most risk-on regime, index -1 the most risk-off.
"""

from __future__ import annotations

import datetime as dt
import math
from collections.abc import Mapping
from enum import StrEnum
from types import MappingProxyType
from typing import Annotated, Final, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

__all__ = [
    "DEFAULT_RANKING_METHODS",
    "REGIME_LADDER",
    "STRESS_SIGNS",
    "AllocationWeights",
    "FeatureSnapshot",
    "MarketRegime",
    "ModelMetadata",
    "RankingMethod",
    "RegimeProbabilities",
    "RegimeStatistics",
    "TelemetryPayload",
    "regime_ladder",
]

# Tolerance used when validating that a discrete distribution sums to one.
_PROB_SUM_TOL = 1e-6


class MarketRegime(StrEnum):
    """Discrete market regime emitted by the classifiers.

    Ordered semantically from risk-on to risk-off; see :data:`REGIME_LADDER`.
    """

    BULL_TREND = "BULL_TREND"
    """Positive drift, contained realised volatility, low systemic fragility."""

    NEUTRAL_TRANSITION = "NEUTRAL_TRANSITION"
    """Drift indistinguishable from zero. Typically a regime-change antechamber."""

    HIGH_VOL_BEAR = "HIGH_VOL_BEAR"
    """Negative drift with elevated volatility, but diversification still works."""

    RISK_OFF = "RISK_OFF"
    """Crisis state: correlations converge to one and diversification fails."""

    @property
    def risk_rank(self) -> int:
        """Position on the risk ladder. ``0`` is the most risk-on regime."""
        return REGIME_LADDER.index(self)

    @property
    def is_risk_on(self) -> bool:
        """Whether the regime warrants a net-long risk posture."""
        return self.risk_rank <= 1


REGIME_LADDER: tuple[MarketRegime, ...] = (
    MarketRegime.BULL_TREND,
    MarketRegime.NEUTRAL_TRANSITION,
    MarketRegime.HIGH_VOL_BEAR,
    MarketRegime.RISK_OFF,
)
"""Canonical risk ordering, from most risk-on to most risk-off."""


# Sub-ladders for lower-cardinality models. Collapsing happens from the middle
# of the ladder outwards so that the extremes are always representable.
_LADDERS: dict[int, tuple[MarketRegime, ...]] = {
    2: (MarketRegime.BULL_TREND, MarketRegime.RISK_OFF),
    3: (
        MarketRegime.BULL_TREND,
        MarketRegime.NEUTRAL_TRANSITION,
        MarketRegime.RISK_OFF,
    ),
    4: REGIME_LADDER,
}


def regime_ladder(n_states: int) -> tuple[MarketRegime, ...]:
    """Return the risk-ordered regime ladder for an ``n_states`` model.

    Parameters
    ----------
    n_states
        Number of latent states in the fitted model. Supported values are 2, 3
        and 4.

    Returns
    -------
    tuple of MarketRegime
        Ladder ordered from most risk-on to most risk-off.

    Raises
    ------
    ValueError
        If ``n_states`` has no defined semantic ladder. We refuse to guess: a
        5-state model has no defensible mapping onto four named regimes.
    """
    try:
        return _LADDERS[n_states]
    except KeyError:
        supported = ", ".join(str(k) for k in sorted(_LADDERS))
        msg = f"no regime ladder defined for n_states={n_states} (supported: {supported})"
        raise ValueError(msg) from None


class RankingMethod(StrEnum):
    """Strategy for ordering latent states from risk-on to risk-off."""

    SHARPE = "sharpe"
    """Rank by ``mean_return / volatility``. It blends both dimensions, so a
    state with modest drift and very low volatility correctly ranks above one
    with similar drift and crisis-level volatility. Not the default for any
    shipped classifier -- see :data:`DEFAULT_RANKING_METHODS` (``hmm`` and
    ``hamilton`` use :attr:`STRESS`, ``gmm`` uses :attr:`VOLATILITY`)."""

    MEAN_RETURN = "mean_return"
    """Rank by conditional mean return alone. Cleaner to explain, but it cannot
    distinguish a calm grind higher from a violent bear-market rally."""

    VOLATILITY = "volatility"
    """Rank by ascending conditional volatility. Appropriate when the model is
    fitted purely on volatility features and drift is not identified."""

    STRESS = "stress"
    """Rank by the mean signed standardised emission mean over features with a
    known stress sign (:data:`STRESS_SIGNS`). Higher
    stress sorts more risk-off; the positive-drift guardrail does not apply."""


# Sign of each feature's contribution to the stress score: +1 means a higher
# standardised emission mean indicates more stress, -1 the opposite.
#
# The defaults are a reasoned convention, not measured ground truth: realised
# volatility and the absorption measures going up means stress; the VIX term
# spread, trailing return and breadth going up means calm.
# ``mean_correlation`` is deliberately excluded: on real data (SPY plus a
# sector/defensive universe, 2008-2026) the crisis state shows the *lowest*
# mean cross-asset correlation -- stock/bond correlation goes negative under
# stress -- so its sign is not stable across eras. A configuration that
# disagrees supplies ``ModelSettings.stress_signs``, which replaces this table
# entirely; the two are never merged.
STRESS_SIGNS: Final[Mapping[str, Literal[-1, 1]]] = MappingProxyType(
    {
        "realised_vol": +1,
        "vix_term_spread": -1,
        "absorption_delta": +1,
        "absorption_ratio": +1,
        "trailing_return": -1,
        "breadth": -1,
        "breadth_momentum": -1,
    },
)
"""Default signed stress contribution per feature, used when
``ModelSettings.stress_signs`` is ``None``."""

# Single source of truth for each classifier's default ranking method: the
# ``ranking_method`` ClassVar on every model and the ``None`` branch of
# ``ModelSettings`` both read this table.
DEFAULT_RANKING_METHODS: Final[Mapping[str, RankingMethod]] = {
    "hmm": RankingMethod.STRESS,
    "hamilton": RankingMethod.STRESS,
    "gmm": RankingMethod.VOLATILITY,
}
"""Default :class:`RankingMethod` per classifier name."""


class _Frozen(BaseModel):
    """Base for immutable, strictly-validated payloads."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        validate_assignment=True,
        ser_json_inf_nan="constants",
    )


class RegimeProbabilities(_Frozen):
    """Posterior distribution over regimes at a single point in time.

    Only *filtered* probabilities belong here. Smoothed probabilities condition
    on the full sample and are therefore look-ahead contaminated; they may be
    used for research plots but must never reach a trading decision.
    """

    probabilities: dict[MarketRegime, Annotated[float, Field(ge=0.0, le=1.0)]]

    @model_validator(mode="after")
    def _check_normalised(self) -> Self:
        total = math.fsum(self.probabilities.values())
        if not math.isclose(total, 1.0, abs_tol=_PROB_SUM_TOL):
            msg = f"regime probabilities must sum to 1.0, got {total!r}"
            raise ValueError(msg)
        return self

    @property
    def argmax(self) -> MarketRegime:
        """Most likely regime.

        Ties are broken towards the more risk-off regime: when the model cannot
        distinguish two states, the conservative reading is the correct default
        for a risk system.
        """
        return max(
            self.probabilities,
            key=lambda regime: (self.probabilities[regime], regime.risk_rank),
        )

    @property
    def confidence(self) -> float:
        """Probability mass assigned to the modal regime."""
        return self.probabilities[self.argmax]

    @property
    def entropy(self) -> float:
        """Shannon entropy in nats. High entropy means an ambiguous read."""
        return -math.fsum(p * math.log(p) for p in self.probabilities.values() if p > 0.0)

    @property
    def normalised_entropy(self) -> float:
        """Entropy rescaled to ``[0, 1]`` by the uniform-distribution maximum."""
        n = len(self.probabilities)
        if n <= 1:
            return 0.0
        return self.entropy / math.log(n)

    @property
    def risk_off_mass(self) -> float:
        """Total probability assigned to risk-off regimes.

        More robust than ``argmax`` for sizing: a 45/45 split between the two
        bearish states is a risk-off signal that ``argmax`` would report with
        misleadingly low confidence.
        """
        return math.fsum(p for regime, p in self.probabilities.items() if not regime.is_risk_on)


class FeatureSnapshot(_Frozen):
    """Systemic-risk telemetry observable at a single as-of date.

    Every field is ``None``-able: a feature whose trailing window is not yet
    full must report absence rather than a silently back-filled value.
    """

    absorption_ratio: Annotated[float, Field(ge=0.0, le=1.0)] | None = None
    """Kritzman-Li absorption ratio: variance share explained by the leading
    eigenvectors. High values indicate a tightly coupled, fragile market."""

    absorption_delta: float | None = None
    """Standardised 15-day shift in the absorption ratio. This, not the level,
    is the tradable signal in Kritzman & Li (2010)."""

    vix_term_spread: float | None = None
    """VIX3M/VIX ratio minus one. Negative means backwardation (stress)."""

    realised_vol: Annotated[float, Field(ge=0.0)] | None = None
    """Annualised Garman-Klass realised volatility of the risk asset."""

    breadth: Annotated[float, Field(ge=0.0, le=1.0)] | None = None
    """Share of the tracked cross-section trading above its 200-day SMA."""

    breadth_momentum: float | None = None
    """Change in breadth over the trailing month; a deterioration oscillator."""

    mean_correlation: Annotated[float, Field(ge=-1.0, le=1.0)] | None = None
    """Average pairwise cross-asset correlation over the rolling window."""

    trailing_return: float | None = None
    """Trailing 21-session log return of the risk asset."""

    def is_complete(self) -> bool:
        """Whether every feature has a value."""
        return all(getattr(self, name) is not None for name in type(self).model_fields)

    def missing(self) -> tuple[str, ...]:
        """Names of the features that are unavailable at this as-of date."""
        return tuple(name for name in type(self).model_fields if getattr(self, name) is None)


class AllocationWeights(_Frozen):
    """Target portfolio weights. Long-only, fully invested, cash residual."""

    weights: dict[str, Annotated[float, Field(ge=0.0, le=1.0)]]

    @model_validator(mode="after")
    def _check_budget(self) -> Self:
        total = math.fsum(self.weights.values())
        if total > 1.0 + _PROB_SUM_TOL:
            msg = f"allocation cannot exceed 100% of capital, got {total:.6f}"
            raise ValueError(msg)
        return self

    @property
    def invested(self) -> float:
        """Fraction of capital deployed in risk assets."""
        return math.fsum(self.weights.values())

    @property
    def cash(self) -> float:
        """Residual cash weight."""
        return max(0.0, 1.0 - self.invested)

    def normalise(self, *, symbols: list[str] | None = None) -> dict[str, float]:
        """Return non-negative symbol weights that sum to at most one.

        Symbols absent from ``self.weights`` are assigned 0.0. Symbols outside the
        provided ``symbols`` set are also assigned 0.0, so an allocation table
        that contains a symbol not present in the current data panel does not
        leak into the portfolio.

        Parameters
        ----------
        symbols
            Allow-list of investable symbols.

        Returns
        -------
        dict of str to float
            Normalised weights. The values may sum to less than one; the residual
            is cash.
        """
        allowed = set(symbols) if symbols is not None else set(self.weights)
        return {
            symbol: max(0.0, min(1.0, float(weight)))
            for symbol, weight in self.weights.items()
            if symbol in allowed
        }


class RegimeStatistics(_Frozen):
    """Economic summary of one canonical regime, from a fitted assignment.

    These describe the model's *training* window, not the period the payload
    reports on: they are the moments of the return distribution conditional on
    the states mapped to ``regime``. Several raw states can share a regime --
    the positive-drift guardrail maps extra positive-drift states onto the
    most risk-off non-bear label -- so the moments are the occupancy-weighted
    combination over ``raw_states``, mirroring how
    :func:`~regimelab.models.labeling.relabel_probabilities` sums their
    posteriors.
    """

    regime: MarketRegime
    raw_states: tuple[int, ...]
    """Raw latent-state indices mapped to this regime, sorted ascending."""
    occupancy: Annotated[float, Field(ge=0.0, le=1.0)]
    """Combined share of the training sample assigned to the member states."""
    ann_return: float | None = None
    """Posterior-weighted conditional mean return, annualised. ``None`` when
    no member state carries weight."""
    ann_volatility: Annotated[float, Field(ge=0.0)] | None = None
    """Posterior-weighted conditional volatility, annualised. ``None`` when no
    member state carries weight."""
    sharpe: float | None = None
    """``ann_return / ann_volatility``; ``0.0`` for a degenerate state,
    ``None`` when no member state carries weight."""
    stress_score: float | None = None
    """Signed stress score under STRESS ranking; ``None`` otherwise."""
    n_observations: Annotated[int, Field(ge=0)]
    """Combined effective observations assigned to the member states."""

    @model_validator(mode="after")
    def _check_raw_states(self) -> Self:
        if not self.raw_states:
            msg = "raw_states must not be empty"
            raise ValueError(msg)
        if any(state < 0 for state in self.raw_states):
            msg = f"raw_states must be non-negative: {self.raw_states}"
            raise ValueError(msg)
        if tuple(sorted(self.raw_states)) != self.raw_states:
            msg = f"raw_states must be sorted ascending: {self.raw_states}"
            raise ValueError(msg)
        if len(set(self.raw_states)) != len(self.raw_states):
            msg = f"raw_states must be unique: {self.raw_states}"
            raise ValueError(msg)
        return self


class ModelMetadata(_Frozen):
    """Provenance of the classifier that produced a payload."""

    name: str
    """Classifier identifier, e.g. ``"hmm"``, ``"gmm"``, ``"hamilton"``."""

    n_states: Annotated[int, Field(ge=2, le=4)]
    seed: int
    """Seed actually used for the fit. Required for the determinism guarantee."""

    train_start: dt.date
    train_end: dt.date
    """Last observation in the training window. Must satisfy
    ``train_end <= TelemetryPayload.asof`` -- enforced by the payload."""

    n_train_observations: Annotated[int, Field(ge=1)]
    converged: bool
    log_likelihood: float | None = None
    feature_names: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _check_window(self) -> Self:
        if self.train_end < self.train_start:
            msg = f"train_end {self.train_end} precedes train_start {self.train_start}"
            raise ValueError(msg)
        return self


class TelemetryPayload(_Frozen):
    """Everything the engine knows at one as-of date.

    This is the boundary object between computation and presentation: the Rich
    dashboard and the HTML reporter render *only* this, and never touch the data
    layer. That keeps rendering trivially testable and guarantees the UI cannot
    introduce look-ahead by re-reading a dataframe.
    """

    asof: dt.date
    """Decision date. All inputs are observable at or before this date."""

    session_date: dt.date
    """Last master-calendar session at or before ``asof``."""

    observation_date: dt.date
    """Date of the last complete feature row used for the posterior."""

    regime: MarketRegime
    probabilities: RegimeProbabilities
    features: FeatureSnapshot
    model: ModelMetadata
    target_weights: AllocationWeights | None = None
    state_statistics: tuple[RegimeStatistics, ...] = ()
    """Per-regime statistics of the signing model's training window --
    ``model.train_start`` through ``model.train_end`` -- not of the live
    window the payload reports on."""
    warnings: tuple[str, ...] = ()

    @property
    def is_stale(self) -> bool:
        """Whether the posterior was computed on data older than the latest
        session -- a data problem, not a calendar one."""
        return self.observation_date < self.session_date

    @model_validator(mode="after")
    def _check_session_ordering(self) -> Self:
        if self.observation_date > self.session_date:
            msg = (
                f"observation_date {self.observation_date} is after "
                f"session_date {self.session_date}"
            )
            raise ValueError(msg)
        if self.session_date > self.asof:
            msg = f"session_date {self.session_date} is after asof {self.asof}"
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _check_causality(self) -> Self:
        """Reject payloads whose training window extends beyond the as-of date.

        This is the last line of defence against look-ahead bias. It is cheap,
        it runs on every payload, and it has caught real bugs during walk-forward
        development -- keep it.
        """
        if self.model.train_end > self.asof:
            msg = (
                f"look-ahead detected: model trained through {self.model.train_end} "
                f"but payload is as-of {self.asof}"
            )
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _check_regime_consistency(self) -> Self:
        if self.regime not in self.probabilities.probabilities:
            msg = f"regime {self.regime} absent from the posterior distribution"
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _check_state_statistics_unique(self) -> Self:
        regimes = [entry.regime for entry in self.state_statistics]
        if len(set(regimes)) != len(regimes):
            msg = "state_statistics contains duplicate regimes"
            raise ValueError(msg)
        return self

    def statistics_for(self, regime: MarketRegime) -> RegimeStatistics:
        """Statistics for ``regime``; raises when the payload has none."""
        for entry in self.state_statistics:
            if entry.regime is regime:
                return entry
        covered = [entry.regime.value for entry in self.state_statistics]
        msg = f"no state statistics for {regime.value}; payload covers {covered}"
        raise KeyError(msg)
