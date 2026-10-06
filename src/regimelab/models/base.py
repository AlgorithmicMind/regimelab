"""Shared machinery for the regime classifiers.

Every classifier follows the same lifecycle, and the parts that are easy to get
wrong live here rather than being reimplemented three times:

1. **Standardise** the feature matrix using moments from the *training window
   only* (:class:`CausalStandardiser`).
2. **Estimate** latent states -- the only step each classifier implements itself.
3. **Label** the states canonically (:mod:`regimelab.models.labeling`).
4. **Filter** causally over any span, including data the fit never saw
   (:mod:`regimelab.models.filters`).

The separation of *fit* from *filter* is what makes walk-forward evaluation
honest: parameters are estimated once on the training window and then frozen
while the filter advances through out-of-sample data. Refitting at every step
would let each day's parameters absorb information from that same day.
"""

from __future__ import annotations

import datetime as dt
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Self, cast

import numpy as np
import pandas as pd

from regimelab.models.filters import forward_filter, gaussian_log_density
from regimelab.models.labeling import (
    LabelAssignment,
    RankingMethod,
    StateStatistics,
    assign_labels,
    relabel_probabilities,
    summarise_states,
)
from regimelab.models.types import (
    MarketRegime,
    ModelMetadata,
    RegimeProbabilities,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from regimelab.settings import ModelSettings

__all__ = [
    "CausalStandardiser",
    "EmissionParameters",
    "FittedRegime",
    "InsufficientDataError",
    "RegimeModel",
    "StateEstimate",
    "probabilities_from_row",
]

# Floor for a feature's training standard deviation. Below this the feature is
# effectively constant over the window and dividing by it manufactures enormous
# spurious values out of numerical noise.
_MIN_SCALE = 1e-12


class InsufficientDataError(ValueError):
    """Raised when the training window is too small to fit the requested model."""


def probabilities_from_row(row: pd.Series) -> RegimeProbabilities:
    """Build the validated schema from a posterior row.

    The row's values are column labels holding regime values (e.g.
    ``"BULL_TREND"``), one probability per regime.
    """
    return RegimeProbabilities(
        probabilities={MarketRegime(str(name)): float(value) for name, value in row.items()},
    )


@dataclass(frozen=True, slots=True)
class CausalStandardiser:
    """Z-scoring with moments frozen from the training window.

    This is a look-ahead vector that is very easy to miss. Calling
    ``(X - X.mean()) / X.std()`` over the full sample embeds the *future* mean and
    variance of every feature into each historical observation. The 2008
    volatility feature then knows how volatile 2009 was going to be, and the
    resulting backtest is not merely optimistic -- it is not a backtest.

    Moments are therefore estimated once, on the training window, and reused
    verbatim for every subsequent observation.

    Attributes
    ----------
    columns
        Feature names in fitted order. Enforced on transform, because a silently
        reordered matrix produces plausible nonsense.
    centre
        Per-feature training mean.
    scale
        Per-feature training standard deviation, floored at :data:`_MIN_SCALE`.
    """

    columns: tuple[str, ...]
    centre: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, frame: pd.DataFrame) -> Self:
        """Estimate centring and scaling from ``frame``.

        Parameters
        ----------
        frame
            Training-window feature matrix with no missing values.

        Returns
        -------
        CausalStandardiser
            Fitted standardiser.
        """
        if frame.empty:
            msg = "cannot fit a standardiser on an empty frame"
            raise InsufficientDataError(msg)

        values = frame.to_numpy(dtype="float64")
        centre = np.nanmean(values, axis=0)
        scale = np.nanstd(values, axis=0, ddof=1)
        scale = np.where(np.isfinite(scale) & (scale > _MIN_SCALE), scale, 1.0)

        return cls(
            columns=tuple(str(c) for c in frame.columns),
            centre=centre,
            scale=scale,
        )

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        """Apply the frozen transformation.

        Parameters
        ----------
        frame
            Feature matrix containing at least the fitted columns.

        Returns
        -------
        numpy.ndarray
            Standardised values, columns in fitted order.

        Raises
        ------
        KeyError
            If a fitted column is absent.
        """
        missing = [c for c in self.columns if c not in frame.columns]
        if missing:
            msg = f"features missing at transform time: {missing}"
            raise KeyError(msg)

        values = cast(
            np.ndarray,
            frame.loc[:, list(self.columns)].to_numpy(dtype="float64"),
        )
        return cast(np.ndarray, (values - self.centre) / self.scale)


@dataclass(frozen=True, slots=True)
class EmissionParameters:
    """Gaussian emission parameters, in standardised feature space.

    Retained separately from the estimator object so the filter can be re-run on
    new data without the original library instance -- which also means a fit can
    be serialised and replayed.
    """

    means: np.ndarray
    covariances: np.ndarray

    def log_density(self, matrix: np.ndarray) -> np.ndarray:
        """Log emission density of every observation under every state."""
        return gaussian_log_density(matrix, self.means, self.covariances)


@dataclass(frozen=True, slots=True)
class StateEstimate:
    """Raw estimator output, before canonical labelling.

    Attributes
    ----------
    posteriors
        ``(n_observations, n_states)`` filtered posteriors over the training
        window, raw state order.
    emission
        Fitted Gaussian emission parameters.
    transition
        Row-stochastic transition matrix, or ``None`` for models without state
        persistence (a Gaussian mixture).
    log_likelihood
        Training log likelihood.
    converged
        Whether the estimator reported convergence.
    """

    posteriors: np.ndarray
    emission: EmissionParameters
    transition: np.ndarray | None
    log_likelihood: float
    converged: bool

    @property
    def n_states(self) -> int:
        """Number of latent states."""
        return cast(int, self.posteriors.shape[1])


@dataclass(frozen=True, slots=True)
class FittedRegime:
    """A fitted, canonically-labelled regime model.

    Attributes
    ----------
    classifier
        Classifier name.
    estimate
        Raw estimator output.
    assignment
        Permutation-invariant state-to-regime mapping.
    standardiser
        Feature standardiser with training-window moments.
    metadata
        Provenance for the payload schema.
    """

    classifier: str
    estimate: StateEstimate
    assignment: LabelAssignment
    standardiser: CausalStandardiser
    metadata: ModelMetadata

    @property
    def n_states(self) -> int:
        """Number of latent states."""
        return self.estimate.n_states

    def probabilities(
        self,
        features: pd.DataFrame,
        initial: np.ndarray | None = None,
    ) -> pd.DataFrame:
        """Filtered regime posteriors over an arbitrary span.

        Parameters
        ----------
        features
            Feature matrix covering the span of interest. Rows with any missing
            fitted feature are dropped -- the filter cannot form a likelihood for
            a partial observation, and imputing one would fabricate evidence.
        initial
            Optional prior over states for the *first* observation in ``features``.
            Used by the walk-forward loop to carry the filtered state from the end
            of one training window into the next out-of-sample segment. ``None``
            uses the stationary distribution or the mixture proportions.

        Returns
        -------
        pandas.DataFrame
            Columns named after :class:`~regimelab.models.types.MarketRegime`
            values, ordered by the risk ladder. Indexed by the retained dates.
        """
        usable = features.loc[:, list(self.standardiser.columns)].dropna(how="any")
        if usable.empty:
            return pd.DataFrame(
                columns=[r.value for r in self.assignment.mapping.values()],
                index=pd.DatetimeIndex([], name="date"),
                dtype="float64",
            )

        if initial is not None:
            initial = np.asarray(initial, dtype="float64")
            if initial.shape != (self.n_states,):
                shape = self.n_states
                msg = f"initial distribution must have shape ({shape},), got {initial.shape}"
                raise ValueError(msg)
            initial = initial / initial.sum()

        matrix = self.standardiser.transform(usable)
        log_emissions = self.estimate.emission.log_density(matrix)
        raw = self._filter_raw(log_emissions, initial=initial)

        posteriors = pd.DataFrame(
            raw,
            index=usable.index,
            columns=list(range(self.n_states)),
        )
        return relabel_probabilities(posteriors, self.assignment)

    def _filter_raw(
        self,
        log_emissions: np.ndarray,
        initial: np.ndarray | None = None,
    ) -> np.ndarray:
        """Convert log emission densities into filtered state posteriors."""
        if self.estimate.transition is None:
            # No state persistence: the posterior is the normalised emission
            # density weighted by the mixture proportions.
            weights = np.array(
                [
                    stat.occupancy
                    for stat in sorted(
                        self.assignment.statistics,
                        key=lambda s: s.state,
                    )
                ],
                dtype="float64",
            )
            weights = weights / weights.sum()
            log_joint = log_emissions + np.log(np.clip(weights, 1e-300, None))
            log_joint -= log_joint.max(axis=1, keepdims=True)
            joint = np.exp(log_joint)
            return cast(np.ndarray, joint / joint.sum(axis=1, keepdims=True))

        return forward_filter(log_emissions, self.estimate.transition, initial=initial).filtered

    def labels(self, features: pd.DataFrame) -> pd.Series:
        """Modal regime per date.

        Ties resolve towards the more risk-off regime, matching
        :attr:`~regimelab.models.types.RegimeProbabilities.argmax`.
        """
        posteriors = self.probabilities(features)
        if posteriors.empty:
            return pd.Series(dtype="object", name="regime")

        ranked = sorted(posteriors.columns, key=lambda c: MarketRegime(str(c)).risk_rank)
        # idxmax takes the first maximum; reversing the column order therefore
        # breaks ties towards the risk-off end of the ladder.
        modal = posteriors.loc[:, ranked[::-1]].idxmax(axis="columns")

        # Constructed with an explicit object dtype: pandas 3 stores the string
        # column labels in an Arrow-backed array, and `.map` would coerce the
        # enum members straight back to plain strings.
        return pd.Series(
            [MarketRegime(str(value)) for value in modal],
            index=modal.index,
            dtype="object",
            name="regime",
        )

    def posterior_at(
        self,
        features: pd.DataFrame,
        asof: dt.date | pd.Timestamp,
    ) -> tuple[RegimeProbabilities, pd.Timestamp]:
        """Validated posterior at the most recent date at or before ``asof``.

        Also returns the date of the observation that produced it, which is
        earlier than ``asof`` whenever the newest rows are not filterable.
        """
        posteriors = self.probabilities(features.loc[: pd.Timestamp(asof)])
        if posteriors.empty:
            msg = f"no filterable observation at or before {asof}"
            raise InsufficientDataError(msg)

        return probabilities_from_row(posteriors.iloc[-1]), posteriors.index[-1]

    def probabilities_at(
        self,
        features: pd.DataFrame,
        asof: dt.date | pd.Timestamp,
    ) -> RegimeProbabilities:
        """Validated posterior at the most recent date at or before ``asof``."""
        return self.posterior_at(features, asof)[0]

    def transition_matrix(self) -> pd.DataFrame:
        """Transition matrix with regime names, ordered by the risk ladder.

        Empty for models without state persistence.
        """
        if self.estimate.transition is None:
            return pd.DataFrame()

        ladder = [self.assignment.mapping[s.state] for s in self.assignment.statistics]
        order = [s.state for s in self.assignment.statistics]
        reordered = self.estimate.transition[np.ix_(order, order)]
        names = [regime.value for regime in ladder]
        return pd.DataFrame(reordered, index=names, columns=names)

    def expected_durations(self) -> pd.Series:
        """Expected regime duration in sessions, from the transition diagonal.

        For a persistent state with self-transition probability :math:`p`, the
        sojourn time is geometric with mean :math:`1 / (1 - p)`.
        """
        matrix = self.transition_matrix()
        if matrix.empty:
            return pd.Series(dtype="float64", name="expected_duration")

        persistence = pd.Series(np.diag(matrix.to_numpy()), index=matrix.index)
        durations = 1.0 / (1.0 - persistence).where(persistence < 1.0)
        return durations.rename("expected_duration")

    def to_bundle(self) -> dict[str, Any]:
        """Serialize the fitted model into a plain dict of arrays and metadata.

        The bundle keeps arrays as :class:`numpy.ndarray` objects so that callers
        can choose how to persist them (``np.savez_compressed``) while the
        metadata is JSON-serialisable.
        """
        arrays: dict[str, np.ndarray] = {
            "emission_means": self.estimate.emission.means,
            "emission_covariances": self.estimate.emission.covariances,
            "standardiser_centre": self.standardiser.centre,
            "standardiser_scale": self.standardiser.scale,
            "posteriors": self.estimate.posteriors,
        }
        if self.estimate.transition is not None:
            arrays["transition"] = self.estimate.transition

        metadata = {
            "classifier": self.classifier,
            "estimate": {
                "log_likelihood": self.estimate.log_likelihood,
                "converged": self.estimate.converged,
            },
            "standardiser": {
                "columns": list(self.standardiser.columns),
            },
            "assignment": {
                "mapping": {
                    str(state): regime.value for state, regime in self.assignment.mapping.items()
                },
                "method": self.assignment.method.value,
                "statistics": [
                    {
                        "state": stat.state,
                        "occupancy": stat.occupancy,
                        "mean_return": stat.mean_return,
                        "volatility": stat.volatility,
                        "n_observations": stat.n_observations,
                        "stress_score": stat.stress_score,
                    }
                    for stat in self.assignment.statistics
                ],
            },
            "metadata": self.metadata.model_dump(mode="json"),
        }

        return {"arrays": arrays, "metadata": metadata}

    @classmethod
    def from_bundle(cls, bundle: dict[str, Any]) -> Self:
        """Reconstruct a :class:`FittedRegime` from a bundle produced by :meth:`to_bundle`."""
        arrays = bundle["arrays"]
        meta = bundle["metadata"]

        emission = EmissionParameters(
            means=arrays["emission_means"],
            covariances=arrays["emission_covariances"],
        )
        transition = arrays.get("transition")
        posteriors = arrays.get("posteriors")
        if posteriors is None:
            # Reconstruct a placeholder if the bundle omitted training posteriors.
            n_states = emission.means.shape[0]
            posteriors = np.empty((0, n_states), dtype="float64")

        estimate = StateEstimate(
            posteriors=posteriors,
            emission=emission,
            transition=transition,
            log_likelihood=meta["estimate"]["log_likelihood"],
            converged=meta["estimate"]["converged"],
        )

        standardiser = CausalStandardiser(
            columns=tuple(meta["standardiser"]["columns"]),
            centre=arrays["standardiser_centre"],
            scale=arrays["standardiser_scale"],
        )

        statistics = tuple(
            StateStatistics(
                state=stat["state"],
                occupancy=stat["occupancy"],
                mean_return=stat["mean_return"],
                volatility=stat["volatility"],
                n_observations=stat["n_observations"],
                stress_score=stat.get("stress_score"),
            )
            for stat in meta["assignment"]["statistics"]
        )

        assignment = LabelAssignment(
            mapping={
                int(state): MarketRegime(regime)
                for state, regime in meta["assignment"]["mapping"].items()
            },
            statistics=statistics,
            method=RankingMethod(meta["assignment"]["method"]),
        )

        metadata = ModelMetadata.model_validate(meta["metadata"])

        return cls(
            classifier=meta["classifier"],
            estimate=estimate,
            assignment=assignment,
            standardiser=standardiser,
            metadata=metadata,
        )

    def save(self, path: Path | str) -> None:
        """Persist the fitted model to ``path`` as a directory.

        ``path`` may be a :class:`str` or a :class:`pathlib.Path`.  The bundle
        is written as a directory containing ``arrays.npz`` (all NumPy arrays,
        compressed) and ``metadata.json`` (JSON-serialisable provenance).  No
        particular file extension is required or enforced; callers may use
        ``.bundle``, ``.npz`` or any other convention.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        bundle = self.to_bundle()
        np.savez_compressed(path / "arrays.npz", **bundle["arrays"])
        (path / "metadata.json").write_text(
            json.dumps(bundle["metadata"], indent=2, sort_keys=True),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path | str) -> Self:
        """Load a fitted model from a directory previously written by :meth:`save`.

        ``path`` may be a :class:`str` or a :class:`pathlib.Path` and must point
        to a bundle directory containing ``arrays.npz`` and ``metadata.json``.
        """
        path = Path(path)
        arrays = np.load(path / "arrays.npz")
        metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
        return cls.from_bundle({"arrays": dict(arrays), "metadata": metadata})


class RegimeModel(ABC):
    """Base class for regime classifiers.

    Subclasses implement :meth:`_estimate` only. Standardisation, labelling,
    metadata capture and causal filtering are handled here so all classifiers
    share one audited implementation.

    Parameters
    ----------
    settings
        Model configuration section.
    """

    name: ClassVar[str]
    ranking_method: ClassVar[RankingMethod] = RankingMethod.SHARPE

    def __init__(self, settings: ModelSettings) -> None:
        self._settings = settings

    @property
    def settings(self) -> ModelSettings:
        """The model configuration."""
        return self._settings

    @abstractmethod
    def _estimate(self, matrix: np.ndarray, seed: int) -> StateEstimate:
        """Fit latent states on a standardised feature matrix.

        Parameters
        ----------
        matrix
            ``(n_observations, n_features)`` standardised, gap-free matrix.
        seed
            RNG seed for the estimator.

        Returns
        -------
        StateEstimate
            Raw estimator output.
        """

    def fit(
        self,
        features: pd.DataFrame,
        reference_returns: pd.Series,
        *,
        train_end: dt.date | pd.Timestamp | None = None,
        feature_names: Sequence[str] | None = None,
    ) -> FittedRegime:
        """Fit the classifier on data observable at or before ``train_end``.

        Parameters
        ----------
        features
            Full feature matrix. Truncated internally to ``train_end``.
        reference_returns
            Daily returns of the reference risk asset, used to attach economic
            meaning to each state for the labelling step.
        train_end
            Last observation admitted into training. ``None`` uses every row,
            which is appropriate only for a single as-of scan and never inside a
            walk-forward loop.
        feature_names
            Feature subset to use. Defaults to ``settings.features``.

        Returns
        -------
        FittedRegime
            Fitted and canonically-labelled model.

        Raises
        ------
        InsufficientDataError
            If the usable training sample is below
            ``settings.min_train_observations``.
        """
        columns = list(feature_names or self._settings.features)
        missing = [c for c in columns if c not in features.columns]
        if missing:
            msg = f"feature matrix is missing configured features: {missing}"
            raise KeyError(msg)

        window = features.loc[:, columns]
        if train_end is not None:
            window = window.loc[: pd.Timestamp(train_end)]

        # Rows with a partially-warm feature are dropped, not imputed.
        training = window.dropna(how="any")
        if len(training) < self._settings.min_train_observations:
            msg = (
                f"training window has {len(training)} usable observations, "
                f"below the configured floor of {self._settings.min_train_observations}. "
                "Extend data.start or lower model.min_train_observations."
            )
            raise InsufficientDataError(msg)

        standardiser = CausalStandardiser.fit(training)
        matrix = standardiser.transform(training)

        estimate = self._estimate(matrix, seed=self._settings.seed)

        posteriors = pd.DataFrame(
            estimate.posteriors,
            index=training.index,
            columns=list(range(estimate.n_states)),
        )
        statistics = summarise_states(reference_returns, posteriors)
        method = self._settings.ranking_method or self.ranking_method
        assignment = assign_labels(
            statistics,
            method=method,
            min_occupancy=self._settings.min_state_occupancy,
            emission_means=estimate.emission.means,
            feature_names=standardiser.columns,
            signs=self._settings.effective_stress_signs,
        )

        metadata = ModelMetadata(
            name=self.name,
            n_states=estimate.n_states,
            seed=self._settings.seed,
            train_start=training.index[0].date(),
            train_end=training.index[-1].date(),
            n_train_observations=len(training),
            converged=estimate.converged,
            log_likelihood=estimate.log_likelihood,
            feature_names=tuple(columns),
        )

        return FittedRegime(
            classifier=self.name,
            estimate=estimate,
            assignment=assignment,
            standardiser=standardiser,
            metadata=metadata,
        )
