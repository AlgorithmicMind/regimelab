"""Markov-switching regression: the Hamilton filter via ``statsmodels``.

Hamilton (1989), *A New Approach to the Economic Analysis of Nonstationary Time
Series and the Business Cycle*, Econometrica.

Why this model is here alongside the HMM
----------------------------------------
It is the econometric reference implementation, and it answers a question
``hmmlearn`` cannot.

``hmmlearn`` is a machine-learning library: it returns point estimates and no
inferential apparatus. ``statsmodels.MarkovRegression`` returns **standard
errors, t-statistics and a proper likelihood-ratio framework**. That makes it
possible to ask whether the regime separation is statistically distinguishable
from a single-state model, rather than merely asserting that four states were
requested and four states were returned. On a portfolio piece being read by
someone who will interrogate the statistics, that distinction matters.

Its structural limitation is that it is univariate in the dependent variable.
The multivariate feature vector is handled by regressing the reference return on
the remaining features with regime-dependent coefficients and variance, which is
a different -- and in some ways sharper -- specification than the HMM's joint
density over all features:

- The **HMM** asks: which latent state best explains the joint behaviour of all
  features?
- The **Markov-switching regression** asks: which latent state best explains how
  returns *respond* to the features?

The second is closer to what an allocator actually wants to know, at the cost of
requiring the choice of a dependent variable.

Both are exposed rather than one being declared correct. Agreement between them
is evidence; disagreement is a signal to investigate rather than to trade.
"""

from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING, ClassVar, cast

import numpy as np

from regimelab.models.base import (
    EmissionParameters,
    InsufficientDataError,
    RegimeModel,
    StateEstimate,
)
from regimelab.models.labeling import RankingMethod

if TYPE_CHECKING:
    from regimelab.settings import ModelSettings

__all__ = ["MarkovSwitchingRegimeModel"]

_LOGGER = logging.getLogger(__name__)


class MarkovSwitchingRegimeModel(RegimeModel):
    """Markov-switching regression classifier backed by ``statsmodels``.

    Parameters
    ----------
    settings
        Model configuration section.
    dependent_index
        Column position within ``settings.features`` to treat as the dependent
        variable. Defaults to 0, which is ``trailing_return`` under the shipped
        configuration.
    switching_variance
        Whether the residual variance is regime-dependent. Should stay ``True``:
        the defining property of a stress regime is its variance, and a
        homoskedastic specification cannot represent it.
    """

    name: ClassVar[str] = "hamilton"
    ranking_method: ClassVar[RankingMethod] = RankingMethod.SHARPE

    def __init__(
        self,
        settings: ModelSettings,
        *,
        dependent_index: int = 0,
        switching_variance: bool = True,
    ) -> None:
        super().__init__(settings)
        self._dependent_index = dependent_index
        self._switching_variance = switching_variance

    def _estimate(self, matrix: np.ndarray, seed: int) -> StateEstimate:
        """Fit the Markov-switching regression and filter causally.

        ``statsmodels`` exposes ``filtered_marginal_probabilities`` directly --
        genuinely filtered, not smoothed -- so those are used for the training
        posteriors. The companion ``smoothed_marginal_probabilities`` are
        deliberately never touched anywhere in this codebase.
        """
        from statsmodels.tsa.regime_switching.markov_regression import (  # noqa: PLC0415
            MarkovRegression,
        )

        n_states = self._settings.n_states
        n_obs, n_features = matrix.shape

        if not 0 <= self._dependent_index < n_features:
            msg = (
                f"dependent_index {self._dependent_index} is out of range for {n_features} features"
            )
            raise ValueError(msg)

        endog = matrix[:, self._dependent_index]
        exog_columns = [i for i in range(n_features) if i != self._dependent_index]
        exog = matrix[:, exog_columns] if exog_columns else None

        if n_obs <= n_states * (len(exog_columns) + 2):
            msg = (
                f"{n_obs} observations cannot identify {n_states} states with "
                f"{len(exog_columns)} regressors"
            )
            raise InsufficientDataError(msg)

        model = MarkovRegression(
            endog=endog,
            k_regimes=n_states,
            exog=exog,
            trend="c",
            switching_variance=self._switching_variance,
        )

        with warnings.catch_warnings():
            # statsmodels emits convergence and Hessian warnings that are
            # informative but not fatal; escalating them would make the fit
            # fail under the strict pytest filter.
            warnings.simplefilter("ignore")
            try:
                np.random.seed(seed % (2**31 - 1))  # noqa: NPY002 - statsmodels uses legacy RNG
                result = model.fit(
                    maxiter=self._settings.max_iter,
                    disp=False,
                    em_iter=max(20, self._settings.n_init),
                    search_reps=self._settings.n_init,
                )
            except (ValueError, np.linalg.LinAlgError, RuntimeError) as exc:
                msg = f"Markov-switching fit failed: {exc}"
                raise InsufficientDataError(msg) from exc

        return self._package(result, matrix, n_states)

    def _package(
        self,
        result: object,
        matrix: np.ndarray,
        n_states: int,
    ) -> StateEstimate:
        """Convert a ``statsmodels`` result into a :class:`StateEstimate`.

        The regression is a conditional model of one variable, so it has no joint
        density over the feature vector. To let the shared filtering path in
        :class:`~regimelab.models.base.FittedRegime` operate on new data, a
        Gaussian emission surrogate is fitted post hoc: the posterior-weighted
        mean and covariance of the *feature vector* under each state.

        This surrogate is an approximation, and its purpose should be understood
        precisely. It is used only to extend filtering to out-of-sample dates,
        where the alternative is refitting the regression at every step -- which
        is both prohibitively slow and, if done naively, look-ahead
        contaminated. The training-window posteriors below come from the exact
        Hamilton filter, not from the surrogate.
        """
        filtered = np.asarray(
            result.filtered_marginal_probabilities,  # type: ignore[attr-defined]
            dtype="float64",
        )
        if filtered.ndim == 1:  # pragma: no cover - defensive
            filtered = filtered.reshape(-1, 1)
        if filtered.shape[0] != matrix.shape[0]:
            # statsmodels drops initial observations for some specifications;
            # pad at the front with the earliest available belief.
            deficit = matrix.shape[0] - filtered.shape[0]
            filtered = np.vstack([np.repeat(filtered[:1], deficit, axis=0), filtered])

        transition = self._transition_from_result(result, n_states)
        emission = self._surrogate_emission(matrix, filtered)

        log_likelihood = float(getattr(result, "llf", np.nan))

        retvals = getattr(result, "mle_retvals", None)
        converged = bool(retvals.get("converged", True)) if isinstance(retvals, dict) else True

        return StateEstimate(
            posteriors=filtered,
            emission=emission,
            transition=transition,
            log_likelihood=log_likelihood,
            converged=converged,
        )

    @staticmethod
    def _transition_from_result(result: object, n_states: int) -> np.ndarray:
        """Extract the row-stochastic transition matrix.

        ``statsmodels`` returns ``regime_transition`` shaped
        ``(k_regimes, k_regimes, nobs)`` with *column*-stochastic orientation and
        a time axis for time-varying specifications. Ours is time-invariant, so
        the first slice is taken and transposed into the row-stochastic
        convention the rest of this codebase uses.
        """
        raw = np.asarray(result.regime_transition, dtype="float64")  # type: ignore[attr-defined]

        matrix = raw[:, :, 0] if raw.ndim == 3 else raw
        matrix = matrix.T

        if matrix.shape != (n_states, n_states):  # pragma: no cover - defensive
            _LOGGER.warning(
                "unexpected transition shape %s; falling back to a persistent prior",
                matrix.shape,
            )
            matrix = np.full((n_states, n_states), 0.05 / max(n_states - 1, 1))
            np.fill_diagonal(matrix, 0.95)

        matrix = np.clip(matrix, 1e-12, None)
        return cast(np.ndarray, matrix / matrix.sum(axis=1, keepdims=True))

    @staticmethod
    def _surrogate_emission(
        matrix: np.ndarray,
        posteriors: np.ndarray,
    ) -> EmissionParameters:
        """Posterior-weighted Gaussian moments of the feature vector per state."""
        n_states = posteriors.shape[1]
        n_features = matrix.shape[1]

        means = np.zeros((n_states, n_features), dtype="float64")
        covariances = np.zeros((n_states, n_features, n_features), dtype="float64")

        for state in range(n_states):
            weights = posteriors[:, state]
            total = float(weights.sum())

            if total <= 0.0:  # pragma: no cover - defensive
                means[state] = matrix.mean(axis=0)
                covariances[state] = np.cov(matrix, rowvar=False) + np.eye(n_features) * 1e-6
                continue

            mean = np.average(matrix, axis=0, weights=weights)
            centred = matrix - mean
            covariance = (centred * weights[:, None]).T @ centred / total

            means[state] = mean
            # Ridge keeps the matrix invertible when a state is thinly populated.
            covariances[state] = covariance + np.eye(n_features) * 1e-6

        return EmissionParameters(means=means, covariances=covariances)
