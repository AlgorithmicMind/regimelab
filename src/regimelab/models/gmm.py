"""Gaussian mixture model for volatility clustering.

A deliberately memoryless counterpart to the HMM. Each observation is assigned
independently:

.. math::

    x_t \\sim \\sum_{j=1}^{K} \\pi_j \\, \\mathcal{N}(\\mu_j, \\Sigma_j)

Why keep a model that is known to be worse
------------------------------------------
It is a control, not a competitor. With no transition matrix, the GMM assigns
each day on that day's evidence alone. Comparing it to the HMM therefore
isolates how much of the HMM's behaviour comes from the *Markov structure*
rather than from the feature set:

- If the two produce similar regime paths, the features are separating the
  states strongly and the persistence prior is doing little work.
- If the GMM path is far noisier -- the usual outcome -- the difference quantifies
  what persistence is worth, and the resulting turnover gap puts a number on it.

That comparison is far more informative than asserting the HMM is better, and it
costs almost nothing given the shared base class.

The GMM is also the more defensible choice for pure volatility-state
classification, where the objective is to bucket observations by conditional
variance rather than to track a persistent latent process. For that reason it
ranks states by ascending volatility rather than by Sharpe ratio.
"""

from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING, ClassVar

import numpy as np

from regimelab.models.base import (
    EmissionParameters,
    InsufficientDataError,
    RegimeModel,
    StateEstimate,
)
from regimelab.models.types import DEFAULT_RANKING_METHODS, RankingMethod

if TYPE_CHECKING:
    from regimelab.settings import ModelSettings

__all__ = ["GaussianMixtureRegimeModel"]

_LOGGER = logging.getLogger(__name__)

_COVARIANCE_FLOOR = 1e-6


class GaussianMixtureRegimeModel(RegimeModel):
    """Gaussian mixture regime classifier backed by ``scikit-learn``.

    Parameters
    ----------
    settings
        Model configuration section.
    covariance_type
        ``scikit-learn`` covariance parameterisation.
    """

    name: ClassVar[str] = "gmm"
    ranking_method: ClassVar[RankingMethod] = DEFAULT_RANKING_METHODS[name]

    def __init__(
        self,
        settings: ModelSettings,
    ) -> None:
        super().__init__(settings)
        self._covariance_type: str = settings.covariance_type

    def _estimate(self, matrix: np.ndarray, seed: int) -> StateEstimate:
        """Fit the mixture and return per-observation posteriors.

        ``transition`` is ``None``, which is what tells
        :class:`~regimelab.models.base.FittedRegime` to use the memoryless
        posterior instead of the forward filter. The mixture posterior is
        inherently causal: it depends only on :math:`x_t`.
        """
        from sklearn.mixture import GaussianMixture  # noqa: PLC0415

        n_states = self._settings.n_states
        if matrix.shape[0] <= n_states:
            msg = f"cannot fit {n_states} components to {matrix.shape[0]} observations"
            raise InsufficientDataError(msg)

        model = GaussianMixture(
            n_components=n_states,
            covariance_type=self._covariance_type,
            max_iter=self._settings.max_iter,
            tol=self._settings.tolerance,
            n_init=self._settings.n_init,
            init_params="k-means++",
            random_state=seed,
            reg_covar=_COVARIANCE_FLOOR,
            verbose=self._settings.verbose,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                model.fit(matrix)
            except (ValueError, np.linalg.LinAlgError) as exc:
                msg = f"Gaussian mixture fit failed: {exc}"
                raise InsufficientDataError(msg) from exc

        posteriors = np.asarray(model.predict_proba(matrix), dtype="float64")
        emission = EmissionParameters(
            means=np.asarray(model.means_, dtype="float64"),
            covariances=self._full_covariances(model, matrix.shape[1]),
        )

        log_likelihood = float(model.score(matrix) * matrix.shape[0])
        converged = bool(model.converged_)
        if not converged:
            _LOGGER.warning(
                "Gaussian mixture did not converge in %d iterations",
                self._settings.max_iter,
            )

        return StateEstimate(
            posteriors=posteriors,
            emission=emission,
            transition=None,
            log_likelihood=log_likelihood,
            converged=converged,
        )

    def _full_covariances(self, model: object, n_features: int) -> np.ndarray:
        """Expand a ``scikit-learn`` covariance parameterisation to full matrices."""
        covariances = np.asarray(model.covariances_, dtype="float64")  # type: ignore[attr-defined]
        n_states = self._settings.n_states

        if self._covariance_type == "full":
            return covariances
        if self._covariance_type == "diag":
            return np.stack([np.diag(row) for row in covariances])
        if self._covariance_type == "spherical":
            return np.stack([np.eye(n_features) * value for value in covariances])
        if self._covariance_type == "tied":
            return np.stack([covariances for _ in range(n_states)])

        msg = f"unsupported covariance_type {self._covariance_type!r}"
        raise ValueError(msg)
