"""Multivariate Gaussian hidden Markov model.

The primary classifier. A market regime is modelled as a latent discrete state
:math:`s_t` following a first-order Markov chain, with the observed feature
vector drawn from a state-conditional multivariate normal:

.. math::

    P(s_t = j \\mid s_{t-1} = i) = P_{ij}, \\qquad
    x_t \\mid s_t = j \\sim \\mathcal{N}(\\mu_j, \\Sigma_j)

Why an HMM rather than clustering
---------------------------------
The transition matrix is the point. A Gaussian mixture would assign each day
independently, so a single anomalous session flips the regime and back again;
turnover becomes unusable. The Markov structure imposes persistence, so a state
change requires sustained evidence. It also yields the expected sojourn time
:math:`1/(1-P_{jj})`, which is directly interpretable and a useful sanity check:
a fit implying a two-day average regime has not found regimes.

Practical caveats, honestly stated
----------------------------------
- **Gaussian emissions understate tails.** Financial features are leptokurtic. A
  Student-t emission would fit better; the cost is a heavier estimation problem
  and a non-standard implementation. The mitigation used here is to let the
  high-volatility state absorb the tail, which is why ``full`` covariances are
  the default -- a diagonal covariance cannot represent the correlation
  restructuring that defines a stress regime.
- **EM finds local optima.** ``n_init`` restarts are run from different seeds and
  the best training likelihood wins. Without this, results move with the seed.
- **First-order Markov is a simplification.** Real regime durations are not
  geometric. The model will systematically under-predict very long calm periods.

Determinism
-----------
Restart seeds derive deterministically from the configured master seed, so a
given ``(seed, data)`` pair always produces the same fit -- required for the
manifest guarantee in :mod:`regimelab.store`.
"""

from __future__ import annotations

import contextlib
import io
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
from regimelab.models.filters import forward_filter
from regimelab.models.types import DEFAULT_RANKING_METHODS, RankingMethod

if TYPE_CHECKING:
    from regimelab.settings import ModelSettings

__all__ = ["HiddenMarkovRegimeModel"]

_LOGGER = logging.getLogger(__name__)

# Regularisation added to the diagonal of each state covariance. Without it, EM
# on standardised financial features occasionally collapses a state onto a
# near-singular covariance and the likelihood diverges.
_COVARIANCE_FLOOR = 1e-3


class HiddenMarkovRegimeModel(RegimeModel):
    """Gaussian HMM regime classifier backed by ``hmmlearn``.

    Parameters
    ----------
    settings
        Model configuration section.
    covariance_type
        ``hmmlearn`` covariance parameterisation. ``"full"`` by default: a
        stress regime is characterised as much by its correlation structure as
        by its variances, and ``"diag"`` cannot express that.
    """

    name: ClassVar[str] = "hmm"
    ranking_method: ClassVar[RankingMethod] = DEFAULT_RANKING_METHODS[name]

    def __init__(
        self,
        settings: ModelSettings,
    ) -> None:
        super().__init__(settings)
        self._covariance_type = settings.covariance_type

    def _estimate(self, matrix: np.ndarray, seed: int) -> StateEstimate:
        """Fit the HMM with multiple restarts, keeping the best likelihood."""
        from hmmlearn.hmm import GaussianHMM  # noqa: PLC0415

        # hmmlearn logs non-convergence as a warning even when verbose is False.
        # We detect convergence ourselves from the monitor, so suppress the noise.
        logging.getLogger("hmmlearn").setLevel(logging.ERROR)

        n_states = self._settings.n_states
        if matrix.shape[0] <= n_states:
            msg = f"cannot fit {n_states} states to {matrix.shape[0]} observations"
            raise InsufficientDataError(msg)

        best_model: object | None = None
        best_score = -np.inf
        best_converged = False

        for restart in range(self._settings.n_init):
            # Deterministic per-restart seed derived from the master seed.
            restart_seed = (seed + 7919 * restart) % (2**31 - 1)

            model = GaussianHMM(
                n_components=n_states,
                covariance_type=self._covariance_type,
                n_iter=self._settings.max_iter,
                tol=self._settings.tolerance,
                random_state=restart_seed,
                min_covar=_COVARIANCE_FLOOR,
                init_params="stmc",
                params="stmc",
                verbose=self._settings.verbose,
            )

            with warnings.catch_warnings():
                # hmmlearn warns on non-convergence; we handle that explicitly
                # below and do not want it escalated by the pytest filter.
                warnings.simplefilter("ignore")
                try:
                    # hmmlearn writes non-convergence messages to stdout even when
                    # verbose is False. Capturing them keeps the CLI and TUI clean.
                    with contextlib.redirect_stdout(io.StringIO()):
                        model.fit(matrix)
                    score = float(model.score(matrix))
                except (ValueError, np.linalg.LinAlgError) as exc:
                    _LOGGER.debug("HMM restart %d failed: %s", restart, exc)
                    continue

            if not np.isfinite(score):
                continue

            monitor_converged = bool(getattr(model.monitor_, "converged", False))

            if score > best_score:
                best_score = score
                best_model = model
                best_converged = monitor_converged

        if best_model is None:
            msg = (
                f"every one of {self._settings.n_init} HMM restarts failed to fit "
                f"{n_states} states; the feature matrix may be degenerate"
            )
            raise InsufficientDataError(msg)

        return self._estimate_from_model(best_model, matrix, best_converged)

    def _estimate_from_model(
        self,
        model: object,
        matrix: np.ndarray,
        converged: bool,
    ) -> StateEstimate:
        """Package a fitted ``hmmlearn`` model into a :class:`StateEstimate`.

        The filtered posteriors are recomputed with our own forward filter rather
        than taken from ``model.predict_proba``, which is forward-*backward* and
        therefore smoothed. See :mod:`regimelab.models.filters`.
        """
        means = np.asarray(model.means_, dtype="float64")  # type: ignore[attr-defined]
        covariances = self._full_covariances(model, means.shape[1])
        transition = np.asarray(model.transmat_, dtype="float64")  # type: ignore[attr-defined]

        emission = EmissionParameters(means=means, covariances=covariances)
        log_emissions = emission.log_density(matrix)

        startprob = np.asarray(model.startprob_, dtype="float64")  # type: ignore[attr-defined]
        result = forward_filter(log_emissions, transition, initial=startprob)

        return StateEstimate(
            posteriors=result.filtered,
            emission=emission,
            transition=transition,
            log_likelihood=result.log_likelihood,
            converged=converged,
        )

    def _full_covariances(self, model: object, n_features: int) -> np.ndarray:
        """Expand any ``hmmlearn`` covariance parameterisation to full matrices.

        Normalising the shape here means the filter has exactly one code path,
        instead of branching on ``covariance_type`` at every use site.
        """
        covariances = np.asarray(model.covars_, dtype="float64")  # type: ignore[attr-defined]

        if covariances.ndim == 3:
            return covariances
        if covariances.ndim == 2:  # 'diag': (n_states, n_features)
            return np.stack([np.diag(row) for row in covariances])
        if covariances.ndim == 1:  # 'spherical': (n_states,)
            return np.stack([np.eye(n_features) * value for value in covariances])

        msg = f"unsupported covariance array with shape {covariances.shape}"
        raise ValueError(msg)
