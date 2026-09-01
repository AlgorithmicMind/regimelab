"""Hamilton forward filter and Gaussian emission densities.

Why this module exists
----------------------
``hmmlearn.predict_proba`` runs forward-*backward*, which returns **smoothed**
posteriors :math:`P(s_t \\mid x_1, \\dots, x_T)`. Those condition on the entire
sample, including observations after :math:`t`. Charting them produces the
beautifully crisp regime plots common in the literature; trading them is
look-ahead bias.

What a decision at time :math:`t` requires is the **filtered** posterior
:math:`P(s_t \\mid x_1, \\dots, x_t)`.

One could recover it by calling ``predict_proba(X[:t])`` for every :math:`t` and
keeping the last row -- correct, because the backward pass is unity at the final
index, but :math:`O(T^2)` and unusable over a multi-decade walk-forward. The
forward recursion below returns the filtered posterior for *every* :math:`t` in a
single :math:`O(T K^2)` pass.

This is Hamilton's (1989) filter. Implementing it explicitly rather than
borrowing a library's private internals also means the HMM and Markov-switching
paths share one audited code path, so their outputs are directly comparable.

Numerical approach
------------------
The recursion runs entirely in log space with :func:`scipy.special.logsumexp`.
The linear-space form underflows to zero within a few hundred observations for
realistic emission densities, and the usual rescaling trick is both harder to
reason about and less accurate than working in logs throughout.
"""

from __future__ import annotations

from typing import NamedTuple, cast

import numpy as np
from scipy.linalg import solve_triangular
from scipy.special import logsumexp

__all__ = [
    "FilterResult",
    "forward_filter",
    "gaussian_log_density",
    "stationary_distribution",
]

_LOG_2PI = float(np.log(2.0 * np.pi))
# Ridge added to a covariance matrix before inversion when it is near-singular.
_JITTER = 1e-10


class FilterResult(NamedTuple):
    """Output of the forward filter.

    Attributes
    ----------
    filtered
        ``(n_observations, n_states)`` array of filtered posteriors
        :math:`P(s_t \\mid x_{1:t})`. Each row sums to one.
    predicted
        ``(n_observations, n_states)`` array of one-step-ahead predictions
        :math:`P(s_t \\mid x_{1:t-1})`, i.e. the posterior *before* observing
        :math:`x_t`. This is the honest input for a next-session allocation
        decision.
    log_likelihood
        Total log likelihood :math:`\\log P(x_{1:T})`, accumulated from the
        one-step predictive densities.
    """

    filtered: np.ndarray
    predicted: np.ndarray
    log_likelihood: float


def gaussian_log_density(
    observations: np.ndarray,
    means: np.ndarray,
    covariances: np.ndarray,
) -> np.ndarray:
    """Log density of multivariate normal emissions for every state.

    Parameters
    ----------
    observations
        ``(n_observations, n_features)`` array. Must be finite.
    means
        ``(n_states, n_features)`` array of state means.
    covariances
        ``(n_states, n_features, n_features)`` array of full covariance matrices.

    Returns
    -------
    numpy.ndarray
        ``(n_observations, n_states)`` array of log densities.

    Notes
    -----
    Uses a Cholesky factorisation rather than an explicit inverse: it is faster,
    numerically better behaved, and yields the log determinant for free. A ridge
    is added and the factorisation retried when a state's covariance is
    numerically singular, which happens when EM collapses a state onto very few
    observations.
    """
    observations = np.asarray(observations, dtype="float64")
    means = np.asarray(means, dtype="float64")
    covariances = np.asarray(covariances, dtype="float64")

    n_obs, n_features = observations.shape
    n_states = means.shape[0]
    result = np.empty((n_obs, n_states), dtype="float64")

    for state in range(n_states):
        covariance = covariances[state]
        try:
            cholesky = np.linalg.cholesky(covariance)
        except np.linalg.LinAlgError:
            ridge = _JITTER * float(np.trace(covariance)) / max(n_features, 1)
            cholesky = np.linalg.cholesky(covariance + np.eye(n_features) * max(ridge, _JITTER))

        deviation = observations - means[state]
        # Solve L z = deviation^T so that z^T z is the Mahalanobis distance.
        solved = solve_triangular(cholesky, deviation.T, lower=True)
        mahalanobis = np.einsum("ij,ij->j", solved, solved)
        log_det = 2.0 * float(np.sum(np.log(np.diag(cholesky))))
        result[:, state] = -0.5 * (mahalanobis + log_det + n_features * _LOG_2PI)

    return result


def stationary_distribution(transition: np.ndarray) -> np.ndarray:
    """Stationary distribution of a row-stochastic transition matrix.

    Used as the filter's initial condition when no explicit prior is supplied.
    The stationary distribution is the right choice: it is the unconditional
    regime probability implied by the fitted dynamics, so the filter starts from
    the model's own long-run belief rather than an arbitrary uniform guess.

    Parameters
    ----------
    transition
        ``(n_states, n_states)`` row-stochastic matrix.

    Returns
    -------
    numpy.ndarray
        Length ``n_states`` probability vector.

    Notes
    -----
    Falls back to the uniform distribution when the matrix is reducible and no
    valid stationary vector can be extracted, rather than raising: a reducible
    transition matrix is a fitting pathology that the occupancy check in
    :mod:`regimelab.models.labeling` is responsible for rejecting, and the filter
    should still return something well-formed for the diagnostics to inspect.
    """
    transition = np.asarray(transition, dtype="float64")
    n_states = transition.shape[0]

    eigenvalues, eigenvectors = np.linalg.eig(transition.T)
    index = int(np.argmin(np.abs(eigenvalues - 1.0)))
    vector = np.real(eigenvectors[:, index])

    total = vector.sum()
    if not np.isfinite(total) or abs(total) < 1e-12:
        return np.full(n_states, 1.0 / n_states)

    distribution = vector / total
    if np.any(distribution < -1e-8):
        return np.full(n_states, 1.0 / n_states)

    distribution = cast(np.ndarray, np.clip(distribution, 0.0, None))
    return cast(np.ndarray, distribution / distribution.sum())


def forward_filter(
    log_emissions: np.ndarray,
    transition: np.ndarray,
    initial: np.ndarray | None = None,
) -> FilterResult:
    """Run Hamilton's forward filter in log space.

    Recursion, for each :math:`t`:

    .. math::

        \\xi_{t|t-1} &= P^\\top \\xi_{t-1|t-1} \\\\
        \\xi_{t|t}    &\\propto \\xi_{t|t-1} \\odot f(x_t \\mid s_t)

    Parameters
    ----------
    log_emissions
        ``(n_observations, n_states)`` log emission densities, e.g. from
        :func:`gaussian_log_density`.
    transition
        ``(n_states, n_states)`` row-stochastic matrix where entry ``(i, j)`` is
        :math:`P(s_t = j \\mid s_{t-1} = i)`.
    initial
        Prior over states at the first observation. Defaults to the stationary
        distribution of ``transition``.

    Returns
    -------
    FilterResult
        Filtered posteriors, one-step-ahead predictions and the log likelihood.

    Raises
    ------
    ValueError
        If the shapes are inconsistent or the inputs are not finite.
    """
    log_emissions = np.asarray(log_emissions, dtype="float64")
    transition = np.asarray(transition, dtype="float64")

    if log_emissions.ndim != 2:
        msg = f"log_emissions must be 2-dimensional, got shape {log_emissions.shape}"
        raise ValueError(msg)

    n_obs, n_states = log_emissions.shape
    if transition.shape != (n_states, n_states):
        msg = f"transition matrix shape {transition.shape} is incompatible with {n_states} states"
        raise ValueError(msg)

    if n_obs == 0:
        empty = np.empty((0, n_states), dtype="float64")
        return FilterResult(filtered=empty, predicted=empty.copy(), log_likelihood=0.0)

    prior = (
        stationary_distribution(transition)
        if initial is None
        else np.asarray(
            initial,
            dtype="float64",
        )
    )
    if prior.shape != (n_states,):
        msg = f"initial distribution must have shape ({n_states},), got {prior.shape}"
        raise ValueError(msg)

    with np.errstate(divide="ignore"):
        log_transition = np.log(np.clip(transition, 1e-300, None))
        log_prior = np.log(np.clip(prior / prior.sum(), 1e-300, None))

    filtered = np.empty((n_obs, n_states), dtype="float64")
    predicted = np.empty((n_obs, n_states), dtype="float64")
    log_likelihood = 0.0

    log_predicted = log_prior
    for t in range(n_obs):
        predicted[t] = np.exp(log_predicted - logsumexp(log_predicted))

        # Bayes update: prediction times likelihood, normalised.
        log_joint = log_predicted + log_emissions[t]
        log_marginal = float(logsumexp(log_joint))

        if not np.isfinite(log_marginal):
            # Every state assigns effectively zero density to this observation
            # (an extreme outlier). Carrying the prediction forward unchanged is
            # the conservative choice; propagating NaN would destroy the rest of
            # the path for one bad tick.
            filtered[t] = predicted[t]
        else:
            log_filtered = log_joint - log_marginal
            filtered[t] = np.exp(log_filtered)
            log_likelihood += log_marginal
            log_predicted = log_filtered

        if t + 1 < n_obs:
            # Propagate one step: log-sum-exp over the previous state.
            log_predicted = logsumexp(
                log_predicted[:, None] + log_transition,
                axis=0,
            )

    # Guard against accumulated drift so downstream schema validation, which
    # requires probabilities summing to one, never fails on rounding alone.
    filtered /= filtered.sum(axis=1, keepdims=True)
    predicted /= predicted.sum(axis=1, keepdims=True)

    return FilterResult(
        filtered=filtered,
        predicted=predicted,
        log_likelihood=log_likelihood,
    )
