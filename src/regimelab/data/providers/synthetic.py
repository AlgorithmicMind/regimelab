"""Seeded synthetic market generator.

Purpose is not simulation realism. It is to provide a *ground-truth* dataset:
because the latent regime path is known by construction, the test suite can
assert that the classifiers recover it, which is impossible with real data where
no label exists.

It also makes the whole engine runnable and CI green with no network access and
no vendor credentials.

Generative model
----------------
A three-state Markov chain drives a single market factor. Each state sets the
factor's drift, its volatility, and -- the point of the exercise -- the
cross-sectional *coupling* that determines how much of each asset's variance the
factor explains. Coupling rises sharply in the stress state, so the absorption
ratio and mean correlation respond the way they do in real crises, and the
features under test actually have signal to find.
"""

from __future__ import annotations

import datetime as dt
import hashlib
from typing import TYPE_CHECKING, Final, cast

import numpy as np
import pandas as pd

from regimelab.data.providers.base import (
    OHLCV_FIELDS,
    ProviderError,
    panel_from_symbol_frames,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = ["SyntheticProvider"]

# State 0 = calm bull, 1 = choppy transition, 2 = stress.
_TRANSITION: Final[np.ndarray] = np.array(
    [
        [0.985, 0.013, 0.002],
        [0.040, 0.930, 0.030],
        [0.015, 0.070, 0.915],
    ],
)

# Annualised drift and volatility of the market factor, per state.
_DRIFT: Final[np.ndarray] = np.array([0.12, 0.00, -0.28])
_VOLATILITY: Final[np.ndarray] = np.array([0.11, 0.17, 0.42])

# Share of each asset's variance explained by the factor, per state.
_COUPLING: Final[np.ndarray] = np.array([0.35, 0.55, 0.85])

_TRADING_DAYS: Final[int] = 252

# Assets with a structurally different relationship to the equity factor.
_DEFENSIVE_BETA: Final[dict[str, float]] = {
    "TLT": -0.35,
    "IEF": -0.25,
    "GLD": 0.05,
    "SHY": -0.05,
}


class SyntheticProvider:
    """Deterministic synthetic OHLCV generator.

    Parameters
    ----------
    seed
        Master seed. Identical seeds and identical requests yield bit-identical
        panels.
    """

    def __init__(self, seed: int = 20260831) -> None:
        self._seed = seed

    @property
    def name(self) -> str:
        """Provider identifier, seed included so cache keys stay distinct."""
        return f"synthetic-{self._seed}"

    @property
    def is_point_in_time(self) -> bool:
        """Trivially true: generated data is never restated."""
        return True

    def fetch(
        self,
        symbols: Sequence[str],
        start: dt.date,
        end: dt.date | None = None,
    ) -> pd.DataFrame:
        """Generate a panel for ``symbols`` over ``[start, end]``."""
        symbols = list(symbols)
        if not symbols:
            msg = "no symbols requested"
            raise ProviderError(msg)

        end = end or dt.date.today()  # noqa: DTZ011
        calendar = pd.bdate_range(start=start, end=end, name="date")
        if len(calendar) < 2:
            msg = f"date range [{start}, {end}] contains fewer than 2 business days"
            raise ProviderError(msg)

        states = self._simulate_states(len(calendar))
        factor = self._simulate_factor(states)

        frames: dict[str, pd.DataFrame] = {}
        for symbol in symbols:
            if symbol.startswith("^"):
                frames[symbol] = self._volatility_index(symbol, states, calendar)
            else:
                frames[symbol] = self._price_series(symbol, states, factor, calendar)

        return panel_from_symbol_frames(frames)

    def latent_states(
        self,
        start: dt.date,
        end: dt.date | None = None,
    ) -> pd.Series[int]:
        """Return the ground-truth latent state path.

        Exposed for testing only. Production code must never consume this: it is
        exactly the information a real engine does not have.
        """
        end = end or dt.date.today()  # noqa: DTZ011
        calendar = pd.bdate_range(start=start, end=end, name="date")
        states = self._simulate_states(len(calendar))
        return pd.Series(states, index=calendar, name="state")

    def _rng(self, *tokens: str) -> np.random.Generator:
        """Derive an independent, reproducible generator from a token path.

        Hashing the tokens rather than incrementing a counter means each symbol's
        stream is stable regardless of how many other symbols were requested, so
        ``fetch(["SPY"])`` and ``fetch(["GLD", "SPY"])`` agree on SPY.
        """
        digest = hashlib.sha256("::".join(("regimelab", str(self._seed), *tokens)).encode())
        return np.random.default_rng(np.frombuffer(digest.digest(), dtype="uint32"))

    def _simulate_states(self, n: int) -> np.ndarray:
        """Sample the hidden Markov chain path."""
        rng = self._rng("states")
        states = np.empty(n, dtype="int64")
        states[0] = 0
        draws = rng.random(n)
        for t in range(1, n):
            cumulative = np.cumsum(_TRANSITION[states[t - 1]])
            states[t] = int(np.searchsorted(cumulative, draws[t]))
        return states

    def _simulate_factor(self, states: np.ndarray) -> np.ndarray:
        """Sample daily market-factor log returns given the state path."""
        rng = self._rng("factor")
        daily_drift = _DRIFT[states] / _TRADING_DAYS
        daily_vol = _VOLATILITY[states] / np.sqrt(_TRADING_DAYS)
        return cast(np.ndarray, daily_drift + daily_vol * rng.standard_normal(states.size))

    def _beta(self, symbol: str) -> float:
        """Deterministic factor loading for a symbol."""
        if symbol in _DEFENSIVE_BETA:
            return _DEFENSIVE_BETA[symbol]
        # Stable pseudo-random beta in [0.75, 1.35] derived from the ticker.
        rng = self._rng("beta", symbol)
        return float(0.75 + 0.60 * rng.random())

    def _price_series(
        self,
        symbol: str,
        states: np.ndarray,
        factor: np.ndarray,
        calendar: pd.DatetimeIndex,
    ) -> pd.DataFrame:
        """Build an OHLCV frame for a tradable symbol."""
        rng = self._rng("asset", symbol)
        beta = self._beta(symbol)
        coupling = np.sqrt(_COUPLING[states])

        idio_vol = _VOLATILITY[states] / np.sqrt(_TRADING_DAYS)
        idio = np.sqrt(1.0 - _COUPLING[states]) * idio_vol * rng.standard_normal(states.size)
        returns = beta * coupling * factor + idio

        base_price = float(30.0 + 170.0 * self._rng("level", symbol).random())
        close = base_price * np.exp(np.cumsum(returns))

        # Intraday range scaled to the day's conditional volatility, so the
        # Garman-Klass and Parkinson estimators receive a coherent high/low.
        day_vol = _VOLATILITY[states] / np.sqrt(_TRADING_DAYS)
        span = np.abs(rng.standard_normal(states.size)) * day_vol * close
        gap = rng.standard_normal(states.size) * day_vol * close * 0.3

        open_ = np.empty_like(close)
        open_[0] = base_price
        open_[1:] = close[:-1] + gap[1:]

        high = np.maximum(close, open_) + 0.5 * span
        low = np.minimum(close, open_) - 0.5 * span
        low = np.maximum(low, 0.01)

        volume = np.round(
            1e6
            * (1.0 + 2.0 * (_VOLATILITY[states] / _VOLATILITY[0]))
            * (1.0 + 0.2 * rng.standard_normal(states.size)).clip(0.3, None),
        )

        return pd.DataFrame(
            {
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
            },
            index=calendar,
            columns=list(OHLCV_FIELDS),
        )

    def _volatility_index(
        self,
        symbol: str,
        states: np.ndarray,
        calendar: pd.DatetimeIndex,
    ) -> pd.DataFrame:
        """Build an OHLCV frame for an implied-volatility index.

        Longer-tenor indices are generated as a decayed blend of the spot level
        towards its unconditional mean, which reproduces the empirical regularity
        the term-structure feature keys on: contango in calm markets, sharp
        backwardation in stress.
        """
        rng = self._rng("volindex", symbol)
        spot = 100.0 * _VOLATILITY[states]

        tenor_days = 93.0 if "3M" in symbol.upper() else 30.0
        decay = float(np.exp(-tenor_days / 120.0))
        unconditional = float(100.0 * _VOLATILITY.mean())
        level = decay * spot + (1.0 - decay) * unconditional
        level = level * (1.0 + 0.05 * rng.standard_normal(states.size))
        level = np.maximum(level, 5.0)

        # Volatility indices are quoted, not traded: no meaningful volume.
        smoothing = pd.Series(level, index=calendar).ewm(span=3, adjust=False).mean()
        close = smoothing.to_numpy()
        noise = np.abs(rng.standard_normal(states.size)) * 0.04 * close

        return pd.DataFrame(
            {
                "open": close + rng.standard_normal(states.size) * 0.02 * close,
                "high": close + noise,
                "low": np.maximum(close - noise, 1.0),
                "close": close,
                "volume": np.full(states.size, np.nan),
            },
            index=calendar,
            columns=list(OHLCV_FIELDS),
        )
