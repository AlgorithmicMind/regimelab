"""Typed, reproducible configuration.

Configuration is a first-class artefact, not a module of constants: every run
serialises its resolved settings into the run manifest
(:mod:`regimelab.store`), which is what makes a historical result auditable and
reproducible.

Resolution order (highest precedence first):

1. Explicit keyword arguments to :class:`Settings`.
2. Environment variables, ``REGIMELAB_`` prefixed, ``__`` for nesting
   (e.g. ``REGIMELAB_MODEL__N_STATES=3``).
3. ``.env`` file.
4. ``regimelab.toml`` in the working directory.
5. The defaults below.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Annotated, Literal, Self

import platformdirs
from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)

from regimelab.models.types import FeatureSnapshot, MarketRegime

__all__ = [
    "BacktestSettings",
    "DataSettings",
    "FeatureSettings",
    "ModelSettings",
    "Settings",
]

ProviderName = Literal["parquet", "synthetic", "yfinance"]


class _Section(BaseModel):
    """Base for configuration sections."""

    model_config = ConfigDict(frozen=True, extra="forbid", validate_assignment=True)


class DataSettings(_Section):
    """Universe definition and ingestion behaviour."""

    provider: ProviderName = "yfinance"
    """Default provider. ``parquet`` reads a frozen local snapshot and is what
    CI uses, so the test suite is offline and deterministic."""

    risk_assets: tuple[str, ...] = ("SPY", "TLT", "GLD")
    """Allocatable sleeves for the adaptive backtest."""

    benchmark: str = "SPY"
    """Buy-and-hold benchmark for the performance attribution."""

    vix_symbol: str = "^VIX"
    vix3m_symbol: str = "^VIX3M"
    """30-day and 93-day implied volatility, for the term-structure feature."""

    cross_section: tuple[str, ...] = (
        "XLB",
        "XLE",
        "XLF",
        "XLI",
        "XLK",
        "XLP",
        "XLU",
        "XLV",
        "XLY",
    )
    """Cross-section used for the absorption ratio and market breadth.

    Sector SPDRs, deliberately. Computing breadth over index *constituents*
    would require point-in-time membership data, which no free provider offers;
    using today's membership over history is survivorship bias. A fixed sector
    cross-section trades some granularity for an honest, reproducible signal.
    The nine sleeves above have continuous history back to 1998; XLRE (2015) and
    XLC (2018) are excluded so the panel width does not change mid-sample.
    """

    start: dt.date = dt.date(2005, 1, 1)
    end: dt.date | None = None
    """Ingestion bounds. ``None`` means "up to the latest available session"."""

    cache_dir: Path = Field(
        default_factory=lambda: Path(platformdirs.user_cache_dir("regimelab")),
    )
    cache_ttl_hours: Annotated[float, Field(ge=0.0)] = 12.0
    """How long a cached provider response stays fresh. Set to 0 to force refetch."""

    calendar_anchor: str = "SPY"
    """Symbol whose sessions define the master trading calendar."""

    @property
    def all_symbols(self) -> tuple[str, ...]:
        """Every symbol the engine needs, de-duplicated and sorted."""
        symbols = {
            *self.risk_assets,
            *self.cross_section,
            self.benchmark,
            self.vix_symbol,
            self.vix3m_symbol,
            self.calendar_anchor,
        }
        return tuple(sorted(symbols))

    @model_validator(mode="after")
    def _check_universe(self) -> Self:
        if len(self.cross_section) < 3:
            msg = "the absorption ratio needs a cross-section of at least 3 assets"
            raise ValueError(msg)
        if self.end is not None and self.end <= self.start:
            msg = f"end {self.end} must be after start {self.start}"
            raise ValueError(msg)
        return self


class FeatureSettings(_Section):
    """Trailing-window lengths for the feature families.

    Every window is *trailing* by construction. No feature in this engine is
    allowed to read a future observation; see :mod:`regimelab.features`.
    """

    absorption_window: Annotated[int, Field(ge=60)] = 500
    """Covariance estimation window for the absorption ratio.
    500 sessions follows Kritzman & Li (2010)."""

    absorption_eigen_fraction: Annotated[float, Field(gt=0.0, le=1.0)] = 0.2
    """Share of eigenvectors treated as "leading". Kritzman & Li use N/5."""

    absorption_delta_short: Annotated[int, Field(ge=2)] = 15
    absorption_delta_long: Annotated[int, Field(ge=20)] = 252
    """The AR shift compares a 15-day mean against a 1-year mean, standardised
    by the 1-year standard deviation of the ratio."""

    correlation_window: Annotated[int, Field(ge=20)] = 63
    """Rolling window for the mean pairwise cross-asset correlation."""

    breadth_sma: Annotated[int, Field(ge=20)] = 200
    breadth_momentum_window: Annotated[int, Field(ge=5)] = 21

    volatility_window: Annotated[int, Field(ge=5)] = 21
    """Window for the Garman-Klass / Parkinson realised-volatility estimators."""

    trailing_return_window: Annotated[int, Field(ge=5)] = 21

    trading_days_per_year: Annotated[int, Field(ge=200, le=260)] = 252

    @property
    def warmup(self) -> int:
        """Sessions of history required before any feature is defined.

        The pipeline uses this to decide where the usable sample begins, so that
        a partially-warm feature never reaches a model.
        """
        return max(
            self.absorption_window + self.absorption_delta_long,
            self.breadth_sma + self.breadth_momentum_window,
            self.correlation_window,
            self.volatility_window,
            self.trailing_return_window,
        )


class ModelSettings(_Section):
    """Latent-state classifier configuration."""

    classifier: Literal["hmm", "gmm", "hamilton"] = "hmm"
    n_states: Annotated[int, Field(ge=2, le=4)] = 4
    seed: int = 20260831
    """Fixed seed. EM is initialisation-sensitive, so an unseeded fit is not
    reproducible and its regime labels are not stable across runs."""

    n_init: Annotated[int, Field(ge=1)] = 10
    """Restarts per fit. The best-likelihood restart wins, which materially
    reduces the chance of settling in a poor local optimum."""

    max_iter: Annotated[int, Field(ge=10)] = 500
    tolerance: Annotated[float, Field(gt=0.0)] = 1e-4

    covariance_type: Literal["full", "diag", "tied", "spherical"] = "diag"
    """Covariance parameterisation for Gaussian emissions.

    ``diag`` is the production default: it is robust to the short samples and
    collinear features common in walk-forward fits. ``full`` is better in
    principle because stress regimes show up in the correlation structure, but
    it needs more data and longer convergence and is best reserved for quarterly
    research fits."""

    verbose: bool = False
    """Emit per-iteration EM logs from ``hmmlearn`` / ``sklearn``."""

    min_train_observations: Annotated[int, Field(ge=252)] = 756
    """Minimum training sample before the engine will emit a regime. Three years
    of daily data is the floor for a 4-state multivariate fit."""

    min_state_occupancy: Annotated[float, Field(ge=0.0, lt=0.5)] = 0.03
    """Reject fits where any state captures less than this share of the sample:
    such states are noise-fitting artefacts, not regimes."""

    features: tuple[str, ...] = (
        "trailing_return",
        "realised_vol",
        "vix_term_spread",
        "absorption_delta",
        "mean_correlation",
    )
    """Observation vector fed to the classifier, in a fixed order.

    Ordering is part of the contract: reordering changes the fitted covariance
    parameterisation and therefore the results.
    """


class BacktestSettings(_Section):
    """Point-in-time simulation parameters."""

    signal_lag_days: Annotated[int, Field(ge=1)] = 1
    """Sessions between observing a regime and trading on it.

    Must be at least 1. A regime inferred from session ``t``'s close cannot be
    traded at session ``t``'s close; a zero lag is the single most common source
    of fictitious alpha in published regime research.
    """

    refit_frequency_days: Annotated[int, Field(ge=1)] = 21
    """How often the walk-forward loop refits the classifier. Between refits the
    parameters are frozen and only the filter is advanced."""

    transaction_cost_bps: Annotated[float, Field(ge=0.0)] = 5.0
    """Round-trip cost per unit of turnover, in basis points."""

    initial_capital: Annotated[float, Field(gt=0.0)] = 1_000_000.0

    risk_free_rate: Annotated[float, Field(ge=0.0)] = 0.02
    """Annualised rate applied to the cash sleeve and used in the Sharpe ratio."""

    regime_weights: dict[MarketRegime, dict[str, float]] = Field(
        default_factory=lambda: {
            MarketRegime.BULL_TREND: {"SPY": 0.90, "TLT": 0.10, "GLD": 0.00},
            MarketRegime.NEUTRAL_TRANSITION: {"SPY": 0.50, "TLT": 0.25, "GLD": 0.10},
            MarketRegime.HIGH_VOL_BEAR: {"SPY": 0.20, "TLT": 0.40, "GLD": 0.15},
            MarketRegime.RISK_OFF: {"SPY": 0.00, "TLT": 0.30, "GLD": 0.20},
        },
    )
    """Target sleeve weights per regime; the residual is cash.

    Note the deliberate asymmetry: RISK_OFF keeps only a partial bond sleeve.
    In a genuine correlation-convergence event Treasuries stop hedging equities
    (2022 is the reference case), so the defensive posture is cash, not duration.
    """

    confidence_floor: Annotated[float, Field(ge=0.0, le=1.0)] = 0.0
    """Below this modal confidence the engine holds its previous allocation
    instead of acting on an ambiguous read. 0 disables the filter."""

    @model_validator(mode="after")
    def _check_weights(self) -> Self:
        for regime, weights in self.regime_weights.items():
            total = sum(weights.values())
            if total > 1.0 + 1e-9:
                msg = f"regime {regime} allocates {total:.2%}, exceeding available capital"
                raise ValueError(msg)
            for symbol, weight in weights.items():
                if weight < 0.0:
                    msg = f"regime {regime} assigns a negative weight to {symbol}"
                    raise ValueError(msg)
        return self


class Settings(BaseSettings):
    """Root configuration object."""

    model_config = SettingsConfigDict(
        env_prefix="REGIMELAB_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        toml_file="regimelab.toml",
        extra="forbid",
        frozen=True,
        validate_assignment=True,
    )

    data: DataSettings = Field(default_factory=DataSettings)
    features: FeatureSettings = Field(default_factory=FeatureSettings)
    model: ModelSettings = Field(default_factory=ModelSettings)
    backtest: BacktestSettings = Field(default_factory=BacktestSettings)

    artifacts_dir: Path = Field(default_factory=lambda: Path("artifacts"))
    """Where run manifests, telemetry and reports are written."""

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Insert the TOML source below the environment sources."""
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            TomlConfigSettingsSource(settings_cls),
            file_secret_settings,
        )

    @model_validator(mode="after")
    def _check_coherence(self) -> Self:
        """Cross-section validation that individual sections cannot see."""
        unknown = set(self.model.features) - set(_FEATURE_FIELDS)
        if unknown:
            msg = f"unknown model features: {sorted(unknown)}; valid: {sorted(_FEATURE_FIELDS)}"
            raise ValueError(msg)

        for regime, weights in self.backtest.regime_weights.items():
            stray = set(weights) - set(self.data.risk_assets)
            if stray:
                msg = (
                    f"regime {regime} allocates to {sorted(stray)}, "
                    f"which are not in data.risk_assets {list(self.data.risk_assets)}"
                )
                raise ValueError(msg)
        return self


def _feature_fields() -> frozenset[str]:
    """Field names available on a feature snapshot."""
    return frozenset(FeatureSnapshot.model_fields)


_FEATURE_FIELDS: frozenset[str] = _feature_fields()
