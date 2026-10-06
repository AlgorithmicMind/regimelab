"""Pluggable market-data providers.

Everything upstream of the feature layer goes through :class:`PriceProvider`.
Swapping ``yfinance`` for a vintage-aware vendor is a one-line configuration
change, not a refactor -- which is the whole point of paying the abstraction
cost.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from regimelab.data.providers.base import (
    OHLCV_FIELDS,
    PriceProvider,
    ProviderError,
    empty_panel,
    normalise_index,
    panel_from_symbol_frames,
    validate_panel,
)
from regimelab.data.providers.official import (
    SUPPORTED_SYMBOLS,
    OfficialVolatilityProvider,
    VolatilityRoutingProvider,
)
from regimelab.data.providers.parquet import ParquetProvider, write_snapshot
from regimelab.data.providers.synthetic import SyntheticProvider
from regimelab.data.providers.yahoo import YahooProvider

if TYPE_CHECKING:
    from regimelab.settings import Settings

__all__ = [
    "OHLCV_FIELDS",
    "SUPPORTED_SYMBOLS",
    "OfficialVolatilityProvider",
    "ParquetProvider",
    "PriceProvider",
    "ProviderError",
    "SyntheticProvider",
    "VolatilityRoutingProvider",
    "YahooProvider",
    "empty_panel",
    "normalise_index",
    "panel_from_symbol_frames",
    "resolve_provider",
    "validate_panel",
    "write_snapshot",
]


def resolve_provider(settings: Settings) -> PriceProvider:
    """Instantiate the provider named in ``settings.data.provider``.

    Parameters
    ----------
    settings
        Resolved engine configuration.

    Returns
    -------
    PriceProvider
        A ready-to-use provider.

    Raises
    ------
    ValueError
        If the configured provider name is unknown. Unreachable through the
        typed settings path, but reachable when settings are built from raw
        JSON, so it is checked rather than assumed.
    """
    name: str = settings.data.provider

    if name == "yfinance":
        # Routing is deliberate about which providers it wraps: ``synthetic``
        # must stay offline with its indices consistent with the latent-state
        # path, and ``parquet`` is the frozen reproducibility snapshot. Only
        # the live yfinance path gets the official-source overlay.
        equity = YahooProvider()
        if settings.data.volatility_source == "official":
            vol_symbols = {
                settings.data.vix_symbol,
                settings.data.vix3m_symbol,
            } & SUPPORTED_SYMBOLS.keys()
            if vol_symbols:
                return VolatilityRoutingProvider(
                    equity,
                    OfficialVolatilityProvider(fallback=equity),
                    vol_symbols,
                )
        return equity
    if name == "synthetic":
        return SyntheticProvider(seed=settings.model.seed)
    if name == "parquet":
        return ParquetProvider(settings.data.cache_dir / "snapshot")

    msg = f"unknown data provider: {name!r}"
    raise ValueError(msg)
