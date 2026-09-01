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
from regimelab.data.providers.parquet import ParquetProvider, write_snapshot
from regimelab.data.providers.synthetic import SyntheticProvider
from regimelab.data.providers.yahoo import YahooProvider

if TYPE_CHECKING:
    from regimelab.settings import Settings

__all__ = [
    "OHLCV_FIELDS",
    "ParquetProvider",
    "PriceProvider",
    "ProviderError",
    "SyntheticProvider",
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
        return YahooProvider()
    if name == "synthetic":
        return SyntheticProvider(seed=settings.model.seed)
    if name == "parquet":
        return ParquetProvider(settings.data.cache_dir / "snapshot")

    msg = f"unknown data provider: {name!r}"
    raise ValueError(msg)
