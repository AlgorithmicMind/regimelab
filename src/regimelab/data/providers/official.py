"""Official sources for the CBOE volatility indices, with provider fallback.

``^VIX`` and ``^VIX3M`` are ingested in priority order:

1. **CBOE** daily-history CSVs (``cdn.cboe.com``) — full OHLC, VIX from
   1990-01-02 and VIX3M from 2009-09-18.
2. **FRED** ``VIXCLS`` / ``VXVCLS`` — close-only; VXVCLS starts 2007-12-04 and
   matches CBOE exactly on every common date.
3. The equity provider (normally ``yfinance``) as a per-date fallback.

Why bother: yfinance has left ``^VIX3M`` empty for whole stretches — 39
sessions, 2026-07-20 to 2026-09-11, before backfilling — which silently turned
the term-structure feature into NaN mid-run. The official sources lag the tape
by about one session (on 2026-10-05 both ended at 2026-10-02), so the equity
provider still covers the freshest dates.

Network note: FRED drops connections carrying a browser User-Agent, so requests
identify as ``regimelab/<version>``. :func:`_read_csv` is the only network call
in this module, which keeps the parsing and merge logic trivially testable.
"""

from __future__ import annotations

import datetime as dt
import io
import logging
from typing import TYPE_CHECKING, Final, cast
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

import regimelab
from regimelab.data.providers.base import (
    INDEX_NAME,
    ProviderError,
    normalise_index,
    panel_from_symbol_frames,
    validate_panel,
)

if TYPE_CHECKING:
    from collections.abc import Collection, Sequence

    from regimelab.data.providers.base import PriceProvider

__all__ = [
    "SUPPORTED_SYMBOLS",
    "OfficialVolatilityProvider",
    "VolatilityRoutingProvider",
]

_LOGGER = logging.getLogger(__name__)

SUPPORTED_SYMBOLS: Final[dict[str, tuple[str, str]]] = {
    "^VIX": ("VIX", "VIXCLS"),
    "^VIX3M": ("VIX3M", "VXVCLS"),
}
"""Engine symbol to (CBOE history name, FRED series)."""

_CBOE_URL: Final[str] = "https://cdn.cboe.com/api/global/us_indices/daily_prices/{name}_History.csv"
_FRED_URL: Final[str] = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"

_VERSION = getattr(regimelab, "__version__", None)
_USER_AGENT: Final[str] = f"regimelab/{_VERSION}" if _VERSION else "regimelab"

_CBOE_REQUIRED: Final[frozenset[str]] = frozenset({"DATE", "OPEN", "HIGH", "LOW", "CLOSE"})


def _read_csv(url: str) -> pd.DataFrame:
    """Download ``url`` and return the raw CSV as a string frame.

    Values are deliberately left as strings — each source has its own
    missing-value convention, and parsing belongs to ``_parse_*``.
    """
    request = Request(url, headers={"User-Agent": _USER_AGENT})
    with urlopen(request, timeout=30) as response:
        body = response.read()
    return pd.read_csv(io.BytesIO(body), dtype=str, keep_default_na=False)


def _parse_cboe(raw: pd.DataFrame) -> pd.DataFrame:
    """Parse a CBOE daily-history CSV (``MM/DD/YYYY`` dates, no volume column)."""
    missing = sorted(_CBOE_REQUIRED - {str(c) for c in raw.columns})
    if missing:
        msg = f"CBOE CSV missing required columns: {missing}"
        raise ValueError(msg)

    frame = pd.DataFrame(
        {
            "open": pd.to_numeric(raw["OPEN"], errors="coerce"),
            "high": pd.to_numeric(raw["HIGH"], errors="coerce"),
            "low": pd.to_numeric(raw["LOW"], errors="coerce"),
            "close": pd.to_numeric(raw["CLOSE"], errors="coerce"),
            "volume": np.nan,
        }
    )
    frame.index = pd.DatetimeIndex(pd.to_datetime(raw["DATE"], format="%m/%d/%Y"))
    return normalise_index(frame.dropna(subset=["close"]))


def _parse_fred(raw: pd.DataFrame, series: str) -> pd.DataFrame:
    """Parse a FRED ``fredgraph`` CSV: an ISO date column plus one value column.

    Missing observations are encoded as ``"."`` in older files and as empty
    fields in current ones; both become NaN and their rows are dropped.
    """
    if raw.shape[1] < 2:
        msg = f"FRED CSV for {series} has {raw.shape[1]} columns; expected 2"
        raise ValueError(msg)

    value_column = series if series in raw.columns else str(raw.columns[1])
    frame = pd.DataFrame(
        {
            "open": np.nan,
            "high": np.nan,
            "low": np.nan,
            "close": pd.to_numeric(raw[value_column], errors="coerce"),
            "volume": np.nan,
        }
    )
    frame.index = pd.DatetimeIndex(pd.to_datetime(raw[raw.columns[0]]))
    return normalise_index(frame.dropna(subset=["close"]))


def _split_panel(panel: pd.DataFrame, symbols: Sequence[str]) -> dict[str, pd.DataFrame]:
    """Extract per-symbol OHLCV frames from a validated panel."""
    available = set(panel.columns.get_level_values("symbol"))
    return {
        symbol: cast(pd.DataFrame, panel.xs(symbol, axis="columns", level="symbol"))
        for symbol in symbols
        if symbol in available
    }


class OfficialVolatilityProvider:
    """Serves ``^VIX``/``^VIX3M`` from CBOE, then FRED, then the fallback.

    Parameters
    ----------
    fallback
        Equity provider used for dates the official sources have not published
        and for symbols outside :data:`SUPPORTED_SYMBOLS`.
    """

    def __init__(self, fallback: PriceProvider | None = None) -> None:
        self._fallback = fallback

    @property
    def name(self) -> str:
        """Provider identifier, with the resolution order baked in."""
        fallback_name = self._fallback.name if self._fallback is not None else "none"
        return f"official-vol[cboe>fred>{fallback_name}]"

    @property
    def is_point_in_time(self) -> bool:
        """Always ``False``: official CSVs are republished, not versioned."""
        return False

    def fetch(
        self,
        symbols: Sequence[str],
        start: dt.date,
        end: dt.date | None = None,
    ) -> pd.DataFrame:
        """Return a panel for ``symbols`` over ``[start, end]``."""
        requested = [str(symbol) for symbol in symbols]
        if not requested:
            msg = "no symbols requested"
            raise ProviderError(msg)

        frames: dict[str, pd.DataFrame] = {}

        unsupported = [s for s in requested if s not in SUPPORTED_SYMBOLS]
        if unsupported:
            if self._fallback is None:
                msg = f"no official source for {unsupported} and no fallback provider"
                raise ProviderError(msg)
            delegated = self._fallback.fetch(unsupported, start, end)
            frames.update(_split_panel(delegated, unsupported))

        supported = [s for s in requested if s in SUPPORTED_SYMBOLS]
        fallback_frames: dict[str, pd.DataFrame] = {}
        if supported and self._fallback is not None:
            # One fetch for every supported symbol in the request: the official
            # sources lag the tape, so the tail dates almost always come from
            # the fallback even when CBOE and FRED are healthy.
            try:
                panel = self._fallback.fetch(supported, start, end)
                fallback_frames = _split_panel(panel, supported)
            except (ProviderError, ValueError) as exc:
                _LOGGER.warning("volatility fallback %s failed: %s", self._fallback.name, exc)

        for symbol in supported:
            cboe_name, fred_series = SUPPORTED_SYMBOLS[symbol]
            frames[symbol] = self._merge_symbol(
                symbol,
                cboe_name,
                fred_series,
                fallback_frames.get(symbol),
                start=start,
                end=end,
            )

        return panel_from_symbol_frames(frames)

    def _merge_symbol(
        self,
        symbol: str,
        cboe_name: str,
        fred_series: str,
        fallback: pd.DataFrame | None,
        *,
        start: dt.date,
        end: dt.date | None,
    ) -> pd.DataFrame:
        """Merge the sources for one symbol with strict per-date ownership."""
        cboe = pd.DataFrame()
        try:
            cboe = _parse_cboe(_read_csv(_CBOE_URL.format(name=cboe_name)))
        except Exception as exc:  # broad catch: any failure demotes the source
            _LOGGER.warning("CBOE %s unavailable: %s", cboe_name, exc)

        fred = pd.DataFrame()
        try:
            fred = _parse_fred(_read_csv(_FRED_URL.format(series=fred_series)), fred_series)
        except Exception as exc:  # broad catch: any failure demotes the source
            _LOGGER.warning("FRED %s unavailable: %s", fred_series, exc)

        if fallback is None:
            fallback = pd.DataFrame()
        else:
            fallback = fallback.dropna(subset=["close"]).copy()
            # The equity provider's volume for an index is meaningless; the
            # panel convention is NaN for genuinely absent fields.
            fallback["volume"] = np.nan

        if cboe.empty and fred.empty:
            fallback_name = self._fallback.name if self._fallback is not None else "none"
            _LOGGER.warning(
                "%s: official sources unavailable; falling back to %s",
                symbol,
                fallback_name,
            )

        # The highest-priority source with a non-NaN close owns the whole row:
        # CBOE first, then FRED on dates CBOE lacks, then the fallback on dates
        # neither covers. Fields are never mixed across sources on one date.
        merged = pd.concat([cboe, fred.loc[~fred.index.isin(cboe.index)]], sort=False)
        merged = pd.concat([merged, fallback.loc[~fallback.index.isin(merged.index)]], sort=False)
        merged = merged.sort_index()

        merged = merged.loc[merged.index >= pd.Timestamp(start)]
        if end is not None:
            merged = merged.loc[merged.index <= pd.Timestamp(end)]

        if merged.empty:
            msg = f"no {symbol} data from CBOE, FRED or the fallback"
            raise ProviderError(msg)

        n_cboe = len(set(merged.index) & set(cboe.index))
        n_fred = len(set(merged.index) & set(fred.index) - set(cboe.index))
        _LOGGER.info(
            "%s: cboe=%d fred=%d fallback=%d",
            symbol,
            n_cboe,
            n_fred,
            len(merged) - n_cboe - n_fred,
        )
        return merged


class VolatilityRoutingProvider:
    """Routes volatility-index symbols to a dedicated provider, the rest to equity.

    Parameters
    ----------
    equity
        Provider for every symbol not in ``volatility_symbols``.
    volatility
        Provider for the volatility indices.
    volatility_symbols
        Symbols owned by ``volatility``.
    """

    def __init__(
        self,
        equity: PriceProvider,
        volatility: PriceProvider,
        volatility_symbols: Collection[str],
    ) -> None:
        self._equity = equity
        self._volatility = volatility
        self._volatility_symbols = {str(s) for s in volatility_symbols}

    @property
    def name(self) -> str:
        """Combined identifier, so cache keys reflect the routing."""
        return f"{self._equity.name}+vol:{self._volatility.name}"

    @property
    def is_point_in_time(self) -> bool:
        """True only if both legs are point-in-time."""
        return self._equity.is_point_in_time and self._volatility.is_point_in_time

    def fetch(
        self,
        symbols: Sequence[str],
        start: dt.date,
        end: dt.date | None = None,
    ) -> pd.DataFrame:
        """Fetch each symbol group from its provider and merge on dates."""
        requested = [str(symbol) for symbol in symbols]
        vol_symbols = [s for s in requested if s in self._volatility_symbols]
        equity_symbols = [s for s in requested if s not in self._volatility_symbols]

        if not vol_symbols:
            return self._equity.fetch(equity_symbols, start, end)
        if not equity_symbols:
            return self._volatility.fetch(vol_symbols, start, end)

        equity_panel = self._equity.fetch(equity_symbols, start, end)
        volatility_panel = self._volatility.fetch(vol_symbols, start, end)
        panel = pd.concat([equity_panel, volatility_panel], axis="columns", sort=False).sort_index()
        panel.index.name = INDEX_NAME
        validate_panel(panel)
        return panel
