"""Yahoo Finance provider.

Use with clear eyes about what this data is.

``auto_adjust=True`` returns a *back-adjusted* series: today's download restates
every historical close for splits and dividends that had not yet occurred. That
series is therefore not what a trader observed at the time, and any backtest run
on it is contaminated regardless of how careful the alignment logic is. The
contamination is small for the liquid ETFs used here and large for anything with
corporate-action history.

The engine handles this by (a) reporting ``is_point_in_time = False``, which
propagates a caveat into every generated report, and (b) making the provider
swappable, so a genuine vintage source can be dropped in without touching a
single line of feature or model code.

Yahoo is the demo path. The snapshot provider is the reproducible path.
"""

from __future__ import annotations

import datetime as dt
from typing import TYPE_CHECKING, Any, cast

import pandas as pd

from regimelab.data.providers.base import (
    OHLCV_FIELDS,
    ProviderError,
    panel_from_symbol_frames,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = ["YahooProvider"]


class YahooProvider:
    """Fetches daily OHLCV from Yahoo Finance via ``yfinance``.

    Parameters
    ----------
    auto_adjust
        Whether to request split- and dividend-adjusted prices. Kept
        configurable rather than hard-coded so the effect of adjustment on
        results can be measured instead of assumed.
    timeout
        Per-request timeout in seconds.
    """

    def __init__(self, *, auto_adjust: bool = True, timeout: int = 30) -> None:
        self._auto_adjust = auto_adjust
        self._timeout = timeout

    @property
    def name(self) -> str:
        """Provider identifier, adjustment mode included."""
        return f"yfinance{'-adjusted' if self._auto_adjust else '-raw'}"

    @property
    def is_point_in_time(self) -> bool:
        """Always ``False``; see the module docstring."""
        return False

    def fetch(
        self,
        symbols: Sequence[str],
        start: dt.date,
        end: dt.date | None = None,
    ) -> pd.DataFrame:
        """Download a panel for ``symbols`` over ``[start, end]``."""
        symbols = list(symbols)
        if not symbols:
            msg = "no symbols requested"
            raise ProviderError(msg)

        import yfinance  # noqa: PLC0415 - imported lazily for fast CLI startup

        # Yahoo treats `end` as exclusive; the panel contract is inclusive.
        end_exclusive = (end + dt.timedelta(days=1)) if end is not None else None

        try:
            raw: pd.DataFrame = yfinance.download(
                tickers=symbols,
                start=start.isoformat(),
                end=end_exclusive.isoformat() if end_exclusive else None,
                interval="1d",
                auto_adjust=self._auto_adjust,
                actions=False,
                progress=False,
                group_by="ticker",
                threads=True,
                timeout=self._timeout,
            )
        except Exception as exc:
            msg = f"yfinance download failed for {symbols}: {exc}"
            raise ProviderError(msg) from exc

        if raw is None or raw.empty:
            msg = f"yfinance returned no data for {symbols} over [{start}, {end}]"
            raise ProviderError(msg)

        return panel_from_symbol_frames(self._split_by_symbol(raw, symbols))

    @staticmethod
    def _split_by_symbol(
        raw: pd.DataFrame,
        symbols: Sequence[str],
    ) -> dict[str, pd.DataFrame]:
        """Normalise yfinance's shape-shifting output into per-symbol frames.

        ``yfinance.download`` returns flat columns for one ticker and a
        ``(ticker, field)`` MultiIndex for several, and the level order has
        changed between releases. Rather than trusting the shape, we detect it.
        """
        frames: dict[str, pd.DataFrame] = {}
        requested = {s.upper(): s for s in symbols}

        if not isinstance(raw.columns, pd.MultiIndex):
            if len(symbols) != 1:
                msg = f"expected MultiIndex columns for {len(symbols)} symbols"
                raise ProviderError(msg)
            frames[symbols[0]] = _select_fields(raw)
            return frames

        # Identify which level carries the tickers.
        level_0 = {str(v).upper() for v in raw.columns.get_level_values(0)}
        ticker_level = 0 if level_0 & requested.keys() else 1

        for key in raw.columns.get_level_values(ticker_level).unique():
            symbol = requested.get(str(key).upper())
            if symbol is None:
                continue
            frame = cast(pd.DataFrame, raw.xs(key, axis="columns", level=ticker_level))
            selected = _select_fields(frame)
            if not selected.empty:
                frames[symbol] = selected

        if not frames:
            msg = f"none of {list(symbols)} were present in the yfinance response"
            raise ProviderError(msg)

        return frames


def _select_fields(frame: pd.DataFrame) -> pd.DataFrame:
    """Lower-case the columns and keep only the canonical OHLCV fields."""
    renamed: dict[Any, str] = {c: str(c).lower().replace(" ", "_") for c in frame.columns}
    result = frame.rename(columns=renamed)
    keep = [c for c in OHLCV_FIELDS if c in result.columns]
    return result.loc[:, keep].dropna(how="all")
