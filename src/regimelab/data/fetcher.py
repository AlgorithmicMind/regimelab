"""Multi-asset ingestion with a content-addressed local cache.

The cache exists for two reasons beyond speed. It removes a network dependency
from the inner development loop, and it makes an interrupted run resumable
without re-hitting a rate-limited endpoint.

Cache keys are content-addressed over ``(provider, symbols, start, end)``. The
provider name is part of the key, so switching adjustment mode or provider can
never serve a stale panel from the previous configuration -- a class of bug that
is extremely hard to spot because the numbers look plausible.

Open-ended requests (``end=None``) are cached with a TTL, since tomorrow's
session will extend them. Closed requests are cached indefinitely, because a
provider that restates a closed historical range is broken and we would rather
detect that explicitly than paper over it.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import pandas as pd

from regimelab.data.alignment import AlignedPanel, align_panel
from regimelab.data.providers import ProviderError, resolve_provider, validate_panel

if TYPE_CHECKING:
    from collections.abc import Sequence

    from regimelab.data.providers import PriceProvider
    from regimelab.settings import Settings

__all__ = ["PanelFetcher", "load_aligned_panel"]

_LOGGER = logging.getLogger(__name__)

_COLUMN_SEPARATOR = "\x1f"
"""ASCII unit separator. Parquet cannot store MultiIndex columns, so they are
flattened for storage; a control character cannot collide with a real ticker."""


@dataclass(frozen=True, slots=True)
class _CacheEntry:
    """Location of a cached panel and its sidecar metadata."""

    panel_path: Path
    meta_path: Path


class PanelFetcher:
    """Fetches panels through a provider, memoised on disk.

    Parameters
    ----------
    provider
        Underlying data source.
    cache_dir
        Root of the on-disk cache. Created on demand.
    ttl_hours
        Freshness budget for open-ended requests. ``0`` disables the cache for
        reads, while still writing it.
    """

    def __init__(
        self,
        provider: PriceProvider,
        cache_dir: Path | str,
        ttl_hours: float = 12.0,
    ) -> None:
        self._provider = provider
        self._cache_dir = Path(cache_dir) / "panels"
        self._ttl = dt.timedelta(hours=ttl_hours)

    @property
    def provider(self) -> PriceProvider:
        """The wrapped provider."""
        return self._provider

    def fetch(
        self,
        symbols: Sequence[str],
        start: dt.date,
        end: dt.date | None = None,
        *,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """Return a panel, from cache when possible.

        Parameters
        ----------
        symbols
            Symbols to retrieve. Sorted internally so that request order does
            not fragment the cache.
        start, end
            Inclusive calendar bounds. ``end=None`` means "latest available".
        force_refresh
            Bypass the cache read. The result is still written back.

        Returns
        -------
        pandas.DataFrame
            A validated panel.
        """
        ordered = tuple(sorted(set(symbols)))
        if not ordered:
            msg = "no symbols requested"
            raise ProviderError(msg)

        entry = self._entry_for(ordered, start, end)

        if not force_refresh:
            cached = self._read_cache(entry, closed=end is not None)
            if cached is not None:
                _LOGGER.debug("panel cache hit: %s", entry.panel_path.name)
                return cached

        _LOGGER.info(
            "fetching %d symbols from %s over [%s, %s]",
            len(ordered),
            self._provider.name,
            start,
            end or "latest",
        )
        panel = self._provider.fetch(ordered, start, end)
        validate_panel(panel)
        self._write_cache(entry, panel, symbols=ordered, start=start, end=end)
        return panel

    def _entry_for(
        self,
        symbols: tuple[str, ...],
        start: dt.date,
        end: dt.date | None,
    ) -> _CacheEntry:
        """Compute the cache paths for a request."""
        payload = json.dumps(
            {
                "provider": self._provider.name,
                "symbols": list(symbols),
                "start": start.isoformat(),
                "end": end.isoformat() if end else None,
            },
            sort_keys=True,
        )
        key = hashlib.sha256(payload.encode()).hexdigest()[:16]
        return _CacheEntry(
            panel_path=self._cache_dir / f"{key}.parquet",
            meta_path=self._cache_dir / f"{key}.json",
        )

    def _read_cache(self, entry: _CacheEntry, *, closed: bool) -> pd.DataFrame | None:
        """Read a cached panel, or ``None`` if absent, stale or unreadable."""
        if not (entry.panel_path.is_file() and entry.meta_path.is_file()):
            return None

        try:
            meta = json.loads(entry.meta_path.read_text(encoding="utf-8"))
            fetched_at = dt.datetime.fromisoformat(meta["fetched_at"])
        except (OSError, ValueError, KeyError):
            _LOGGER.warning("discarding unreadable cache metadata: %s", entry.meta_path)
            return None

        # A closed historical range cannot change; only open-ended requests expire.
        if not closed:
            age = dt.datetime.now(dt.UTC) - fetched_at
            if self._ttl.total_seconds() == 0 or age > self._ttl:
                return None

        try:
            stored = pd.read_parquet(entry.panel_path)
        except (OSError, ValueError):
            _LOGGER.warning("discarding unreadable cached panel: %s", entry.panel_path)
            return None

        panel = _unflatten_columns(stored)
        try:
            validate_panel(panel)
        except ValueError:
            _LOGGER.warning("discarding malformed cached panel: %s", entry.panel_path)
            return None
        return panel

    def _write_cache(
        self,
        entry: _CacheEntry,
        panel: pd.DataFrame,
        *,
        symbols: tuple[str, ...],
        start: dt.date,
        end: dt.date | None,
    ) -> None:
        """Persist a panel and its metadata, tolerating a read-only cache dir."""
        try:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            _flatten_columns(panel).to_parquet(
                entry.panel_path,
                index=True,
                compression="zstd",
            )
            meta: dict[str, Any] = {
                "provider": self._provider.name,
                "is_point_in_time": self._provider.is_point_in_time,
                "symbols": list(symbols),
                "start": start.isoformat(),
                "end": end.isoformat() if end else None,
                "fetched_at": dt.datetime.now(dt.UTC).isoformat(),
                "rows": len(panel),
            }
            entry.meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        except OSError as exc:
            # A cache is an optimisation. Losing it must never fail a run.
            _LOGGER.warning("could not write panel cache: %s", exc)


def _flatten_columns(panel: pd.DataFrame) -> pd.DataFrame:
    """Collapse ``(symbol, field)`` columns into separator-joined strings."""
    flat = panel.copy()
    flat.columns = pd.Index(
        [
            f"{symbol}{_COLUMN_SEPARATOR}{field}"
            for symbol, field in cast("list[tuple[str, str]]", panel.columns.to_list())
        ],
    )
    return flat


def _unflatten_columns(stored: pd.DataFrame) -> pd.DataFrame:
    """Restore the ``(symbol, field)`` MultiIndex from flattened columns."""
    panel = stored.copy()
    panel.columns = pd.MultiIndex.from_tuples(
        [tuple(str(c).split(_COLUMN_SEPARATOR, 1)) for c in stored.columns],
        names=("symbol", "field"),
    )
    return panel


def load_aligned_panel(
    settings: Settings,
    *,
    end: dt.date | None = None,
    force_refresh: bool = False,
) -> AlignedPanel:
    """Fetch and align the full universe described by ``settings``.

    The standard entry point for every command: it guarantees that the panel
    handed to the pipeline has been through both provider validation and
    point-in-time alignment.

    Parameters
    ----------
    settings
        Resolved engine configuration.
    end
        Override for ``settings.data.end``, used by the walk-forward loop to
        restrict ingestion to the evaluated span.
    force_refresh
        Bypass the cache.

    Returns
    -------
    AlignedPanel
        Universe aligned on the anchor's calendar.
    """
    provider = resolve_provider(settings)
    fetcher = PanelFetcher(
        provider,
        cache_dir=settings.data.cache_dir,
        ttl_hours=settings.data.cache_ttl_hours,
    )

    panel = fetcher.fetch(
        settings.data.all_symbols,
        start=settings.data.start,
        end=end if end is not None else settings.data.end,
        force_refresh=force_refresh,
    )

    return align_panel(
        panel,
        anchor=settings.data.calendar_anchor,
        start=settings.data.start,
        end=end if end is not None else settings.data.end,
    )
