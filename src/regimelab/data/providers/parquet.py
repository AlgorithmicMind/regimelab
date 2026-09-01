"""Frozen local snapshot provider.

This is the reproducibility anchor of the project. A vendor endpoint can change
its adjustment methodology, rate-limit a CI run or disappear entirely; a Parquet
snapshot committed alongside the code cannot. Every published figure in this
repository is regenerable from a snapshot plus a git commit.

Layout::

    <root>/
        SPY.parquet
        TLT.parquet
        _VIX.parquet        # '^' is not portable in filenames

Each file holds a date-indexed frame with a subset of the OHLCV fields.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from regimelab.data.providers.base import (
    ProviderError,
    panel_from_symbol_frames,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = ["ParquetProvider", "symbol_to_filename", "write_snapshot"]


def symbol_to_filename(symbol: str) -> str:
    """Map a vendor ticker to a portable filename stem.

    ``^`` prefixes index symbols at Yahoo and is illegal on Windows, so it is
    replaced with an underscore. ``/`` and ``\\`` are likewise neutralised to
    keep a ticker from escaping the snapshot directory.
    """
    return symbol.replace("^", "_").replace("/", "-").replace("\\", "-")


class ParquetProvider:
    """Reads a frozen on-disk snapshot.

    Parameters
    ----------
    root
        Directory holding the per-symbol Parquet files.
    strict
        When ``True`` (default), a missing symbol raises. When ``False``, missing
        symbols are returned as all-``NaN`` columns, which is useful for partial
        snapshots in unit tests.
    """

    def __init__(self, root: Path | str, *, strict: bool = True) -> None:
        self._root = Path(root)
        self._strict = strict

    @property
    def name(self) -> str:
        """Provider identifier including the snapshot location."""
        return f"parquet:{self._root.as_posix()}"

    @property
    def is_point_in_time(self) -> bool:
        """A snapshot is as point-in-time as whatever produced it.

        Reported as ``False`` because the snapshots shipped with this repository
        were captured from an adjusted-close vendor. Regenerating them from a
        genuine vintage source would make this ``True``, and nothing else in the
        engine would need to change -- which is the entire reason this
        abstraction exists.
        """
        return False

    def fetch(
        self,
        symbols: Sequence[str],
        start: dt.date,
        end: dt.date | None = None,
    ) -> pd.DataFrame:
        """Read ``symbols`` from the snapshot and slice to ``[start, end]``."""
        if not self._root.is_dir():
            msg = f"snapshot directory not found: {self._root}"
            raise ProviderError(msg)

        frames: dict[str, pd.DataFrame] = {}
        missing: list[str] = []

        for symbol in symbols:
            path = self._root / f"{symbol_to_filename(symbol)}.parquet"
            if not path.is_file():
                missing.append(symbol)
                continue
            try:
                frames[symbol] = pd.read_parquet(path)
            except (OSError, ValueError) as exc:  # pragma: no cover - corrupt file
                msg = f"failed to read snapshot for {symbol} at {path}: {exc}"
                raise ProviderError(msg) from exc

        if missing and self._strict:
            msg = f"snapshot at {self._root} is missing symbols: {sorted(missing)}"
            raise ProviderError(msg)

        if not frames:
            msg = f"snapshot at {self._root} contained none of the requested symbols"
            raise ProviderError(msg)

        panel = panel_from_symbol_frames(frames)

        for symbol in missing:
            for field in panel.columns.get_level_values("field").unique():
                panel[(symbol, field)] = pd.NA

        panel = panel.sort_index(axis="columns")
        upper = pd.Timestamp(end) if end is not None else panel.index.max()
        return panel.loc[pd.Timestamp(start) : upper]


def write_snapshot(panel: pd.DataFrame, root: Path | str) -> tuple[Path, ...]:
    """Persist a panel as a Parquet snapshot readable by :class:`ParquetProvider`.

    Parameters
    ----------
    panel
        A panel satisfying the contract in :mod:`regimelab.data.providers.base`.
    root
        Destination directory; created if absent.

    Returns
    -------
    tuple of Path
        The files written, in symbol order.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for symbol in panel.columns.get_level_values("symbol").unique():
        frame = panel.xs(symbol, axis="columns", level="symbol").dropna(how="all")
        path = root / f"{symbol_to_filename(str(symbol))}.parquet"
        frame.to_parquet(path, index=True, compression="zstd")
        written.append(path)

    return tuple(written)
