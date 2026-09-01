"""Strict point-in-time temporal alignment.

The single most common defect in published regime research is a join that leaks
future information. This module is the chokepoint where that is made impossible
by construction.

Three rules, enforced here and relied upon everywhere downstream.

**1. One master calendar.** Sessions are defined by a single anchor symbol.
Taking the union of every symbol's dates would invent sessions -- a US holiday on
which a foreign listing traded becomes a bar on which the engine believes it
could have rebalanced.

**2. Forward-fill only, with a staleness budget.** Carrying the last *observed*
value forward is causal: on date ``t`` a trader genuinely knew the value from
``t-1``. Back-filling is not causal, it is time travel, and it is never
performed here. Fills are capped at a staleness budget so a delisted or
long-halted symbol degrades to ``NaN`` instead of silently contributing a stale
constant to a covariance matrix.

**3. As-of truncation is a hard cut.** :meth:`AlignedPanel.asof` slices the panel
to ``index <= asof`` and returns a new panel. Downstream code receives no
mechanism to see past its own cursor: there is no future to read.

What this module does *not* fix
------------------------------
Alignment cannot repair a contaminated *source*. A back-adjusted close is
already restated before it arrives here (see
:mod:`regimelab.data.providers.yahoo`). Correct alignment of contaminated inputs
yields a correctly-aligned contaminated result, and the reports say so.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, cast

import numpy as np
import pandas as pd

from regimelab.data.providers.base import OHLCV_FIELDS, validate_panel
from regimelab.store import digest_frame

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = [
    "AlignedPanel",
    "AlignmentError",
    "align_panel",
    "trailing_window",
    "truncate_asof",
]

DEFAULT_MAX_STALENESS: Final[int] = 5
"""Default staleness budget in sessions: one trading week."""


class AlignmentError(ValueError):
    """Raised when a panel cannot be aligned point-in-time."""


def truncate_asof(
    frame: pd.DataFrame | pd.Series,
    asof: dt.date | pd.Timestamp,
) -> pd.DataFrame | pd.Series:
    """Return the rows of ``frame`` observable at or before ``asof``.

    The one-line primitive on which the causality guarantee rests. Prefer it to
    an inline ``.loc[:asof]`` so that the intent is greppable and the boundary
    condition is defined in exactly one place.

    Parameters
    ----------
    frame
        Any date-indexed object.
    asof
        Decision date, inclusive. A value observed *on* ``asof`` is available,
        because these are end-of-session observations and the decision is taken
        after the close. Acting on that decision is deferred by
        ``BacktestSettings.signal_lag_days``.

    Returns
    -------
    pandas.DataFrame or pandas.Series
        Truncated view, same type as the input.
    """
    return frame.loc[: pd.Timestamp(asof)]


def trailing_window(
    frame: pd.DataFrame | pd.Series,
    asof: dt.date | pd.Timestamp,
    length: int,
) -> pd.DataFrame | pd.Series:
    """Return the last ``length`` observations up to and including ``asof``.

    Returns fewer rows than requested when history is short; it does not pad.
    Callers must check the length, because a covariance matrix estimated on a
    partially-warm window is not the quantity they think it is.
    """
    if length <= 0:
        msg = f"window length must be positive, got {length}"
        raise ValueError(msg)
    return truncate_asof(frame, asof).iloc[-length:]


@dataclass(frozen=True, slots=True)
class AlignedPanel:
    """A price panel on a single master calendar, with staleness accounting.

    Attributes
    ----------
    panel
        Aligned panel with ``(symbol, field)`` columns on the master calendar.
    anchor
        Symbol whose sessions define the calendar.
    staleness
        Per-symbol count of sessions since the last genuinely observed close.
        ``0`` means observed today. Used by the features to exclude assets whose
        contribution would be an artefact of forward-filling.
    max_staleness
        Staleness budget applied when filling.
    dropped_symbols
        Symbols removed for having no usable history at all.
    """

    panel: pd.DataFrame
    anchor: str
    staleness: pd.DataFrame
    max_staleness: int
    dropped_symbols: tuple[str, ...] = ()

    @property
    def calendar(self) -> pd.DatetimeIndex:
        """The master trading calendar."""
        index = self.panel.index
        assert isinstance(index, pd.DatetimeIndex)
        return index

    @property
    def symbols(self) -> tuple[str, ...]:
        """Symbols present in the aligned panel, sorted."""
        return tuple(sorted({str(s) for s in self.panel.columns.get_level_values("symbol")}))

    @property
    def start(self) -> pd.Timestamp:
        """First session on the calendar."""
        return self.calendar[0]

    @property
    def end(self) -> pd.Timestamp:
        """Last session on the calendar."""
        return self.calendar[-1]

    def field(self, field: str, symbols: Sequence[str] | None = None) -> pd.DataFrame:
        """Return one OHLCV field across symbols as a ``dates x symbols`` frame.

        Parameters
        ----------
        field
            One of :data:`regimelab.data.providers.base.OHLCV_FIELDS`.
        symbols
            Subset to return, in the order given. ``None`` returns every symbol
            in sorted order, so the column order is deterministic -- which
            matters because PCA on a column-permuted matrix is a different
            computation.
        """
        if field not in OHLCV_FIELDS:
            msg = f"unknown field {field!r}; expected one of {list(OHLCV_FIELDS)}"
            raise ValueError(msg)

        frame = self.panel.xs(field, axis="columns", level="field")
        wanted = list(symbols) if symbols is not None else list(self.symbols)

        absent = [s for s in wanted if s not in frame.columns]
        if absent:
            msg = f"symbols absent from the aligned panel: {absent}"
            raise KeyError(msg)

        return cast(pd.DataFrame, frame.loc[:, wanted])

    def close(self, symbols: Sequence[str] | None = None) -> pd.DataFrame:
        """Closing prices as a ``dates x symbols`` frame."""
        return self.field("close", symbols)

    def log_returns(self, symbols: Sequence[str] | None = None) -> pd.DataFrame:
        """Daily log returns of the closing prices.

        Log rather than simple returns because the features aggregate across
        time (sums are cheaper and better-behaved than products) and because the
        Gaussian emission assumption in the HMM is far less abused by log
        returns.

        The first row is dropped rather than zero-filled: a fabricated zero
        return would be indistinguishable from a genuine flat session and would
        bias the first volatility window downwards.
        """
        prices = self.close(symbols)
        log_prices = cast(pd.DataFrame, np.log(prices.astype(float)))
        return log_prices.diff().iloc[1:]

    def asof(self, asof: dt.date | pd.Timestamp) -> AlignedPanel:
        """Return a new panel truncated to ``index <= asof``.

        The point-in-time cursor. Feature and model code is handed the result of
        this call, never the full panel, so look-ahead is structurally
        unavailable rather than merely discouraged.
        """
        cut = pd.Timestamp(asof)
        if cut < self.start:
            msg = f"as-of date {cut.date()} precedes the calendar start {self.start.date()}"
            raise AlignmentError(msg)

        return AlignedPanel(
            panel=self.panel.loc[:cut],
            anchor=self.anchor,
            staleness=self.staleness.loc[:cut],
            max_staleness=self.max_staleness,
            dropped_symbols=self.dropped_symbols,
        )

    def fresh_symbols(
        self,
        asof: dt.date | pd.Timestamp | None = None,
        tolerance: int = 0,
    ) -> tuple[str, ...]:
        """Symbols genuinely observed within ``tolerance`` sessions of ``asof``.

        The features use this to build their cross-section, so a halted symbol
        contributes nothing rather than contributing a flat line -- a flat line
        depresses measured volatility and inflates measured correlation, biasing
        the absorption ratio in the dangerous direction.
        """
        cut = pd.Timestamp(asof) if asof is not None else self.end
        row = self.staleness.loc[:cut]
        if row.empty:
            return ()
        latest = row.iloc[-1]
        return tuple(sorted(str(s) for s in latest.index[latest <= tolerance]))

    def digest(self) -> str:
        """Content digest of the aligned panel, for the run manifest."""
        return digest_frame(self.panel)

    def summary(self) -> dict[str, object]:
        """Compact description for logging and reports."""
        return {
            "anchor": self.anchor,
            "sessions": len(self.calendar),
            "start": self.start.date().isoformat(),
            "end": self.end.date().isoformat(),
            "symbols": len(self.symbols),
            "dropped_symbols": list(self.dropped_symbols),
            "max_staleness": self.max_staleness,
        }


def align_panel(
    panel: pd.DataFrame,
    *,
    anchor: str,
    max_staleness: int = DEFAULT_MAX_STALENESS,
    start: dt.date | None = None,
    end: dt.date | None = None,
) -> AlignedPanel:
    """Align a raw provider panel onto a single master calendar.

    Parameters
    ----------
    panel
        Raw panel from a provider.
    anchor
        Symbol whose observed sessions define the master calendar. Should be the
        most liquid instrument in the universe.
    max_staleness
        Maximum number of consecutive sessions a value may be carried forward.
        Beyond the budget the value becomes ``NaN``.
    start, end
        Optional inclusive calendar bounds applied after alignment.

    Returns
    -------
    AlignedPanel
        Aligned panel with staleness accounting.

    Raises
    ------
    AlignmentError
        If the anchor is absent or has no observations.
    """
    validate_panel(panel)

    if max_staleness < 0:
        msg = f"max_staleness must be non-negative, got {max_staleness}"
        raise AlignmentError(msg)

    available = {str(s) for s in panel.columns.get_level_values("symbol")}
    if anchor not in available:
        msg = f"calendar anchor {anchor!r} absent from panel; available: {sorted(available)}"
        raise AlignmentError(msg)

    anchor_close = panel[(anchor, "close")].dropna()
    if anchor_close.empty:
        msg = f"calendar anchor {anchor!r} has no observed closes"
        raise AlignmentError(msg)

    calendar = pd.DatetimeIndex(anchor_close.index, name="date")
    if start is not None:
        calendar = calendar[calendar >= pd.Timestamp(start)]
    if end is not None:
        calendar = calendar[calendar <= pd.Timestamp(end)]

    if len(calendar) == 0:
        msg = f"master calendar is empty after applying bounds [{start}, {end}]"
        raise AlignmentError(msg)

    # Reindex first: this is where non-anchor sessions are discarded and gaps
    # become explicit NaNs.
    reindexed = panel.reindex(index=calendar)

    staleness = _staleness_matrix(reindexed, calendar)

    # Forward-fill only, bounded. `limit` counts rows on the master calendar,
    # so the budget is in sessions -- the unit a trader actually experiences.
    filled = reindexed.ffill(limit=max_staleness) if max_staleness > 0 else reindexed

    # Beyond the budget, restore NaN. `ffill(limit=n)` already stops after n
    # rows, but a symbol whose history starts mid-sample must not be filled
    # backwards, so we mask explicitly against the staleness matrix.
    for symbol in {str(s) for s in filled.columns.get_level_values("symbol")}:
        expired = staleness[symbol] > max_staleness
        if not expired.any():
            continue
        for field in OHLCV_FIELDS:
            key = (symbol, field)
            if key in filled.columns:
                filled.loc[expired, key] = np.nan

    usable, dropped = _drop_empty_symbols(filled)

    aligned = AlignedPanel(
        panel=usable.sort_index(axis="columns"),
        anchor=anchor,
        staleness=staleness.loc[:, [s for s in staleness.columns if s not in dropped]],
        max_staleness=max_staleness,
        dropped_symbols=dropped,
    )
    validate_panel(aligned.panel)
    return aligned


def _staleness_matrix(
    reindexed: pd.DataFrame,
    calendar: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Sessions elapsed since each symbol's last observed close.

    ``0`` on a session where the close was observed, ``NaN`` before the symbol's
    first observation -- deliberately not ``inf``, so that "never seen" and "seen
    but stale" remain distinguishable.
    """
    positions = np.arange(len(calendar), dtype="float64")
    columns: dict[str, pd.Series] = {}

    for symbol in sorted({str(s) for s in reindexed.columns.get_level_values("symbol")}):
        key = (symbol, "close")
        if key not in reindexed.columns:
            columns[symbol] = pd.Series(np.nan, index=calendar)
            continue

        observed = reindexed[key].notna().to_numpy()
        last_seen = pd.Series(np.where(observed, positions, np.nan), index=calendar).ffill()
        columns[symbol] = pd.Series(positions, index=calendar) - last_seen

    matrix = pd.DataFrame(columns, index=calendar)
    matrix.index.name = "date"
    matrix.columns.name = "symbol"
    return matrix


def _drop_empty_symbols(panel: pd.DataFrame) -> tuple[pd.DataFrame, tuple[str, ...]]:
    """Remove symbols with no usable close anywhere on the calendar."""
    dropped: list[str] = []

    for symbol in sorted({str(s) for s in panel.columns.get_level_values("symbol")}):
        key = (symbol, "close")
        if key not in panel.columns or panel[key].notna().sum() == 0:
            dropped.append(symbol)

    if not dropped:
        return panel, ()

    keep = [c for c in panel.columns if str(c[0]) not in dropped]
    return panel.loc[:, keep], tuple(dropped)
