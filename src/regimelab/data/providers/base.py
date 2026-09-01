"""Provider protocol and the canonical price-panel contract.

The engine never talks to a vendor SDK directly. Every source implements
:class:`PriceProvider` and returns the same panel shape, which is what allows
the test suite to run offline against a frozen snapshot while production reads
a live vendor.

Panel contract
--------------
A *panel* is a :class:`pandas.DataFrame` with:

- a tz-naive, normalised, strictly increasing, unique ``DatetimeIndex`` named
  ``date``;
- ``MultiIndex`` columns ``(symbol, field)`` with names ``("symbol", "field")``;
- ``field`` drawn from :data:`OHLCV_FIELDS`;
- ``float64`` values, with ``NaN`` for genuinely missing observations.

Missing means missing. Providers must not forward-fill, interpolate or
back-fill: gap policy is a point-in-time decision and belongs to
:mod:`regimelab.data.alignment`, where it can be applied causally and audited.
"""

from __future__ import annotations

import datetime as dt
from typing import TYPE_CHECKING, Final, Protocol, runtime_checkable

import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = [
    "OHLCV_FIELDS",
    "PriceProvider",
    "ProviderError",
    "empty_panel",
    "normalise_index",
    "panel_from_symbol_frames",
    "validate_panel",
]

OHLCV_FIELDS: Final[tuple[str, ...]] = ("open", "high", "low", "close", "volume")
"""Canonical field names, lower-cased. Vendors disagree on casing; we do not."""

INDEX_NAME: Final[str] = "date"
COLUMN_NAMES: Final[tuple[str, str]] = ("symbol", "field")


class ProviderError(RuntimeError):
    """Raised when a provider cannot satisfy a request.

    Deliberately not a subclass of ``ValueError``: a data-availability failure
    is an operational problem, not a programming error, and callers routinely
    need to distinguish the two to decide whether to fall back to a cache.
    """


@runtime_checkable
class PriceProvider(Protocol):
    """A source of daily OHLCV data.

    Implementations must be *referentially transparent* for a closed date range:
    given the same ``(symbols, start, end)`` with ``end`` in the past, repeated
    calls must return identical data. Providers that silently restate history
    break the reproducibility guarantee and must document it.
    """

    @property
    def name(self) -> str:
        """Short identifier, used in cache keys and the run manifest."""
        ...

    @property
    def is_point_in_time(self) -> bool:
        """Whether the source delivers genuine as-reported vintages.

        ``False`` for every free equity provider: adjusted closes are restated
        retroactively for splits and dividends, so a series downloaded today is
        not what a trader observed historically. The engine surfaces this in the
        report rather than pretending otherwise.
        """
        ...

    def fetch(
        self,
        symbols: Sequence[str],
        start: dt.date,
        end: dt.date | None = None,
    ) -> pd.DataFrame:
        """Return a panel for ``symbols`` over ``[start, end]``.

        Parameters
        ----------
        symbols
            Vendor tickers to retrieve.
        start
            First calendar date to include, inclusive.
        end
            Last calendar date to include, inclusive. ``None`` means the latest
            available session.

        Returns
        -------
        pandas.DataFrame
            A panel satisfying the contract documented in this module.

        Raises
        ------
        ProviderError
            If the request cannot be satisfied.
        """
        ...


def empty_panel(symbols: Sequence[str] = ()) -> pd.DataFrame:
    """Return a well-formed, zero-row panel.

    Useful as a neutral element: downstream code can treat "no data" uniformly
    instead of branching on ``None``.
    """
    columns = pd.MultiIndex.from_product(
        [list(symbols), list(OHLCV_FIELDS)],
        names=COLUMN_NAMES,
    )
    index = pd.DatetimeIndex([], name=INDEX_NAME)
    return pd.DataFrame(index=index, columns=columns, dtype="float64")


def normalise_index(frame: pd.DataFrame) -> pd.DataFrame:
    """Coerce a frame's index to the panel convention.

    Drops the time-of-day component and any timezone, sorts, and removes
    duplicate dates keeping the last occurrence.

    Timezone information is discarded on purpose. Mixing tz-aware US/Eastern
    equity sessions with tz-naive index data is a reliable way to produce
    off-by-one-day joins, and a daily engine has no legitimate use for
    intraday resolution.
    """
    result = frame.copy()
    index = pd.DatetimeIndex(pd.to_datetime(result.index, errors="coerce"))

    if index.tz is not None:
        index = index.tz_convert("UTC").tz_localize(None)

    result.index = index.normalize()
    result.index.name = INDEX_NAME
    result = result[result.index.notna()]
    result = result[~result.index.duplicated(keep="last")]
    return result.sort_index()


def panel_from_symbol_frames(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Assemble a panel from per-symbol OHLCV frames.

    Parameters
    ----------
    frames
        Mapping of symbol to a frame whose columns are a subset of
        :data:`OHLCV_FIELDS`. Absent fields are filled with ``NaN`` so the panel
        stays rectangular.

    Returns
    -------
    pandas.DataFrame
        A validated panel. Symbols are ordered as given.
    """
    if not frames:
        return empty_panel()

    prepared: dict[str, pd.DataFrame] = {}
    for symbol, frame in frames.items():
        normalised = normalise_index(frame)
        normalised.columns = pd.Index([str(c).lower() for c in normalised.columns])
        # Reindex onto the canonical field set: rectangular beats convenient.
        prepared[symbol] = normalised.reindex(columns=list(OHLCV_FIELDS)).astype("float64")

    panel = pd.concat(
        list(prepared.values()),
        axis="columns",
        keys=list(prepared.keys()),
        names=list(COLUMN_NAMES),
        sort=True,
    )
    panel.index.name = INDEX_NAME
    panel = panel.sort_index()
    validate_panel(panel)
    return panel


def validate_panel(panel: pd.DataFrame) -> None:
    """Assert that ``panel`` satisfies the panel contract.

    Called at every provider boundary. Structural violations are far cheaper to
    diagnose here than three layers downstream inside an EM loop.

    Raises
    ------
    ValueError
        If the panel violates the contract.
    """
    if not isinstance(panel.index, pd.DatetimeIndex):
        msg = f"panel index must be a DatetimeIndex, got {type(panel.index).__name__}"
        raise ValueError(msg)

    if panel.index.tz is not None:
        msg = "panel index must be tz-naive"
        raise ValueError(msg)

    if not panel.index.is_monotonic_increasing:
        msg = "panel index must be sorted ascending"
        raise ValueError(msg)

    if panel.index.has_duplicates:
        duplicates = panel.index[panel.index.duplicated()].unique()
        msg = f"panel index contains duplicate dates: {list(duplicates[:5])}"
        raise ValueError(msg)

    if not isinstance(panel.columns, pd.MultiIndex) or panel.columns.nlevels != 2:
        msg = "panel columns must be a 2-level MultiIndex of (symbol, field)"
        raise ValueError(msg)

    unknown = set(panel.columns.get_level_values("field")) - set(OHLCV_FIELDS)
    if unknown:
        msg = f"panel contains unknown fields {sorted(unknown)}; expected {list(OHLCV_FIELDS)}"
        raise ValueError(msg)
