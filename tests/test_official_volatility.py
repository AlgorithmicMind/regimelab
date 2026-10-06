"""Offline tests for the official CBOE/FRED volatility ingestion.

No test in this module touches the network: ``_read_csv`` is monkeypatched with
a dispatcher that returns fixed string frames, exactly as the real function
would produce.
"""

from __future__ import annotations

import datetime as dt
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Literal
from urllib.error import URLError

import pandas as pd
import pytest

from regimelab.data.fetcher import PanelFetcher
from regimelab.data.providers import (
    ProviderError,
    SyntheticProvider,
    VolatilityRoutingProvider,
    YahooProvider,
    official,
    panel_from_symbol_frames,
    resolve_provider,
    validate_panel,
)
from regimelab.data.providers.official import OfficialVolatilityProvider
from regimelab.settings import DataSettings, ProviderName, Settings


def _settings(
    provider: ProviderName, volatility_source: Literal["official", "provider"]
) -> Settings:
    """Settings with only the data section overridden."""
    return Settings(data=DataSettings(provider=provider, volatility_source=volatility_source))


# --- Fixture data -----------------------------------------------------------

CBOE_VIX3M_CSV = pd.DataFrame(
    {
        "DATE": ["09/18/2009", "09/21/2009", "09/22/2009"],
        "OPEN": ["24.00", "23.50", "23.10"],
        "HIGH": ["25.50", "24.90", "24.20"],
        "LOW": ["23.80", "23.20", "22.90"],
        "CLOSE": ["25.10", "24.30", "23.80"],
    }
)

CBOE_VIX_CSV = pd.DataFrame(
    {
        "DATE": ["09/22/2009"],
        "OPEN": ["20.00"],
        "HIGH": ["21.00"],
        "LOW": ["19.50"],
        "CLOSE": ["20.50"],
    }
)

# FRED covers two sessions before the CBOE VIX3M history begins, shares the
# CBOE dates (identical closes), and carries one "" and one "." missing row.
FRED_VXVCLS_CSV = pd.DataFrame(
    {
        "observation_date": [
            "2009-09-16",
            "2009-09-17",
            "2009-09-18",
            "2009-09-21",
            "2009-09-22",
            "2009-09-23",
            "2009-09-24",
        ],
        "VXVCLS": ["28.50", "27.90", "25.10", "24.30", "23.80", "", "."],
    }
)

FRED_VIXCLS_CSV = pd.DataFrame(
    {
        "observation_date": ["2009-09-22"],
        "VIXCLS": ["20.50"],
    }
)

FALLBACK_DATES = pd.bdate_range("2009-09-16", "2009-09-25")


def _fallback_frame() -> pd.DataFrame:
    """Fallback OHLCV with closes (~90.xx) far from the official values."""
    n = len(FALLBACK_DATES)
    return pd.DataFrame(
        {
            "open": [80.0] * n,
            "high": [81.0] * n,
            "low": [79.0] * n,
            "close": [90.0 + i / 100 for i in range(n)],
            "volume": [12345.0] * n,
        },
        index=FALLBACK_DATES,
    )


class FakeEquityProvider:
    """Minimal provider double returning fixed per-symbol frames.

    Attributes
    ----------
    calls
        Ordered ``(symbols, start, end)`` records of every fetch.
    """

    def __init__(self, frames: dict[str, pd.DataFrame], *, name: str = "fake-eq") -> None:
        self._frames = frames
        self._name = name
        self.calls: list[tuple[tuple[str, ...], dt.date, dt.date | None]] = []

    @property
    def name(self) -> str:
        return self._name

    @property
    def is_point_in_time(self) -> bool:
        return False

    def fetch(
        self,
        symbols: Sequence[str],
        start: dt.date,
        end: dt.date | None = None,
    ) -> pd.DataFrame:
        self.calls.append((tuple(symbols), start, end))
        if not symbols:
            raise ProviderError("no symbols requested")
        return panel_from_symbol_frames({s: self._frames[s] for s in symbols if s in self._frames})


def _install_fake_csv(
    monkeypatch: pytest.MonkeyPatch,
    *,
    cboe_ok: bool = True,
    fred_ok: bool = True,
) -> None:
    """Replace ``_read_csv`` with a URL dispatcher; selected sources can fail."""

    def fake(url: str) -> pd.DataFrame:
        if "cboe.com" in url:
            if not cboe_ok:
                raise URLError("cboe down")
            # Order matters: "VIX3M_History" must not be caught by "VIX_History".
            if "VIX3M_History" in url:
                return CBOE_VIX3M_CSV
            if "VIX_History" in url:
                return CBOE_VIX_CSV
        if "fred" in url:
            if not fred_ok:
                raise URLError("fred down")
            if "id=VXVCLS" in url:
                return FRED_VXVCLS_CSV
            if "id=VIXCLS" in url:
                return FRED_VIXCLS_CSV
        raise AssertionError(f"unexpected url: {url}")

    monkeypatch.setattr(official, "_read_csv", fake)


# --- Parsing ----------------------------------------------------------------


def test_parse_cboe_mm_dd_yyyy() -> None:
    """CBOE dates parse strictly and volume is NaN."""
    frame = official._parse_cboe(CBOE_VIX3M_CSV)

    assert list(frame.index) == list(pd.DatetimeIndex(["2009-09-18", "2009-09-21", "2009-09-22"]))
    assert frame.loc[pd.Timestamp("2009-09-18"), "open"] == pytest.approx(24.00)
    assert frame.loc[pd.Timestamp("2009-09-18"), "high"] == pytest.approx(25.50)
    assert frame.loc[pd.Timestamp("2009-09-18"), "low"] == pytest.approx(23.80)
    assert frame.loc[pd.Timestamp("2009-09-18"), "close"] == pytest.approx(25.10)
    assert frame["volume"].isna().all()

    with pytest.raises(ValueError, match="missing"):
        official._parse_cboe(CBOE_VIX3M_CSV.drop(columns="CLOSE"))


def test_parse_fred_drops_missing_values() -> None:
    """FRED ``""`` and ``"."`` rows are dropped; only the close survives."""
    frame = official._parse_fred(FRED_VXVCLS_CSV, "VXVCLS")

    assert frame.loc[pd.Timestamp("2009-09-16"), "close"] == pytest.approx(28.50)
    assert pd.Timestamp("2009-09-23") not in frame.index
    assert pd.Timestamp("2009-09-24") not in frame.index
    assert frame["open"].isna().all()
    assert frame["high"].isna().all()
    assert frame["low"].isna().all()
    assert frame["volume"].isna().all()

    # When the series name is absent, the second column is the value column.
    renamed = FRED_VXVCLS_CSV.rename(columns={"VXVCLS": "value"})
    assert official._parse_fred(renamed, "VXVCLS").equals(frame)

    with pytest.raises(ValueError, match="columns"):
        official._parse_fred(FRED_VXVCLS_CSV[["observation_date"]], "VXVCLS")


# --- Merge priority ---------------------------------------------------------


def test_priority_per_date(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Each date is owned by the highest-priority source with a close."""
    _install_fake_csv(monkeypatch)
    fallback = FakeEquityProvider({"^VIX3M": _fallback_frame()})
    provider = OfficialVolatilityProvider(fallback=fallback)

    with caplog.at_level(logging.INFO, logger="regimelab.data.providers.official"):
        panel = provider.fetch(["^VIX3M"], dt.date(2009, 9, 16), dt.date(2009, 9, 25))

    vix3m = panel.xs("^VIX3M", axis="columns", level="symbol")

    # CBOE dates carry CBOE OHLC — the fallback close (~90.xx) must not appear.
    assert vix3m.loc[pd.Timestamp("2009-09-18"), "close"] == pytest.approx(25.10)
    assert vix3m.loc[pd.Timestamp("2009-09-18"), "open"] == pytest.approx(24.00)

    # Dates before the CBOE history begins come from FRED: close only.
    assert vix3m.loc[pd.Timestamp("2009-09-16"), "close"] == pytest.approx(28.50)
    assert vix3m.loc[pd.Timestamp("2009-09-17"), "close"] == pytest.approx(27.90)
    assert pd.isna(vix3m.loc[pd.Timestamp("2009-09-16"), "open"])

    # The freshest date — after both official sources end — is the fallback's.
    assert vix3m.loc[pd.Timestamp("2009-09-25"), "close"] == pytest.approx(90.07)
    assert vix3m.loc[pd.Timestamp("2009-09-25"), "open"] == pytest.approx(80.0)
    # The equity provider's index volume is forced to NaN.
    assert pd.isna(vix3m.loc[pd.Timestamp("2009-09-25"), "volume"])

    # One fallback call for the whole request.
    assert fallback.calls == [(("^VIX3M",), dt.date(2009, 9, 16), dt.date(2009, 9, 25))]

    assert any(
        record.getMessage() == "^VIX3M: cboe=3 fred=2 fallback=3" for record in caplog.records
    )


def test_cboe_failure_falls_to_fred(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A CBOE failure is logged and FRED fills its dates."""
    _install_fake_csv(monkeypatch, cboe_ok=False)
    fallback = FakeEquityProvider({"^VIX3M": _fallback_frame()})
    provider = OfficialVolatilityProvider(fallback=fallback)

    with caplog.at_level(logging.WARNING, logger="regimelab.data.providers.official"):
        panel = provider.fetch(["^VIX3M"], dt.date(2009, 9, 16), dt.date(2009, 9, 25))

    vix3m = panel.xs("^VIX3M", axis="columns", level="symbol")
    # The CBOE dates now come from FRED: same closes, no OHLC fields.
    assert vix3m.loc[pd.Timestamp("2009-09-18"), "close"] == pytest.approx(25.10)
    assert pd.isna(vix3m.loc[pd.Timestamp("2009-09-18"), "open"])
    # The freshest date is still the fallback's.
    assert vix3m.loc[pd.Timestamp("2009-09-25"), "close"] == pytest.approx(90.07)

    assert any(
        "CBOE" in record.getMessage()
        for record in caplog.records
        if record.levelno >= logging.WARNING
    )


def test_both_sources_fail_uses_fallback(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """With both official sources down, the whole panel is the fallback's."""
    _install_fake_csv(monkeypatch, cboe_ok=False, fred_ok=False)
    fallback = FakeEquityProvider({"^VIX3M": _fallback_frame()})
    provider = OfficialVolatilityProvider(fallback=fallback)

    with caplog.at_level(logging.WARNING, logger="regimelab.data.providers.official"):
        panel = provider.fetch(["^VIX3M"], dt.date(2009, 9, 16), dt.date(2009, 9, 25))

    vix3m = panel.xs("^VIX3M", axis="columns", level="symbol")
    assert list(vix3m.index) == list(FALLBACK_DATES)
    assert vix3m["close"].tolist() == pytest.approx(
        [90.0 + i / 100 for i in range(len(FALLBACK_DATES))]
    )

    assert any(
        "fake-eq" in record.getMessage()
        for record in caplog.records
        if record.levelno >= logging.WARNING
    )


def test_clips_to_requested_range(monkeypatch: pytest.MonkeyPatch) -> None:
    """Rows outside ``[start, end]`` are dropped after merging."""
    _install_fake_csv(monkeypatch)
    fallback = FakeEquityProvider({"^VIX3M": _fallback_frame()})
    provider = OfficialVolatilityProvider(fallback=fallback)

    panel = provider.fetch(["^VIX3M"], dt.date(2009, 9, 18), dt.date(2009, 9, 22))
    vix3m = panel.xs("^VIX3M", axis="columns", level="symbol")
    assert list(vix3m.index) == list(pd.DatetimeIndex(["2009-09-18", "2009-09-21", "2009-09-22"]))


# --- Routing ----------------------------------------------------------------


def test_routing_splits_symbol_groups() -> None:
    """Volatility symbols go to the vol provider; the rest to equity."""
    equity = FakeEquityProvider(
        {"SPY": _fallback_frame(), "TLT": _fallback_frame()}, name="fake-eq"
    )
    vol = FakeEquityProvider({"^VIX3M": _fallback_frame()}, name="fake-vol")
    router = VolatilityRoutingProvider(equity, vol, {"^VIX3M"})

    panel = router.fetch(["SPY", "^VIX3M", "TLT"], dt.date(2009, 9, 16), dt.date(2009, 9, 25))

    assert equity.calls == [(("SPY", "TLT"), dt.date(2009, 9, 16), dt.date(2009, 9, 25))]
    assert vol.calls == [(("^VIX3M",), dt.date(2009, 9, 16), dt.date(2009, 9, 25))]

    validate_panel(panel)
    symbols = set(panel.columns.get_level_values("symbol"))
    assert symbols == {"SPY", "TLT", "^VIX3M"}

    assert router.name == "fake-eq+vol:fake-vol"
    assert router.is_point_in_time is False

    # Single-group requests skip the merge entirely.
    equity_only = router.fetch(["SPY"], dt.date(2009, 9, 16), dt.date(2009, 9, 25))
    assert set(equity_only.columns.get_level_values("symbol")) == {"SPY"}
    assert len(vol.calls) == 1


# --- resolve_provider / cache keys ------------------------------------------


def test_resolve_provider_routing() -> None:
    """``volatility_source`` routes only the yfinance provider."""
    provider = resolve_provider(_settings("yfinance", "official"))
    assert isinstance(provider, VolatilityRoutingProvider)
    assert "official" in provider.name

    assert isinstance(resolve_provider(_settings("yfinance", "provider")), YahooProvider)
    assert isinstance(resolve_provider(_settings("synthetic", "official")), SyntheticProvider)


def test_cache_key_differs_by_volatility_source(tmp_path: Path) -> None:
    """The provider name participates in the cache key, so sources never share."""
    symbols = ("SPY", "^VIX")
    start, end = dt.date(2020, 1, 1), dt.date(2020, 12, 31)

    official_fetcher = PanelFetcher(
        resolve_provider(_settings("yfinance", "official")),
        tmp_path,
    )
    plain_fetcher = PanelFetcher(
        resolve_provider(_settings("yfinance", "provider")),
        tmp_path,
    )

    assert (
        official_fetcher._entry_for(symbols, start, end).panel_path
        != plain_fetcher._entry_for(symbols, start, end).panel_path
    )
