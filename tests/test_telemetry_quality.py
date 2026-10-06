"""Data-quality reporting on the telemetry payload.

The payload describes gaps between the requested as-of date and the data that
actually produced the posterior -- it never fills them. All tests run on the
synthetic provider; nothing here touches the network.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pydantic
import pytest

from regimelab import pipeline
from regimelab.data.alignment import AlignedPanel, align_panel
from regimelab.data.fetcher import PanelFetcher
from regimelab.data.providers import (
    OHLCV_FIELDS,
    panel_from_symbol_frames,
    resolve_provider,
)
from regimelab.features import build_feature_matrix
from regimelab.models.types import (
    FeatureSnapshot,
    MarketRegime,
    ModelMetadata,
    RegimeProbabilities,
    TelemetryPayload,
)
from regimelab.pipeline import PipelineResult
from regimelab.settings import Settings

_GAP_SESSIONS = 8


@pytest.fixture(scope="module")
def raw_panel(test_settings: Settings) -> pd.DataFrame:
    """The raw provider panel, fetched the same way ``_fetch_and_align`` does."""
    provider = resolve_provider(test_settings)
    fetcher = PanelFetcher(provider, test_settings.artifacts_dir / "cache", ttl_hours=0)
    return fetcher.fetch(
        test_settings.data.all_symbols,
        test_settings.data.start,
        test_settings.data.end,
    )


def _payload(
    *,
    asof: dt.date,
    session_date: dt.date,
    observation_date: dt.date,
) -> TelemetryPayload:
    """A minimally valid payload for exercising the schema validators."""
    return TelemetryPayload(
        asof=asof,
        session_date=session_date,
        observation_date=observation_date,
        regime=MarketRegime.BULL_TREND,
        probabilities=RegimeProbabilities(
            probabilities={
                MarketRegime.BULL_TREND: 0.6,
                MarketRegime.RISK_OFF: 0.4,
            }
        ),
        features=FeatureSnapshot(),
        model=ModelMetadata(
            name="hmm",
            n_states=2,
            seed=1,
            train_start=dt.date(2019, 1, 1),
            train_end=dt.date(2019, 12, 31),
            n_train_observations=252,
            converged=True,
        ),
    )


def test_session_and_observation_date_validators() -> None:
    """The schema enforces ``observation_date <= session_date <= asof``."""
    asof = dt.date(2020, 1, 10)
    session = dt.date(2020, 1, 9)

    with pytest.raises(pydantic.ValidationError, match="is after session_date"):
        _payload(asof=asof, session_date=session, observation_date=asof)

    with pytest.raises(pydantic.ValidationError, match="is after asof"):
        _payload(asof=asof, session_date=dt.date(2020, 1, 11), observation_date=session)

    # A non-session asof is not stale by itself: only data lag counts.
    assert _payload(asof=asof, session_date=session, observation_date=session).is_stale is False
    assert (
        _payload(asof=asof, session_date=session, observation_date=dt.date(2020, 1, 8)).is_stale
        is True
    )


def test_gap_produces_stale_observation_warnings(
    test_settings: Settings,
    raw_panel: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An 8-session ^VIX3M gap: the payload reports the staleness, verbatim."""
    vix3m = test_settings.data.vix3m_symbol
    raw = raw_panel.copy()
    columns = [(vix3m, field) for field in OHLCV_FIELDS]
    raw.loc[raw.index[-_GAP_SESSIONS:], columns] = np.nan

    aligned = align_panel(raw, anchor=test_settings.data.benchmark)
    calendar = aligned.calendar

    def _aligned(_settings: Settings) -> AlignedPanel:
        return aligned

    monkeypatch.setattr(pipeline, "_fetch_and_align", _aligned)

    asof = calendar[-1].date()
    payload = pipeline.run_single_asof(test_settings, asof)

    assert payload.is_stale is True
    assert payload.observation_date == calendar[-4].date()

    # Independent recomputation of the last filterable row.
    features = build_feature_matrix(aligned, test_settings)
    complete = features.loc[:, list(test_settings.model.features)].dropna()
    assert complete.index[-1].date() == payload.observation_date

    assert payload.features.vix_term_spread is None
    assert payload.warnings == (
        (
            "stale_observation: regime computed on "
            f"{payload.observation_date.isoformat()}, last session {asof.isoformat()}"
        ),
        "missing_feature: vix_term_spread unavailable at asof",
        f"stale_input: {vix3m} last observed {_GAP_SESSIONS} sessions earlier",
    )


def test_non_session_asof_reports_session_date(
    test_settings: Settings,
    raw_panel: pd.DataFrame,
) -> None:
    """A Saturday as-of resolves to Friday's session and says so."""
    aligned = align_panel(raw_panel, anchor=test_settings.data.benchmark)
    calendar = aligned.calendar

    # Premise: 2020-12-25 is a session on the synthetic calendar and
    # 2020-12-26 is a Saturday.
    assert pd.Timestamp("2020-12-25") in calendar
    assert pd.Timestamp("2020-12-26") not in calendar
    assert dt.date(2020, 12, 26).weekday() == 5

    payload = pipeline.run_single_asof(test_settings, dt.date(2020, 12, 26))

    assert payload.session_date == dt.date(2020, 12, 25)
    assert payload.observation_date == dt.date(2020, 12, 25)
    assert payload.is_stale is False
    assert payload.warnings == (
        (
            "non_session_asof: requested asof 2020-12-26 is not a trading "
            "session; last session 2020-12-25"
        ),
    )


def test_non_session_asof_with_gap(
    test_settings: Settings,
    raw_panel: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-session as-of plus a data gap produces both warnings, in order."""
    vix3m = test_settings.data.vix3m_symbol
    session = pd.Timestamp("2020-12-25")

    raw = raw_panel.copy()
    columns = [(vix3m, field) for field in OHLCV_FIELDS]
    sessions = raw.index[raw.index <= session]
    pos = len(sessions) - 1
    raw.loc[sessions[-_GAP_SESSIONS:], columns] = np.nan

    aligned = align_panel(raw, anchor=test_settings.data.benchmark)
    calendar = aligned.calendar
    assert calendar[pos] == session

    def _aligned(_settings: Settings) -> AlignedPanel:
        return aligned

    monkeypatch.setattr(pipeline, "_fetch_and_align", _aligned)

    asof = dt.date(2020, 12, 26)
    payload = pipeline.run_single_asof(test_settings, asof)

    assert payload.is_stale is True
    assert payload.session_date == session.date()
    # The gap blanks the last 3 sessions, so the last filterable row is the
    # session 3 before the last observed session.
    assert payload.observation_date == calendar[pos - 3].date()

    # Independent recomputation of the last filterable row at or before asof.
    features = build_feature_matrix(aligned, test_settings)
    complete = features.loc[:session, list(test_settings.model.features)].dropna()
    assert complete.index[-1].date() == payload.observation_date

    assert payload.warnings == (
        (
            "non_session_asof: requested asof 2020-12-26 is not a trading "
            "session; last session 2020-12-25"
        ),
        (
            "stale_observation: regime computed on "
            f"{payload.observation_date.isoformat()}, last session 2020-12-25"
        ),
        "missing_feature: vix_term_spread unavailable at asof",
        f"stale_input: {vix3m} last observed {_GAP_SESSIONS} sessions earlier",
    )


def test_no_gap_reports_clean_payload(
    test_settings: Settings,
    pipeline_result: PipelineResult,
) -> None:
    """With no data gap the payload observes exactly the requested date."""
    asof = pipeline_result.panel.calendar[-1].date()
    payload = pipeline.run_single_asof(test_settings, asof)

    assert payload.is_stale is False
    assert payload.session_date == asof
    assert payload.observation_date == asof
    assert payload.warnings == ()


def test_warnings_report_stale_and_dropped_symbols() -> None:
    """``stale_input`` and ``dropped_symbol`` formats, in documented order."""
    dates = pd.bdate_range("2024-01-02", periods=10)
    raw = panel_from_symbol_frames(
        {
            "SPY": pd.DataFrame(
                {
                    "open": 100.0,
                    "high": 101.0,
                    "low": 99.0,
                    "close": np.linspace(100.0, 109.0, len(dates)),
                    "volume": 1000.0,
                },
                index=dates,
            ),
            # Observed except the final two sessions -> staleness 2.
            "AAA": pd.DataFrame({"close": [50.0] * 8 + [np.nan] * 2}, index=dates),
            # Never observed -> dropped entirely.
            "BBB": pd.DataFrame({"close": [np.nan] * len(dates)}, index=dates),
        }
    )
    aligned = align_panel(raw, anchor="SPY")
    assert aligned.dropped_symbols == ("BBB",)

    features = pd.DataFrame({"f1": [1.0] * len(dates)}, index=aligned.calendar)
    warnings = pipeline._data_quality_warnings(
        asof=aligned.calendar[-1].date(),
        session_date=aligned.calendar[-1].date(),
        observation_date=aligned.calendar[-1].date(),
        features=features,
        feature_names=("f1",),
        panel=aligned,
    )

    assert warnings == (
        "stale_input: AAA last observed 2 sessions earlier",
        "dropped_symbol: BBB",
    )


def test_latest_telemetry_reports_last_session(pipeline_result: PipelineResult) -> None:
    """``latest_telemetry`` anchors on the last master-calendar session."""
    payload = pipeline.latest_telemetry(pipeline_result)

    expected = pipeline_result.features.index[-1].date()
    assert payload.asof == expected
    assert payload.session_date == expected
    assert payload.observation_date == expected
    assert payload.is_stale is False
    assert payload.warnings == ()
