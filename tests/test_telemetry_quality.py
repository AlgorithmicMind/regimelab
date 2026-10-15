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
from regimelab.models.labeling import StateStatistics, assign_labels
from regimelab.models.types import (
    FeatureSnapshot,
    MarketRegime,
    ModelMetadata,
    RankingMethod,
    RegimeProbabilities,
    RegimeStatistics,
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
    state_statistics: tuple[RegimeStatistics, ...] = (),
) -> TelemetryPayload:
    """A minimally valid payload for exercising the schema validators."""
    return TelemetryPayload(
        asof=asof,
        session_date=session_date,
        observation_date=observation_date,
        regime=MarketRegime.BULL_TREND,
        state_statistics=state_statistics,
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

    def _aligned(_settings: Settings, **_kwargs: object) -> AlignedPanel:
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

    def _aligned(_settings: Settings, **_kwargs: object) -> AlignedPanel:
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


def _statistics() -> tuple[RegimeStatistics, ...]:
    """Two valid state-statistics entries on distinct regimes."""
    return (
        RegimeStatistics(
            regime=MarketRegime.BULL_TREND,
            raw_states=(0,),
            occupancy=0.6,
            ann_return=0.10,
            ann_volatility=0.12,
            sharpe=0.83,
            n_observations=150,
        ),
        RegimeStatistics(
            regime=MarketRegime.RISK_OFF,
            raw_states=(1,),
            occupancy=0.4,
            ann_return=-0.20,
            ann_volatility=0.35,
            sharpe=-0.57,
            n_observations=100,
        ),
    )


def test_state_statistics_validator_and_lookup() -> None:
    """Duplicate regimes are rejected; ``statistics_for`` resolves by regime."""
    asof = dt.date(2020, 1, 10)
    payload = _payload(
        asof=asof,
        session_date=asof,
        observation_date=asof,
        state_statistics=_statistics(),
    )

    assert payload.statistics_for(MarketRegime.RISK_OFF).raw_states == (1,)
    with pytest.raises(KeyError, match="no state statistics for NEUTRAL_TRANSITION"):
        payload.statistics_for(MarketRegime.NEUTRAL_TRANSITION)

    duplicated = (*_statistics(), _statistics()[0].model_copy())
    with pytest.raises(pydantic.ValidationError, match="duplicate regimes"):
        _payload(
            asof=asof,
            session_date=asof,
            observation_date=asof,
            state_statistics=duplicated,
        )


def test_state_statistics_on_payloads(
    test_settings: Settings,
    pipeline_result: PipelineResult,
) -> None:
    """Telemetry carries the statistics of the model owning the observation."""
    payload = pipeline.latest_telemetry(pipeline_result)

    observed = payload.observation_date
    fold = next(
        f for f in pipeline_result.walkforward.folds if f.test_start <= observed <= f.test_end
    )
    # The payload is signed by the model of the fold that labelled the
    # observation -- for the last session that is the last fold.
    assert payload.model.train_end == fold.train_end
    assert payload.state_statistics == fold.statistics
    assert payload.statistics_for(payload.regime).regime == payload.regime

    asof = pipeline_result.panel.calendar[-2].date()
    single = pipeline.run_single_asof(test_settings, asof)
    assert single.state_statistics
    assert [s.regime for s in single.state_statistics] == sorted(
        (s.regime for s in single.state_statistics), key=lambda r: r.risk_rank
    )
    # A 3-state ladder has no HIGH_VOL_BEAR entry.
    with pytest.raises(KeyError, match="no state statistics"):
        single.statistics_for(MarketRegime.HIGH_VOL_BEAR)


def test_payload_accepts_combined_state_statistics() -> None:
    """A duplicate-regime mapping (guardrail) produces unique statistics.

    Regression for the 0.3.0 bug: under VOLATILITY/SHARPE the positive-drift
    guardrail can map two states to the same regime, and one-entry-per-state
    statistics then broke the payload's uniqueness validator.
    """
    stats = (
        StateStatistics(
            state=0, occupancy=0.4, mean_return=0.15, volatility=0.10, n_observations=400
        ),
        StateStatistics(
            state=1, occupancy=0.3, mean_return=0.05, volatility=0.20, n_observations=300
        ),
        StateStatistics(
            state=2, occupancy=0.3, mean_return=0.02, volatility=0.30, n_observations=300
        ),
    )
    assignment = assign_labels(stats, method=RankingMethod.VOLATILITY)
    assert len(set(assignment.mapping.values())) < len(assignment.mapping)

    asof = dt.date(2020, 1, 10)
    payload = _payload(
        asof=asof,
        session_date=asof,
        observation_date=asof,
        state_statistics=assignment.to_statistics(),
    )

    assert len(payload.state_statistics) == 2
    assert payload.statistics_for(MarketRegime.NEUTRAL_TRANSITION).raw_states == (1, 2)


@pytest.fixture(scope="module")
def gmm_pipeline(test_settings: Settings) -> tuple[Settings, PipelineResult]:
    """A full pipeline run under ``gmm`` (VOLATILITY ranking), once per module."""
    settings = test_settings.model_copy(
        update={
            "model": test_settings.model.model_copy(update={"classifier": "gmm"}),
        },
    )
    return settings, pipeline.run_pipeline(settings, command="test-gmm")


def test_gmm_telemetry_carries_unique_state_statistics(
    gmm_pipeline: tuple[Settings, PipelineResult],
) -> None:
    """VOLATILITY-ranked payloads stay valid even when states share a regime."""
    settings, result = gmm_pipeline

    payload = pipeline.latest_telemetry(result)
    assert payload.state_statistics
    regimes = [entry.regime for entry in payload.state_statistics]
    assert len(set(regimes)) == len(regimes)
    assert regimes == sorted(regimes, key=lambda regime: regime.risk_rank)

    asof = result.walkforward.labels.index[-1].date()
    single = pipeline.run_single_asof(settings, asof)
    single_regimes = [entry.regime for entry in single.state_statistics]
    assert len(set(single_regimes)) == len(single_regimes)
    assert single_regimes == sorted(single_regimes, key=lambda regime: regime.risk_rank)

    # On the synthetic panel the gmm fit does produce duplicate mappings:
    # folds exist whose model maps two states to the same regime. The as-of
    # path inside such a fold is the reported crash -- it must stay valid.
    duplicate_fold = next(
        (
            fold
            for fold in result.walkforward.folds
            if any(len(entry.raw_states) > 1 for entry in fold.statistics)
        ),
        None,
    )
    assert duplicate_fold is not None
    probe = pipeline.run_single_asof(settings, duplicate_fold.test_end)
    probe_regimes = [entry.regime for entry in probe.state_statistics]
    assert len(set(probe_regimes)) == len(probe_regimes)
    assert any(len(entry.raw_states) > 1 for entry in probe.state_statistics)
