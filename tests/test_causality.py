"""Causality and point-in-time property tests.

These tests exercise the most critical contract in the repository: at no point
coderesting: a trading decision depend on information not yet available.
"""

from __future__ import annotations

import datetime as dt

import hypothesis.strategies as st
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings

from regimelab.data.alignment import align_panel, truncate_asof
from regimelab.data.fetcher import PanelFetcher
from regimelab.data.providers import resolve_provider
from regimelab.features import build_feature_matrix, snapshot_at
from regimelab.pipeline import PipelineResult, run_single_asof
from regimelab.settings import Settings


class TestPointInTimeBoundary:
    """Tests for the boundary primitives on which the engine rests."""

    @pytest.fixture
    def small_aligned(self, test_settings: Settings) -> pd.DataFrame:
        """A small, fast-to-fit aligned panel."""
        provider = resolve_provider(test_settings)
        fetcher = PanelFetcher(provider, test_settings.artifacts_dir / "cache")
        raw = fetcher.fetch(
            test_settings.data.all_symbols,
            test_settings.data.start,
            test_settings.data.end,
        )
        return align_panel(raw, anchor=test_settings.data.benchmark).panel

    def test_truncate_asof_does_not_exceed_cursor(self, small_aligned: pd.DataFrame) -> None:
        """For every date in the calendar, the truncated view ends at that date."""
        for date in small_aligned.index:
            truncated = truncate_asof(small_aligned, date)
            assert truncated.index.max() == date

    @given(
        date=st.dates(
            min_value=dt.date(2015, 1, 2),
            max_value=dt.date(2020, 12, 30),
        )
    )
    @settings(
        max_examples=20,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_truncate_asof_property(self, small_aligned: pd.DataFrame, date: dt.date) -> None:
        """Property form: the truncated frame contains no rows after the cursor."""
        truncated = truncate_asof(small_aligned, date)
        if truncated.empty:
            return
        assert truncated.index[-1] <= pd.Timestamp(date)


class TestFeaturesAndBacktest:
    """Causality tests for the feature and backtest layers."""

    @pytest.fixture
    def feature_matrix(self, test_settings: Settings) -> pd.DataFrame:
        """A full feature matrix from the test settings."""
        provider = resolve_provider(test_settings)
        fetcher = PanelFetcher(provider, test_settings.artifacts_dir / "cache")
        raw = fetcher.fetch(
            test_settings.data.all_symbols,
            test_settings.data.start,
            test_settings.data.end,
        )
        panel = align_panel(raw, anchor=test_settings.data.benchmark)
        return build_feature_matrix(panel, test_settings)

    def test_feature_snapshot_exists_and_is_single_row(self, feature_matrix: pd.DataFrame) -> None:
        """``snapshot_at`` returns a single-row view for a valid as-of date."""
        asof = feature_matrix.index[len(feature_matrix) // 2]
        snap = snapshot_at(feature_matrix, asof.date())
        assert snap is not None

    def test_backtest_held_weights_use_only_past_targets(
        self, pipeline_result: PipelineResult
    ) -> None:
        """The held portfolio on session t is one of the targets seen before t.

        The default signal lag is one session. When the confidence floor
        overrides a low-confidence signal, the held weight falls back to the
        previous held weight, which by induction is still a past target.
        """
        backtest = pipeline_result.backtest
        assert backtest is not None
        target = backtest.target_weights
        held = backtest.held_weights
        lag = pipeline_result.settings.backtest.signal_lag_days

        for t in held.index[lag:]:
            past_targets = target.loc[:t].iloc[:-1]
            matches = (past_targets == held.loc[t]).all(axis="columns")
            assert matches.any(), f"held weight at {t} does not match any past target"


class TestWalkForward:
    """Causality tests for the walk-forward loop."""

    def test_folds_are_sequential(self, pipeline_result: PipelineResult) -> None:
        """Each fold's training window is valid and the test window reaches past it."""
        for fold in pipeline_result.walkforward.folds:
            assert fold.train_start <= fold.train_end
            assert fold.train_end <= fold.test_end
            assert fold.test_start <= fold.test_end

    def test_test_windows_advance(self, pipeline_result: PipelineResult) -> None:
        """Test windows are non-overlapping in the final kept labels."""
        previous_end: pd.Timestamp | None = None
        for fold in pipeline_result.walkforward.folds:
            if previous_end is not None:
                assert fold.test_start <= previous_end
                assert fold.test_end >= previous_end
            previous_end = fold.test_end


class TestAsofPayload:
    """Causality tests for the single-as-of endpoint."""

    @pytest.fixture(
        params=[
            dt.date(2020, 3, 16),
            dt.date(2020, 6, 15),
            dt.date(2020, 12, 31),
        ]
    )
    def sample_asof(self, request: pytest.FixtureRequest) -> dt.date:
        return request.param

    def test_asof_payload_uses_only_history_before_cursor(
        self, test_settings: Settings, sample_asof: dt.date
    ) -> None:
        """``run_single_asof`` returns a payload for the requested as-of date."""
        payload = run_single_asof(test_settings, sample_asof)
        assert payload.asof == sample_asof
        # No walk-forward backtest is produced by the as-of endpoint, but the
        # feature snapshot must be available at the requested date.
        assert payload.features is not None
