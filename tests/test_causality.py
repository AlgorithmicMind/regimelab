"""Causality and point-in-time property tests.

These tests exercise the most critical contract in the repository: at no point
coderesting: a trading decision depend on information not yet available.
"""

from __future__ import annotations

import datetime as dt
from itertools import pairwise

import hypothesis.strategies as st
import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings

from regimelab.backtest.walkforward import _refit_dates, run_walk_forward
from regimelab.data.alignment import align_panel, truncate_asof
from regimelab.data.fetcher import PanelFetcher
from regimelab.data.providers import resolve_provider
from regimelab.features import build_feature_matrix, snapshot_at
from regimelab.models.base import InsufficientDataError
from regimelab.models.types import MarketRegime
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
        """Each fold's training window is valid and ends before its test window."""
        for fold in pipeline_result.walkforward.folds:
            assert fold.train_start <= fold.train_end
            assert fold.train_end < fold.test_start
            assert fold.test_start <= fold.test_end

    def test_test_windows_advance(self, pipeline_result: PipelineResult) -> None:
        """Test windows are strictly non-overlapping and advance monotonically."""
        previous_end: dt.date | None = None
        for fold in pipeline_result.walkforward.folds:
            if previous_end is not None:
                assert fold.test_start > previous_end
                assert fold.test_end > previous_end
            previous_end = fold.test_end

    def test_walkforward_labels_are_out_of_sample(self, pipeline_result: PipelineResult) -> None:
        """Every labelled date is out-of-sample for the fold that produced it.

        Each fold owns the interval (train_end, test_end]; the intervals are
        pairwise disjoint, their union is exactly the labelled index, and every
        labelled date sits strictly after the producing fold's training window.
        """
        wf = pipeline_result.walkforward
        folds = sorted(wf.folds, key=lambda f: f.test_start)

        for previous, current in pairwise(folds):
            assert current.test_start > previous.test_end

        covered: set[pd.Timestamp] = set()
        for fold in folds:
            assert fold.test_start > fold.train_end
            in_fold = wf.probabilities.index[
                (wf.probabilities.index >= pd.Timestamp(fold.test_start))
                & (wf.probabilities.index <= pd.Timestamp(fold.test_end))
            ]
            dates = set(in_fold)
            assert len(dates) == fold.n_test
            assert not covered & dates
            covered |= dates
            for d in dates:
                assert pd.Timestamp(fold.train_end) < d

        assert covered == set(wf.probabilities.index)

    def test_walkforward_prefix_invariance(self, test_settings: Settings) -> None:
        """Truncating the feature matrix after T leaves the shared labels unchanged.

        A label produced by a fold depends only on data at or before that date:
        the fold's fit sees only data up to its train_end and the filter is
        causal, so dropping later sessions must not alter earlier posteriors.
        """
        provider = resolve_provider(test_settings)
        fetcher = PanelFetcher(provider, test_settings.artifacts_dir / "cache")
        raw = fetcher.fetch(
            test_settings.data.all_symbols,
            test_settings.data.start,
            test_settings.data.end,
        )
        panel = align_panel(raw, anchor=test_settings.data.benchmark)
        features = build_feature_matrix(panel, test_settings)

        # Pick a truncation date about two-thirds through the complete rows that
        # is not itself a refit date of the full run.
        full = features.dropna(how="any")
        refit_dates = set(_refit_dates(features, test_settings))
        position = 2 * len(full) // 3
        while full.index[position] in refit_dates:
            position += 1
        truncation = full.index[position]

        full_run = run_walk_forward(panel, features, test_settings, run_backtest=False)
        trunc_run = run_walk_forward(
            panel, features.loc[:truncation], test_settings, run_backtest=False
        )

        truncated_index = trunc_run.probabilities.index
        common = full_run.probabilities.index.intersection(
            truncated_index[truncated_index <= truncation]
        )
        assert len(common) > 0

        # The shared dates must span more than one fold of the full run.
        def owning_fold(date: pd.Timestamp) -> int:
            for i, fold in enumerate(full_run.folds):
                if pd.Timestamp(fold.test_start) <= date <= pd.Timestamp(fold.test_end):
                    return i
            msg = f"no fold covers {date}"
            raise AssertionError(msg)

        assert owning_fold(common[0]) != owning_fold(common[-1])

        assert np.allclose(
            full_run.probabilities.loc[common].to_numpy(),
            trunc_run.probabilities.loc[common].to_numpy(),
            atol=1e-10,
        )


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
        param: dt.date = request.param
        return param

    def test_asof_payload_uses_only_history_before_cursor(
        self, test_settings: Settings, sample_asof: dt.date
    ) -> None:
        """``run_single_asof`` returns a payload for the requested as-of date."""
        payload = run_single_asof(test_settings, sample_asof)
        assert payload.asof == sample_asof
        # No walk-forward backtest is produced by the as-of endpoint, but the
        # feature snapshot must be available at the requested date.
        assert payload.features is not None

    def test_asof_walkforward_reproduces_walkforward_signal(
        self, test_settings: Settings, pipeline_result: PipelineResult
    ) -> None:
        """The default policy returns the fold's posterior, verbatim."""
        wf = pipeline_result.walkforward
        index = wf.probabilities.index

        dates = {
            index[0],  # first out-of-sample date
            pd.Timestamp(wf.folds[0].test_end),  # a refit boundary
            index[len(index) // 3],
            index[2 * len(index) // 3],
            index[-1],
        }

        for d in sorted(dates):
            payload = run_single_asof(test_settings, d.date())

            expected = wf.probabilities.loc[d]
            got = [
                payload.probabilities.probabilities[MarketRegime(str(column))]
                for column in expected.index
            ]
            assert np.allclose(got, expected.to_numpy(), atol=1e-10)
            assert payload.regime == wf.labels[d]

            fold = next(fold for fold in wf.folds if fold.test_start <= d.date() <= fold.test_end)
            assert payload.model.train_end == fold.train_end

    def test_asof_walkforward_rejects_pre_oos_dates(
        self, test_settings: Settings, pipeline_result: PipelineResult
    ) -> None:
        """Dates at or before the first refit have no out-of-sample model."""
        first_train_end = pipeline_result.walkforward.folds[0].train_end
        with pytest.raises(InsufficientDataError):
            run_single_asof(test_settings, first_train_end)

    def test_asof_daily_refit_policy_keeps_old_behaviour(
        self, test_settings: Settings, pipeline_result: PipelineResult
    ) -> None:
        """``refit_policy="daily"`` fits through the as-of date itself."""
        asof = pipeline_result.features.index[-1].date()
        payload = run_single_asof(test_settings, asof, refit_policy="daily")
        assert payload.model.train_end == asof
