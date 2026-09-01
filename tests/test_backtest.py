"""Tests for the backtest engine and the metrics it produces."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

from regimelab.backtest.engine import run_backtest
from regimelab.data.alignment import align_panel
from regimelab.data.fetcher import PanelFetcher
from regimelab.data.providers import resolve_provider
from regimelab.features import build_feature_matrix
from regimelab.models.registry import build_classifier
from regimelab.models.types import MarketRegime
from regimelab.pipeline import PipelineResult
from regimelab.settings import BacktestSettings, Settings


def test_held_weights_use_only_past_targets(
    pipeline_result: PipelineResult,
) -> None:
    """Every held allocation is a target that was already available.

    With the default signal lag of one session, the portfolio held on session t
    must be identical to a target weight that appeared on or before t - 1.
    """
    backtest = pipeline_result.backtest
    assert backtest is not None
    held = backtest.held_weights
    target = backtest.target_weights
    lag = pipeline_result.settings.backtest.signal_lag_days

    for t in held.index[lag:]:
        past_targets = target.loc[:t].iloc[:-1]
        matches = (past_targets == held.loc[t]).all(axis="columns")
        assert matches.any(), f"held weight at {t} does not match any past target"


def test_transaction_costs_reduce_returns(
    pipeline_result: PipelineResult,
) -> None:
    """Gross returns are always weakly greater than net returns."""
    backtest = pipeline_result.backtest
    assert backtest is not None
    assert (backtest.gross_returns >= backtest.net_returns).all()


def test_weights_sum_to_one(
    pipeline_result: PipelineResult,
) -> None:
    """Every target and held allocation sums to one."""
    backtest = pipeline_result.backtest
    assert backtest is not None
    sums = backtest.held_weights.sum(axis="columns")
    assert ((sums - 1.0).abs() < 1e-12).all()

    target_sums = backtest.target_weights.sum(axis="columns")
    assert ((target_sums - 1.0).abs() < 1e-12).all()


def test_confidence_floor_preserves_previous_allocation() -> None:
    """When confidence is below the floor the prior allocation is retained."""
    backtest_cfg = BacktestSettings(
        regime_weights={
            MarketRegime.BULL_TREND: {"SPY": 1.0},
            MarketRegime.RISK_OFF: {"TLT": 1.0},
        },
        confidence_floor=0.7,
        signal_lag_days=1,
        transaction_cost_bps=0.0,
    )
    settings = Settings(
        data={"provider": "synthetic", "start": dt.date(2018, 1, 1), "end": dt.date(2020, 1, 1)},
        backtest=backtest_cfg,
        model={
            "n_states": 2,
            "n_init": 2,
            "max_iter": 100,
            "min_train_observations": 252,
            "covariance_type": "diag",
            "verbose": False,
        },
        features={
            "absorption_window": 60,
            "absorption_delta_long": 40,
            "breadth_sma": 40,
            "breadth_momentum_window": 10,
            "correlation_window": 20,
            "volatility_window": 10,
            "trailing_return_window": 10,
        },
        artifacts_dir=Path(".test-artifacts"),
    )

    provider = resolve_provider(settings)
    fetcher = PanelFetcher(provider, settings.artifacts_dir / "cache")
    raw = fetcher.fetch(settings.data.all_symbols, settings.data.start, settings.data.end)
    panel = align_panel(raw, anchor=settings.data.benchmark)
    features = build_feature_matrix(panel, settings)
    classifier = build_classifier(settings.model)
    returns = panel.log_returns([settings.data.benchmark])[settings.data.benchmark]
    fit = classifier.fit(features, returns)
    probs = fit.probabilities(features)
    labels = fit.labels(features)
    result = run_backtest(
        panel.close(settings.data.risk_assets),
        labels,
        settings,
        confidence=probs.max(axis="columns"),
    )

    low_confidence = probs.max(axis="columns") < backtest_cfg.confidence_floor
    # For sessions after the first, a low-confidence signal should produce the
    # same held weights as the previous session.
    for t in low_confidence.index[1:]:
        if low_confidence.loc[t]:
            prev_t = result.held_weights.index[result.held_weights.index.get_loc(t) - 1]
            assert result.held_weights.loc[t].equals(result.held_weights.loc[prev_t])
