"""Shared fixtures for the regimelab test suite.

Every fixture that runs the pipeline uses the synthetic provider so the suite is
network-agnostic and reproducible.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest

from regimelab.pipeline import PipelineResult, run_pipeline
from regimelab.settings import Settings


@pytest.fixture(scope="session")
def test_settings() -> Settings:
    """Default settings used for the whole test session."""
    return Settings(
        data={
            "provider": "synthetic",
            "start": dt.date(2015, 1, 1),
            "end": dt.date(2020, 12, 31),
            "cache_dir": Path(".test-artifacts") / "cache",
        },
        features={
            "absorption_window": 120,
            "absorption_delta_long": 60,
            "breadth_sma": 60,
            "breadth_momentum_window": 20,
            "correlation_window": 20,
            "volatility_window": 10,
            "trailing_return_window": 10,
        },
        model={
            "n_states": 3,
            "n_init": 2,
            "max_iter": 100,
            "min_train_observations": 252,
            "covariance_type": "diag",
            "verbose": False,
        },
        backtest={
            "refit_frequency_days": 126,
            "transaction_cost_bps": 5.0,
            "confidence_floor": 0.4,
        },
        artifacts_dir=Path(".test-artifacts"),
    )


@pytest.fixture(scope="session")
def pipeline_result(test_settings: Settings) -> PipelineResult:
    """A full pipeline result produced once per session."""
    return run_pipeline(test_settings, command="test-session")
