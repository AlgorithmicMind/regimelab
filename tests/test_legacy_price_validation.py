"""Reject undefined exposed returns without changing valid legacy arithmetic."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from regimelab.backtest import run_backtest, simulate_weights
from regimelab.models.types import MarketRegime
from tests._next_open_fixtures import make_settings, market


@pytest.mark.parametrize("bad", [np.nan, np.inf, 0.0, -1.0])
def test_used_current_close_must_be_observed_positive_and_finite(bad: float) -> None:
    prices, _, _ = market(4)
    prices.loc[prices.index[2], "GLD"] = bad
    targets = pd.DataFrame({"GLD": [1.0]}, index=prices.index[:1])
    with pytest.raises(ValueError, match=r"GLD.*2024-01-04"):
        simulate_weights(prices, targets, make_settings())


@pytest.mark.parametrize("bad", [np.nan, 0.0])
def test_used_return_also_requires_its_previous_close(bad: float) -> None:
    prices, _, _ = market(4)
    prices.loc[prices.index[0], "GLD"] = bad
    targets = pd.DataFrame({"GLD": [1.0]}, index=prices.index[:1])
    with pytest.raises(ValueError, match=r"GLD.*2024-01-03"):
        simulate_weights(prices, targets, make_settings())


def test_unused_missing_asset_does_not_remove_valid_spy_returns() -> None:
    prices, _, _ = market(4)
    prices["GLD"] = np.nan
    prices["SPY"] = [100.0, 110.0, 121.0, 121.0]
    targets = pd.DataFrame({"SPY": [1.0]}, index=prices.index[:1])
    result = simulate_weights(prices, targets, make_settings())
    np.testing.assert_allclose(result.net_returns, [0.0, 0.1, 0.1, 0.0], atol=1e-15)


def test_unexposed_warmup_gap_is_not_mistaken_for_a_held_position() -> None:
    prices, _, _ = market(5)
    prices["GLD"] = [np.nan, np.nan, 100.0, 110.0, 110.0]
    targets = pd.DataFrame({"GLD": [1.0]}, index=prices.index[2:3])
    result = simulate_weights(prices, targets, make_settings())
    np.testing.assert_allclose(result.net_returns, [0.0, 0.1, 0.0], atol=1e-15)


def test_benchmark_gap_is_an_error_even_when_the_strategy_holds_another_asset() -> None:
    prices, _, _ = market(4)
    prices.loc[prices.index[2], "SPY"] = np.nan
    targets = pd.DataFrame({"GLD": [1.0]}, index=prices.index[:1])
    with pytest.raises(ValueError, match=r"SPY.*2024-01-04"):
        simulate_weights(prices, targets, make_settings())


def test_only_the_first_benchmark_interval_may_lack_a_previous_input_mark() -> None:
    prices, _, _ = market(3)
    prices["SPY"] = [100.0, 110.0, 121.0]
    targets = pd.DataFrame({"SPY": [0.0]}, index=prices.index[:1])
    result = simulate_weights(prices, targets, make_settings())
    assert pd.isna(result.benchmark_returns.iloc[0])
    assert result.benchmark_metrics.n_observations == 2
    assert result.benchmark_metrics.total_return == pytest.approx(0.21)
    assert result.benchmark_equity.iloc[-1] == pytest.approx(1.21)


def test_missing_first_benchmark_mark_is_not_a_missing_previous_interval() -> None:
    prices, _, _ = market(3)
    prices.loc[prices.index[0], "SPY"] = np.nan
    targets = pd.DataFrame({"GLD": [1.0]}, index=prices.index[:1])
    with pytest.raises(ValueError, match=r"SPY.*2024-01-02"):
        simulate_weights(prices, targets, make_settings())


def test_finite_marks_with_an_unrepresentable_return_do_not_reach_metrics() -> None:
    prices, _, _ = market(3)
    prices["GLD"] = [1e-308, 1e308, 1e308]
    targets = pd.DataFrame({"GLD": [1.0]}, index=prices.index[:1])
    with pytest.raises(ValueError, match=r"GLD.*2024-01-03"):
        simulate_weights(prices, targets, make_settings())


def test_nullable_missing_prices_are_not_silently_skipped() -> None:
    prices, _, _ = market(4)
    prices["GLD"] = pd.Series([100.0, 100.0, pd.NA, 100.0], index=prices.index, dtype="Float64")
    targets = pd.DataFrame({"GLD": [1.0]}, index=prices.index[:1])
    with pytest.raises(ValueError, match=r"GLD.*2024-01-04"):
        simulate_weights(prices, targets, make_settings())


def test_run_backtest_inherits_the_same_price_guard() -> None:
    prices, _, _ = market(4)
    prices.loc[prices.index[2], "GLD"] = np.nan
    settings = make_settings()
    settings = settings.model_copy(
        update={
            "backtest": settings.backtest.model_copy(
                update={"regime_weights": {MarketRegime.BULL_TREND: {"GLD": 1.0}}}
            )
        }
    )
    regimes = pd.Series(MarketRegime.BULL_TREND, index=prices.index, dtype="object")
    with pytest.raises(ValueError, match=r"GLD.*2024-01-04"):
        run_backtest(prices, regimes, settings)


def test_nonnumeric_close_prices_raise_a_clear_error() -> None:
    prices, _, _ = market(3)
    prices["GLD"] = "unavailable"
    targets = pd.DataFrame({"GLD": [1.0]}, index=prices.index[:1])
    with pytest.raises(ValueError, match="close prices must be numeric"):
        simulate_weights(prices, targets, make_settings())
