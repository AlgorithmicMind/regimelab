"""Independent accounting oracles frozen before the ledger implementation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from regimelab.backtest.portfolio import simulate_next_open
from tests._next_open_fixtures import make_settings, market


def test_buy_does_not_capture_the_pre_purchase_gap() -> None:
    prices, opens, sessions = market(3)
    prices["SPY"] = [100.0, 104.0, 104.0]
    opens["SPY"] = [100.0, 103.0, 104.0]
    targets = pd.DataFrame({"SPY": [1.0]}, index=prices.index[:1])
    result = simulate_next_open(
        prices, targets, make_settings(), open_prices=opens, sessions=sessions
    )
    assert result.net_returns.iloc[0] == pytest.approx(1 / 103)
    assert result.positions["SPY"].iloc[0] == pytest.approx(1000 / 103)
    assert result.nav.iloc[0] == pytest.approx(1000 * 104 / 103)
    pd.testing.assert_series_equal(result.net_returns, result.benchmark_returns, check_names=False)


def test_sell_incurs_overnight_loss_but_not_post_sale_loss() -> None:
    prices, opens, sessions = market(4)
    prices["SPY"] = [100.0, 100.0, 80.0, 80.0]
    opens["SPY"] = [100.0, 100.0, 90.0, 80.0]
    targets = pd.DataFrame({"SPY": [1.0, 0.0]}, index=prices.index[:2])
    result = simulate_next_open(
        prices, targets, make_settings(capital=100), open_prices=opens, sessions=sessions
    )
    np.testing.assert_allclose(result.nav, [100.0, 90.0, 90.0])
    np.testing.assert_allclose(result.net_returns, [0.0, -0.1, 0.0], atol=1e-15)
    assert result.positions["SPY"].iloc[1] == 0.0
    assert result.cash_balance.iloc[1] == 90.0
    assert result.traded_units["SPY"].iloc[1] == -1.0


def test_repeated_targets_preserve_units_and_allow_weight_drift() -> None:
    prices, opens, sessions = market(4)
    prices["SPY"] = [100.0, 100.0, 120.0, 120.0]
    opens["SPY"] = [100.0, 100.0, 100.0, 120.0]
    targets = pd.DataFrame({"SPY": 0.5, "GLD": 0.5}, index=prices.index[:-1])
    result = simulate_next_open(
        prices, targets, make_settings(capital=200), open_prices=opens, sessions=sessions
    )
    np.testing.assert_allclose(result.positions, 1.0)
    np.testing.assert_allclose(result.nav, [200.0, 220.0, 220.0])
    np.testing.assert_allclose(result.closing_weights.iloc[1], [6 / 11, 5 / 11, 0.0])
    assert (result.traded_units.iloc[1:] == 0.0).all(axis=None)


def test_buy_fees_are_financed_without_negative_cash_or_hidden_initial_peak() -> None:
    prices, opens, sessions = market(3)
    targets = pd.DataFrame({"SPY": [1.0]}, index=prices.index[:1])
    result = simulate_next_open(
        prices, targets, make_settings(cost=100), open_prices=opens, sessions=sessions
    )
    expected_nav = 100000 / 101
    assert result.nav.iloc[0] == pytest.approx(expected_nav)
    assert result.fees.iloc[0] == pytest.approx(1000 / 101)
    assert result.cash_balance.iloc[0] == pytest.approx(0.0, abs=1e-10)
    assert (result.cash_balance >= 0.0).all()
    assert result.drawdown.iloc[0] == pytest.approx(-1 / 101)
    assert result.metrics.max_drawdown == pytest.approx(-1 / 101)
    assert result.traded_notional.iloc[0] == pytest.approx(expected_nav / 1000)
    assert result.costs.iloc[0] == pytest.approx(1 / 101)
    assert result.fees.iloc[-1] == 0.0
    assert result.positions["SPY"].iloc[-1] > 0.0


def test_finite_large_capital_does_not_overflow_the_rebalance_midpoint() -> None:
    prices, opens, sessions = market(3)
    targets = pd.DataFrame({"SPY": [1.0]}, index=prices.index[:1])
    result = simulate_next_open(
        prices,
        targets,
        make_settings(capital=1e308, cost=100),
        open_prices=opens,
        sessions=sessions,
    )
    assert result.nav.iloc[0] == pytest.approx(1e308 / 1.01)
    assert result.opening_weights["SPY"].iloc[0] == pytest.approx(1.0, abs=1e-12)
    assert result.opening_weights["CASH"].iloc[0] <= 1e-12


def test_sell_fee_counts_the_asset_sale_not_the_cash_receipt() -> None:
    prices, opens, sessions = market(4)
    targets = pd.DataFrame({"SPY": [1.0, 0.0]}, index=prices.index[:2])
    result = simulate_next_open(
        prices,
        targets,
        make_settings(capital=1010, cost=100),
        open_prices=opens,
        sessions=sessions,
    )
    assert result.nav.iloc[0] == pytest.approx(1000)
    assert result.fees.iloc[1] == pytest.approx(10)
    assert result.cash_balance.iloc[1] == pytest.approx(990)
    assert result.traded_notional.iloc[1] == pytest.approx(1.0)
    assert result.costs.iloc[1] == pytest.approx(0.01)


@pytest.mark.parametrize(
    ("new_gld", "post_nav", "fee", "cash"),
    [(0.6, 499000 / 501, 2000 / 501, 0.0), (0.2, 495000 / 497, 2000 / 497, 198000 / 497)],
)
def test_rebalance_matches_independently_solved_rational_oracle(
    new_gld: float, post_nav: float, fee: float, cash: float
) -> None:
    prices, opens, sessions = market(4)
    targets = pd.DataFrame({"SPY": [0.6, 0.4], "GLD": [0.4, new_gld]}, index=prices.index[:2])
    result = simulate_next_open(
        prices,
        targets,
        make_settings(capital=1010, cost=100),
        open_prices=opens,
        sessions=sessions,
    )
    assert result.nav.iloc[0] == pytest.approx(1000)
    assert result.nav_after_trade.iloc[1] == pytest.approx(post_nav)
    assert result.fees.iloc[1] == pytest.approx(fee)
    assert result.cash_balance.iloc[1] == pytest.approx(cash, abs=1e-10)
    np.testing.assert_allclose(
        result.traded_values.iloc[1], [0.4 * post_nav - 600, new_gld * post_nav - 400]
    )


def test_cash_has_interest_without_synthetic_etf_trades() -> None:
    prices, opens, sessions = market(3)
    targets = pd.DataFrame(columns=["SPY"], dtype="float64")
    result = simulate_next_open(
        prices,
        targets,
        make_settings(cost=100, rate=0.02),
        open_prices=opens,
        sessions=sessions,
    )
    expected = 1000 * (1 + 0.02 / 252) ** np.arange(1, 4)
    np.testing.assert_allclose(result.nav, expected)
    assert (result.fees == 0.0).all()
    assert (result.traded_units == 0.0).all(axis=None)
    assert result.metrics.average_exposure == 0.0
    assert (result.opening_weights["CASH"] == 1.0).all()


def test_cash_created_at_the_open_first_accrues_next_session() -> None:
    prices, opens, sessions = market(4)
    targets = pd.DataFrame({"SPY": [1.0, 0.0]}, index=prices.index[:2])
    result = simulate_next_open(
        prices, targets, make_settings(rate=0.252), open_prices=opens, sessions=sessions
    )
    np.testing.assert_allclose(result.cash_interest, [1.0, 0.0, 1.001], atol=1e-10)


def test_all_currency_flows_reconcile_and_inputs_remain_unchanged() -> None:
    prices, opens, sessions = market(6)
    prices["SPY"] = [100.0, 105.0, 110.0, 108.0, 112.0, 111.0]
    prices["GLD"] = [100.0, 98.0, 103.0, 102.0, 104.0, 105.0]
    opens["SPY"] = [100.0, 101.0, 103.0, 110.0, 107.0, 113.0]
    opens["GLD"] = [100.0, 101.0, 99.0, 104.0, 101.0, 104.0]
    targets = pd.DataFrame(
        {"SPY": [0.6, 0.6, 0.2, 0.2, 0.7], "GLD": [0.3, 0.3, 0.5, 0.5, 0.0]},
        index=prices.index[:-1],
    )
    originals = [f.copy(deep=True) for f in (prices, opens, targets, sessions)]
    result = simulate_next_open(
        prices, targets, make_settings(cost=5, rate=0.02), open_prices=opens, sessions=sessions
    )
    cash, nav = 1000.0, 1000.0
    units = np.zeros(2)
    for session in result.nav.index:
        now_units = result.positions.loc[session].to_numpy()
        old_value = float(units @ opens.loc[session].to_numpy())
        assert result.nav_before_trade.loc[session] == pytest.approx(
            cash + result.cash_interest.loc[session] + old_value
        )
        signed = result.traded_values.loc[session]
        np.testing.assert_allclose(signed, (now_units - units) * opens.loc[session])
        fee = signed.abs().sum() * 0.0005
        cash += result.cash_interest.loc[session] - signed.sum() - fee
        assert result.cash_balance.loc[session] == pytest.approx(cash, abs=1e-9)
        assert result.fees.loc[session] == pytest.approx(fee)
        assert result.nav_after_trade.loc[session] == pytest.approx(
            result.nav_before_trade.loc[session] - fee
        )
        marked = float(now_units @ prices.loc[session].to_numpy()) + cash
        assert result.nav.loc[session] == pytest.approx(marked)
        assert result.net_returns.loc[session] == pytest.approx(marked / nav - 1)
        assert result.costs.loc[session] == pytest.approx(fee / nav)
        units, nav = now_units, marked
    for original, current in zip(originals, (prices, opens, targets, sessions), strict=True):
        pd.testing.assert_frame_equal(original, current)
    np.testing.assert_allclose(result.opening_weights.sum(axis=1), 1.0)
    np.testing.assert_allclose(result.closing_weights.sum(axis=1), 1.0)
    assert (result.positions >= 0).all(axis=None)
    assert (result.cash_balance >= 0).all()
    audit = result.to_frame()
    assert {"q_SPY", "w_open_SPY", "w_close_SPY", "scheduled_open", "has_fills"} <= set(audit)
    assert not audit["has_fills"].iloc[1]


def test_benchmark_reuses_identical_accounting_and_future_prices_do_not_change_history() -> None:
    prices, opens, sessions = market(5)
    prices["SPY"] = [100.0, 104.0, 108.0, 101.0, 109.0]
    opens["SPY"] = [100.0, 103.0, 105.0, 106.0, 102.0]
    targets = pd.DataFrame({"SPY": [1.0]}, index=prices.index[:1])
    settings = make_settings(cost=5, rate=0.02)
    original = simulate_next_open(prices, targets, settings, open_prices=opens, sessions=sessions)
    pd.testing.assert_series_equal(
        original.net_returns, original.benchmark_returns, check_names=False
    )
    assert original.metrics == original.benchmark_metrics
    changed = opens.copy()
    changed.iloc[-1, 0] = 150.0
    replay = simulate_next_open(prices, targets, settings, open_prices=changed, sessions=sessions)
    pd.testing.assert_series_equal(original.nav.iloc[:-1], replay.nav.iloc[:-1])
    pd.testing.assert_frame_equal(original.target_weights, replay.target_weights)
