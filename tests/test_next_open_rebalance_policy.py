"""Prespecified synthetic oracles for explicit, not carried, rebalance requests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from regimelab.backtest import portfolio
from regimelab.backtest.portfolio import simulate_next_open
from tests._next_open_fixtures import make_settings, market


def drift_market() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prices, opens, sessions = market(4)
    prices["SPY"] = [100.0, 120.0, 132.0, 132.0]
    prices["GLD"] = [100.0, 100.0, 90.0, 90.0]
    opens["SPY"] = [100.0, 100.0, 120.0, 132.0]
    opens["GLD"] = [100.0, 100.0, 100.0, 90.0]
    return prices, opens, sessions


def test_explicit_request_restores_weights_but_default_preserves_units() -> None:
    prices, opens, sessions = drift_market()
    targets = pd.DataFrame({"SPY": 0.5, "GLD": 0.5}, index=prices.index[:2])
    settings = make_settings(capital=200)
    default = simulate_next_open(prices, targets, settings, open_prices=opens, sessions=sessions)
    explicit_default = simulate_next_open(
        prices,
        targets,
        settings,
        open_prices=opens,
        sessions=sessions,
        rebalance_policy="target_change",
    )
    restored = simulate_next_open(
        prices,
        targets,
        settings,
        open_prices=opens,
        sessions=sessions,
        rebalance_policy="each_instruction",
    )
    assert default.rebalance_policy == explicit_default.rebalance_policy == "target_change"
    assert restored.rebalance_policy == "each_instruction"
    pd.testing.assert_frame_equal(default.to_frame(), explicit_default.to_frame(), check_exact=True)
    assert default.metrics == explicit_default.metrics
    np.testing.assert_allclose(default.nav, [220.0, 222.0, 222.0], rtol=0, atol=1e-10)
    np.testing.assert_allclose(default.positions, 1.0, rtol=0, atol=1e-12)
    np.testing.assert_allclose(default.traded_units.iloc[1:], 0.0, rtol=0, atol=0)
    np.testing.assert_allclose(restored.nav, [220.0, 220.0, 220.0], rtol=0, atol=1e-10)
    np.testing.assert_allclose(
        restored.positions.iloc[1:], [[11 / 12, 11 / 10]] * 2, rtol=0, atol=1e-12
    )
    np.testing.assert_allclose(restored.traded_units.iloc[1], [-1 / 12, 1 / 10], rtol=0, atol=1e-12)
    np.testing.assert_allclose(restored.traded_values.iloc[1], [-10.0, 10.0], rtol=0, atol=1e-10)
    np.testing.assert_allclose(restored.traded_units.iloc[-1], 0.0, rtol=0, atol=0)
    np.testing.assert_allclose(
        restored.opening_weights.iloc[1], [0.5, 0.5, 0.0], rtol=0, atol=1e-12
    )
    pd.testing.assert_series_equal(
        default.benchmark_returns, restored.benchmark_returns, check_exact=True
    )
    pd.testing.assert_series_equal(
        default.benchmark_equity, restored.benchmark_equity, check_exact=True
    )
    assert default.benchmark_metrics == restored.benchmark_metrics
    assert restored.decision_at_open.iloc[-1] == targets.index[-1]


def test_carried_target_is_not_a_new_request() -> None:
    prices, opens, sessions = drift_market()
    targets = pd.DataFrame({"SPY": [0.5], "GLD": [0.5]}, index=prices.index[:1])
    result = simulate_next_open(
        prices,
        targets,
        make_settings(capital=200),
        open_prices=opens,
        sessions=sessions,
        rebalance_policy="each_instruction",
    )
    np.testing.assert_allclose(result.nav, [220.0, 222.0, 222.0], rtol=0, atol=1e-10)
    np.testing.assert_allclose(result.positions, 1.0, rtol=0, atol=1e-12)
    np.testing.assert_allclose(result.traded_units.iloc[1:], 0.0, rtol=0, atol=0)
    assert result.decision_at_open.eq(targets.index[0]).all()


def test_last_close_decision_cannot_create_an_unavailable_future_fill() -> None:
    prices, opens, sessions = drift_market()
    targets = pd.DataFrame(
        {"SPY": [0.5, 1.0], "GLD": [0.5, 0.0]},
        index=prices.index[[0, -1]],
    )
    result = simulate_next_open(
        prices,
        targets,
        make_settings(capital=200),
        open_prices=opens,
        sessions=sessions,
        rebalance_policy="each_instruction",
    )
    np.testing.assert_allclose(result.nav, [220.0, 222.0, 222.0], rtol=0, atol=1e-10)
    np.testing.assert_allclose(result.positions, 1.0, rtol=0, atol=1e-12)
    np.testing.assert_allclose(result.traded_units.iloc[1:], 0.0, rtol=0, atol=0)
    assert result.decision_at_open.eq(targets.index[0]).all()


def test_already_matching_positions_generate_no_phantom_fills_or_fees() -> None:
    prices, opens, sessions = market(5)
    targets = pd.DataFrame({"SPY": 0.5, "GLD": 0.5}, index=prices.index[:-1])
    result = simulate_next_open(
        prices,
        targets,
        make_settings(capital=202, cost=100),
        open_prices=opens,
        sessions=sessions,
        rebalance_policy="each_instruction",
    )
    np.testing.assert_allclose(result.nav, 200.0, rtol=0, atol=1e-10)
    assert result.fees.iloc[0] == pytest.approx(2.0, abs=1e-10)
    np.testing.assert_allclose(result.traded_units.iloc[1:], 0.0, rtol=0, atol=0)
    np.testing.assert_allclose(result.fees.iloc[1:], 0.0, rtol=0, atol=0)
    for row in range(1, len(result.positions)):
        pd.testing.assert_series_equal(
            result.positions.iloc[0],
            result.positions.iloc[row],
            check_names=False,
            check_exact=True,
        )


def test_changed_target_already_matches_actual_drifted_exposure() -> None:
    prices, opens, sessions = market(3)
    prices["SPY"] = [100.0, 200.0, 200.0]
    opens["SPY"] = [100.0, 100.0, 200.0]
    targets = pd.DataFrame({"SPY": [0.5, 2 / 3]}, index=prices.index[:2])
    result = simulate_next_open(
        prices,
        targets,
        make_settings(capital=201, cost=100),
        open_prices=opens,
        sessions=sessions,
        rebalance_policy="each_instruction",
    )
    np.testing.assert_allclose(result.nav, 300.0, rtol=0, atol=1e-10)
    np.testing.assert_allclose(result.positions["SPY"], 1.0, rtol=0, atol=1e-12)
    assert result.fees.iloc[0] == pytest.approx(1.0, abs=1e-10)
    assert result.fees.iloc[1] == 0.0
    np.testing.assert_allclose(result.traded_units.iloc[1], 0.0, rtol=0, atol=0)


def test_restoration_reuses_per_leg_self_financing_cost_equation() -> None:
    prices, opens, sessions = drift_market()
    targets = pd.DataFrame({"SPY": 0.5, "GLD": 0.5}, index=prices.index[:2])
    result = simulate_next_open(
        prices,
        targets,
        make_settings(capital=202, cost=100),
        open_prices=opens,
        sessions=sessions,
        rebalance_policy="each_instruction",
    )
    # Existing values 120/100 -> 109.9/109.9: sell 10.1, buy 9.9, fee 0.2.
    np.testing.assert_allclose(result.nav, [220.0, 219.8, 219.8], rtol=0, atol=1e-10)
    np.testing.assert_allclose(result.fees, [2.0, 0.2, 0.0], rtol=0, atol=1e-10)
    np.testing.assert_allclose(result.traded_values.iloc[1], [-10.1, 9.9], rtol=0, atol=1e-10)
    np.testing.assert_allclose(
        result.positions.iloc[1], [1099 / 1200, 1099 / 1000], rtol=0, atol=1e-12
    )
    assert result.nav_after_trade.iloc[1] == pytest.approx(219.8, abs=1e-10)
    assert result.traded_notional.iloc[1] == pytest.approx(1 / 11, abs=1e-12)
    assert result.costs.iloc[1] == pytest.approx(1 / 1100, abs=1e-12)
    assert result.net_returns.iloc[1] == pytest.approx(219.8 / 220 - 1, abs=1e-12)
    np.testing.assert_allclose(result.cash_balance, 0.0, rtol=0, atol=1e-10)
    assert result.cash_balance.ge(0.0).all()
    assert result.positions.ge(0.0).all(axis=None)
    np.testing.assert_allclose(
        result.nav_after_trade + result.fees, result.nav_before_trade, rtol=0, atol=1e-10
    )


def test_cash_interest_is_credited_before_a_requested_restoration() -> None:
    prices, opens, sessions = market(3)
    targets = pd.DataFrame({"SPY": 0.5}, index=prices.index[:2])
    result = simulate_next_open(
        prices,
        targets,
        make_settings(capital=200, rate=2.52),
        open_prices=opens,
        sessions=sessions,
        rebalance_policy="each_instruction",
    )
    np.testing.assert_allclose(result.cash_interest, [2.0, 1.01], rtol=0, atol=1e-10)
    np.testing.assert_allclose(result.nav, [202.0, 203.01], rtol=0, atol=1e-10)
    np.testing.assert_allclose(result.positions["SPY"], [1.01, 1.01505], rtol=0, atol=1e-12)
    np.testing.assert_allclose(result.cash_balance, [101.0, 101.505], rtol=0, atol=1e-10)
    assert result.traded_values["SPY"].iloc[1] == pytest.approx(0.505, abs=1e-10)


def test_cash_only_requests_do_not_manufacture_risky_trades() -> None:
    prices, opens, sessions = market(4)
    targets = pd.DataFrame({"CASH": 1.0}, index=prices.index[:-1])
    result = simulate_next_open(
        prices,
        targets,
        make_settings(cost=100, rate=0.252),
        open_prices=opens,
        sessions=sessions,
        rebalance_policy="each_instruction",
    )
    np.testing.assert_allclose(result.nav, 1000 * 1.001 ** np.arange(1, 4), rtol=0, atol=1e-10)
    np.testing.assert_allclose(result.traded_units, 0.0, rtol=0, atol=0)
    np.testing.assert_allclose(result.fees, 0.0, rtol=0, atol=0)


def test_no_decisions_retains_cash_without_inventing_requests() -> None:
    prices, opens, sessions = market(4)
    targets = pd.DataFrame(columns=["SPY"], dtype="float64")
    result = simulate_next_open(
        prices,
        targets,
        make_settings(cost=100, rate=0.252),
        open_prices=opens,
        sessions=sessions,
        rebalance_policy="each_instruction",
    )
    np.testing.assert_allclose(result.nav, 1000 * 1.001 ** np.arange(1, 5), rtol=0, atol=1e-10)
    np.testing.assert_allclose(result.traded_units, 0.0, rtol=0, atol=0)
    np.testing.assert_allclose(result.fees, 0.0, rtol=0, atol=0)
    assert result.decision_at_open.isna().all()


def test_only_newest_eligible_observation_authorizes_a_request() -> None:
    prices, opens, sessions = market(6)
    prices["SPY"] = [100.0, 100.0, 120.0, 120.0, 120.0, 120.0]
    opens["SPY"] = [100.0, 100.0, 100.0, 120.0, 120.0, 120.0]
    targets = pd.DataFrame({"SPY": 0.5, "GLD": 0.5}, index=prices.index[:2])
    published = pd.Series(
        [
            sessions["market_open"].iloc[3] + pd.Timedelta(hours=1),
            sessions["market_close"].iloc[1],
        ],
        index=targets.index,
    )
    result = simulate_next_open(
        prices,
        targets,
        make_settings(capital=200),
        open_prices=opens,
        sessions=sessions,
        published_at=published,
        rebalance_policy="each_instruction",
    )
    assert pd.isna(result.decision_at_open.iloc[0])
    assert result.decision_at_open.iloc[1:].eq(targets.index[1]).all()
    np.testing.assert_allclose(result.positions.iloc[0], 0.0, rtol=0, atol=0)
    np.testing.assert_allclose(result.positions.iloc[1:], 1.0, rtol=0, atol=1e-12)
    np.testing.assert_allclose(result.traded_units.iloc[2:], 0.0, rtol=0, atol=0)


def test_publication_at_the_open_cannot_receive_that_fill() -> None:
    prices, opens, sessions = market(4)
    targets = pd.DataFrame({"SPY": [1.0]}, index=prices.index[:1])
    published = pd.Series([sessions["market_open"].iloc[1]], index=targets.index)
    result = simulate_next_open(
        prices,
        targets,
        make_settings(),
        open_prices=opens,
        sessions=sessions,
        published_at=published,
        rebalance_policy="each_instruction",
    )
    np.testing.assert_allclose(result.positions["SPY"], [0.0, 10.0, 10.0], rtol=0, atol=0)
    np.testing.assert_allclose(result.traded_units["SPY"], [0.0, 10.0, 0.0], rtol=0, atol=0)
    assert pd.isna(result.decision_at_open.iloc[0])
    assert result.decision_at_open.iloc[1] == targets.index[0]


def test_inception_adopts_a_preexisting_eligible_observation_once() -> None:
    prices, opens, sessions = drift_market()
    targets = pd.DataFrame({"SPY": [0.5], "GLD": [0.5]}, index=prices.index[:1])
    result = simulate_next_open(
        prices,
        targets,
        make_settings(capital=220),
        open_prices=opens,
        sessions=sessions,
        start=prices.index[2].date(),
        rebalance_policy="each_instruction",
    )
    np.testing.assert_allclose(result.positions, [[11 / 12, 11 / 10]] * 2, rtol=0, atol=1e-12)
    np.testing.assert_allclose(result.nav, 220.0, rtol=0, atol=1e-10)
    np.testing.assert_allclose(result.traded_units.iloc[1], 0.0, rtol=0, atol=0)
    assert result.decision_at_open.eq(targets.index[0]).all()


@pytest.mark.parametrize("mark", ["open", "close"])
@pytest.mark.parametrize("bad", ["missing", "unobserved"])
def test_each_instruction_preserves_required_price_provenance(mark: str, bad: str) -> None:
    prices, opens, sessions = drift_market()
    targets = pd.DataFrame({"SPY": 0.5, "GLD": 0.5}, index=prices.index[:2])
    open_mask = pd.DataFrame(True, index=prices.index, columns=prices.columns)
    close_mask = open_mask.copy(deep=True)
    if bad == "missing":
        frame = opens if mark == "open" else prices
        frame.loc[prices.index[2], "GLD"] = np.nan
    else:
        mask = open_mask if mark == "open" else close_mask
        mask.loc[prices.index[2], "GLD"] = False
    with pytest.raises(ValueError, match=r"GLD.*2024-01-04"):
        simulate_next_open(
            prices,
            targets,
            make_settings(capital=200),
            open_prices=opens,
            sessions=sessions,
            open_observed=open_mask,
            close_observed=close_mask,
            rebalance_policy="each_instruction",
        )


def test_each_instruction_ignores_unused_missing_asset() -> None:
    prices, opens, sessions = market(4)
    prices["GLD"] = np.nan
    opens["GLD"] = np.nan
    mask = pd.DataFrame({"SPY": True, "GLD": False}, index=prices.index)
    targets = pd.DataFrame({"SPY": 1.0}, index=prices.index[:-1])
    result = simulate_next_open(
        prices,
        targets,
        make_settings(),
        open_prices=opens,
        sessions=sessions,
        open_observed=mask,
        close_observed=mask,
        rebalance_policy="each_instruction",
    )
    np.testing.assert_allclose(result.nav, 1000.0, rtol=0, atol=0)
    np.testing.assert_allclose(result.positions["GLD"], 0.0, rtol=0, atol=0)
    np.testing.assert_allclose(result.traded_units.iloc[1:], 0.0, rtol=0, atol=0)


def test_each_instruction_does_not_value_a_fully_sold_asset_at_the_close() -> None:
    prices, opens, sessions = market(4)
    prices.loc[prices.index[2] :, "GLD"] = np.nan
    targets = pd.DataFrame({"GLD": [1.0, 0.0]}, index=prices.index[:2])
    result = simulate_next_open(
        prices,
        targets,
        make_settings(),
        open_prices=opens,
        sessions=sessions,
        close_observed=prices.notna(),
        rebalance_policy="each_instruction",
    )
    np.testing.assert_allclose(result.nav, 1000.0, rtol=0, atol=0)
    np.testing.assert_allclose(result.positions["GLD"], [10.0, 0.0, 0.0], rtol=0, atol=0)


def test_future_prices_and_decisions_do_not_change_prior_fills() -> None:
    prices, opens, sessions = market(6)
    prices["SPY"] = [100.0, 120.0, 132.0, 132.0, 132.0, 132.0]
    prices["GLD"] = [100.0, 100.0, 90.0, 90.0, 90.0, 90.0]
    opens["SPY"] = [100.0, 100.0, 120.0, 132.0, 132.0, 132.0]
    opens["GLD"] = [100.0, 100.0, 100.0, 90.0, 90.0, 90.0]
    targets = pd.DataFrame({"SPY": 0.5, "GLD": 0.5}, index=prices.index[:2])
    settings = make_settings(capital=200, cost=5, rate=0.02)
    prefix = simulate_next_open(
        prices.iloc[:3],
        targets,
        settings,
        open_prices=opens.iloc[:3],
        sessions=sessions.iloc[:3],
        rebalance_policy="each_instruction",
    )
    prices.iloc[3:, :] *= 7.0
    opens.iloc[3:, :] *= 11.0
    targets.loc[prices.index[3]] = [0.9, 0.1]
    originals = [frame.copy(deep=True) for frame in (prices, opens, targets, sessions)]
    replay = simulate_next_open(
        prices,
        targets,
        settings,
        open_prices=opens,
        sessions=sessions,
        rebalance_policy="each_instruction",
    )
    pd.testing.assert_frame_equal(prefix.to_frame(), replay.to_frame().iloc[:2], check_exact=True)
    for before, after in zip(originals, (prices, opens, targets, sessions), strict=True):
        pd.testing.assert_frame_equal(before, after, check_exact=True)


@pytest.mark.parametrize(
    "bad", ["daily", "", None, True, 1, np.array(["target_change"]), np.array(["each_instruction"])]
)
def test_invalid_policy_rejected_before_input_preparation(
    bad: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("invalid policy must not reach input preparation")

    monkeypatch.setattr(portfolio, "prepare_inputs", fail)
    empty = pd.DataFrame()
    with pytest.raises(ValueError, match="rebalance_policy"):
        simulate_next_open(
            empty,
            empty,
            make_settings(),
            open_prices=empty,
            sessions=empty,
            rebalance_policy=bad,  # type: ignore[arg-type]
        )
