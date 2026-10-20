"""Availability, exchange-session and observed-price guards for the ledger."""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from regimelab.backtest.portfolio import simulate_next_open
from regimelab.data.alignment import align_panel
from regimelab.data.providers import panel_from_symbol_frames
from tests._next_open_fixtures import make_settings, market


def test_instruction_at_the_open_is_not_filled_retroactively() -> None:
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
    )
    np.testing.assert_allclose(result.positions["SPY"], [0.0, 10.0, 10.0])
    assert pd.isna(result.decision_at_open.iloc[0])
    assert result.decision_at_open.iloc[1] == targets.index[0]


def test_delayed_older_decision_never_overrides_a_newer_observation() -> None:
    prices, opens, sessions = market(5)
    targets = pd.DataFrame({"SPY": [1.0, 0.2]}, index=prices.index[:2])
    published = pd.Series(
        [sessions["market_open"].iloc[2] + pd.Timedelta(hours=1), sessions["market_close"].iloc[1]],
        index=targets.index,
    )
    result = simulate_next_open(
        prices,
        targets,
        make_settings(),
        open_prices=opens,
        sessions=sessions,
        published_at=published,
    )
    np.testing.assert_allclose(result.positions["SPY"], [0.0, 2.0, 2.0, 2.0])
    assert (result.decision_at_open.iloc[1:] == targets.index[1]).all()


@pytest.mark.parametrize(
    "dates",
    [
        ["2024-01-12", "2024-01-16", "2024-01-17"],  # MLK holiday is not a price session.
        ["2024-03-08", "2024-03-11", "2024-03-12"],  # US daylight-saving transition.
        ["2024-11-29", "2024-12-02", "2024-12-03"],  # Short Friday, then a weekend.
    ],
)
def test_calendar_not_weekdays_controls_the_next_open(dates: list[str]) -> None:
    prices, opens, sessions = market(dates=dates)
    if dates[0] == "2024-11-29":
        sessions.loc[prices.index[0], "market_close"] = pd.Timestamp(
            "2024-11-29 13:00", tz="America/New_York"
        ).tz_convert("UTC")
    targets = pd.DataFrame({"SPY": [1.0]}, index=prices.index[:1])
    result = simulate_next_open(
        prices, targets, make_settings(), open_prices=opens, sessions=sessions
    )
    assert result.nav.index[0] == prices.index[1]
    assert result.execution_times.iloc[0] == sessions["market_open"].iloc[1]
    assert result.positions["SPY"].iloc[0] == 10.0
    if dates[0] == "2024-03-08":
        assert result.execution_times.iloc[0].hour == 13


def test_explicit_start_is_cash_inception_not_a_crop_of_prior_holdings() -> None:
    prices, opens, sessions = market(7)
    targets = pd.DataFrame({"SPY": [0.5]}, index=prices.index[:1])
    result = simulate_next_open(
        prices,
        targets,
        make_settings(),
        open_prices=opens,
        sessions=sessions,
        start=dt.date(2024, 1, 6),
    )
    assert result.nav.index[0] == pd.Timestamp("2024-01-08")
    assert result.positions["SPY"].iloc[0] == 5.0
    assert result.nav.iloc[0] == 1000.0
    assert result.decision_at_open.iloc[0] == targets.index[0]


@pytest.mark.parametrize("mark", ["open", "close"])
@pytest.mark.parametrize("bad_price", [np.nan, np.inf, 0.0, -1.0])
def test_used_missing_or_invalid_prices_raise(mark: str, bad_price: float) -> None:
    prices, opens, sessions = market(3)
    target = pd.DataFrame({"SPY": [1.0]}, index=prices.index[:1])
    frame = opens if mark == "open" else prices
    frame.loc[prices.index[1], "SPY"] = bad_price
    with pytest.raises(ValueError, match=r"SPY.*2024-01-03"):
        simulate_next_open(prices, target, make_settings(), open_prices=opens, sessions=sessions)


@pytest.mark.parametrize("mark", ["open", "close"])
def test_finite_but_unobserved_prices_are_not_usable(mark: str) -> None:
    prices, opens, sessions = market(3)
    target = pd.DataFrame({"SPY": [1.0]}, index=prices.index[:1])
    observed = pd.DataFrame(True, index=prices.index, columns=prices.columns)
    observed.loc[prices.index[1], "SPY"] = False
    masks = {f"{mark}_observed": observed}
    with pytest.raises(ValueError, match="unobserved"):
        simulate_next_open(
            prices, target, make_settings(), open_prices=opens, sessions=sessions, **masks
        )


def test_unused_missing_asset_does_not_poison_pnl() -> None:
    prices, opens, sessions = market(3)
    prices["GLD"] = np.nan
    opens["GLD"] = np.nan
    observed = pd.DataFrame({"SPY": True, "GLD": False}, index=prices.index)
    target = pd.DataFrame({"SPY": [1.0]}, index=prices.index[:1])
    result = simulate_next_open(
        prices,
        target,
        make_settings(),
        open_prices=opens,
        sessions=sessions,
        open_observed=observed,
        close_observed=observed,
    )
    np.testing.assert_allclose(result.net_returns, 0.0)
    assert (result.positions["GLD"] == 0.0).all()
    assert (result.closing_weights["GLD"] == 0.0).all()


def test_fully_sold_asset_does_not_need_a_later_closing_valuation() -> None:
    prices, opens, sessions = market(4)
    prices.loc[prices.index[2] :, "GLD"] = np.nan
    observed = prices.notna()
    targets = pd.DataFrame({"GLD": [1.0, 0.0]}, index=prices.index[:2])
    result = simulate_next_open(
        prices,
        targets,
        make_settings(),
        open_prices=opens,
        sessions=sessions,
        close_observed=observed,
    )
    np.testing.assert_allclose(result.nav, 1000.0)
    np.testing.assert_allclose(result.positions["GLD"], [10.0, 0.0, 0.0])


@pytest.mark.parametrize(
    "weights",
    [
        {"SPY": [np.nan]},
        {"SPY": [np.inf]},
        {"SPY": [-0.1]},
        {"SPY": [1.1]},
        {"SPY": [0.6], "GLD": [0.5]},
        {"SPY": [0.5], "CASH": [0.1]},
        {"UNKNOWN": [0.1]},
    ],
)
def test_invalid_weights_are_rejected(weights: dict[str, list[float]]) -> None:
    prices, opens, sessions = market(3)
    target = pd.DataFrame(weights, index=prices.index[:1])
    with pytest.raises(ValueError, match=r"weights|CASH|unpriced"):
        simulate_next_open(prices, target, make_settings(), open_prices=opens, sessions=sessions)


def test_bookkeeping_tolerance_is_normalized_without_leverage() -> None:
    prices, opens, sessions = market(3)
    targets = pd.DataFrame({"SPY": [0.6], "GLD": [0.4 + 5e-10]}, index=prices.index[:1])
    result = simulate_next_open(
        prices, targets, make_settings(cost=5), open_prices=opens, sessions=sessions
    )
    assert (result.cash_balance >= 0.0).all()
    np.testing.assert_allclose(result.target_weights.sum(axis=1), 1.0, atol=1e-12)
    assert (result.target_weights >= 0.0).all(axis=None)


@pytest.mark.parametrize("kind", ["duplicate", "unsorted", "intraday", "aware"])
def test_ambiguous_price_indexes_are_rejected(kind: str) -> None:
    prices, opens, sessions = market(4)
    index = prices.index
    if kind == "duplicate":
        changed = pd.DatetimeIndex([index[0], index[0], index[2], index[3]])
    elif kind == "unsorted":
        changed = index[::-1]
    elif kind == "intraday":
        changed = index + pd.Timedelta(minutes=1)
    else:
        changed = index.tz_localize("UTC")
    prices.index = changed
    target = pd.DataFrame({"SPY": [1.0]}, index=index[:1])
    with pytest.raises(ValueError, match=r"sessions|session index"):
        simulate_next_open(prices, target, make_settings(), open_prices=opens, sessions=sessions)


@pytest.mark.parametrize("kind", ["missing_session", "naive", "close_before_open", "overlap"])
def test_invalid_exchange_schedules_are_rejected(kind: str) -> None:
    prices, opens, sessions = market(4)
    if kind == "missing_session":
        sessions = sessions.iloc[1:]
    elif kind == "naive":
        sessions["market_open"] = sessions["market_open"].dt.tz_localize(None)
    elif kind == "close_before_open":
        sessions["market_close"] = sessions["market_open"] - pd.Timedelta(minutes=1)
    else:
        sessions.loc[prices.index[0], "market_close"] = sessions["market_open"].iloc[1]
    target = pd.DataFrame({"SPY": [1.0]}, index=prices.index[:1])
    with pytest.raises(ValueError, match=r"session|market_open|timestamps"):
        simulate_next_open(prices, target, make_settings(), open_prices=opens, sessions=sessions)


@pytest.mark.parametrize("kind", ["naive", "missing", "before_close", "wrong_index"])
def test_invalid_publication_evidence_is_rejected(kind: str) -> None:
    prices, opens, sessions = market(3)
    target = pd.DataFrame({"SPY": [1.0]}, index=prices.index[:1])
    published = sessions["market_close"].iloc[:1].copy()
    if kind == "naive":
        published = published.dt.tz_localize(None)
    elif kind == "missing":
        published.iloc[0] = pd.NaT
    elif kind == "before_close":
        published -= pd.Timedelta(seconds=1)
    else:
        published.index = prices.index[1:2]
    with pytest.raises(ValueError, match="published_at"):
        simulate_next_open(
            prices,
            target,
            make_settings(),
            open_prices=opens,
            sessions=sessions,
            published_at=published,
        )


def test_old_lag_two_configuration_cannot_silently_select_next_open() -> None:
    prices, opens, sessions = market(3)
    target = pd.DataFrame({"SPY": [1.0]}, index=prices.index[:1])
    settings = make_settings()
    settings = settings.model_copy(
        update={"backtest": settings.backtest.model_copy(update={"signal_lag_days": 2})}
    )
    with pytest.raises(ValueError, match="signal_lag_days=1"):
        simulate_next_open(prices, target, settings, open_prices=opens, sessions=sessions)


@pytest.mark.parametrize("kind", ["duplicate", "unsorted", "outside"])
def test_invalid_decision_dates_are_rejected(kind: str) -> None:
    prices, opens, sessions = market(4)
    dates = prices.index[:2]
    if kind == "duplicate":
        dates = pd.DatetimeIndex([dates[0], dates[0]])
    elif kind == "unsorted":
        dates = dates[::-1]
    else:
        dates = pd.DatetimeIndex(["2025-01-02", "2025-01-03"])
    targets = pd.DataFrame({"SPY": 0.5}, index=dates)
    with pytest.raises(ValueError, match=r"target_weights|sessions"):
        simulate_next_open(prices, targets, make_settings(), open_prices=opens, sessions=sessions)


def test_unpriced_asset_is_allowed_only_when_unused() -> None:
    prices, opens, sessions = market(3)
    prices = prices.drop(columns="GLD")
    opens = opens.drop(columns="GLD")
    targets = pd.DataFrame({"GLD": [0.0]}, index=prices.index[:1])
    result = simulate_next_open(
        prices, targets, make_settings(), open_prices=opens, sessions=sessions
    )
    np.testing.assert_allclose(result.nav, 1000.0)
    assert "GLD" not in result.positions.columns
    targets["GLD"] = 1e-10
    with pytest.raises(ValueError, match="unpriced"):
        simulate_next_open(prices, targets, make_settings(), open_prices=opens, sessions=sessions)


@pytest.mark.parametrize("kind", ["missing_flags", "nonboolean", "wrong_index"])
def test_ambiguous_observation_masks_are_rejected(kind: str) -> None:
    prices, opens, sessions = market(3)
    targets = pd.DataFrame({"SPY": [1.0]}, index=prices.index[:1])
    mask = pd.DataFrame(True, index=prices.index, columns=prices.columns, dtype="boolean")
    if kind == "missing_flags":
        mask.iloc[0, 0] = pd.NA
    elif kind == "nonboolean":
        mask = mask.astype(int)
    else:
        mask = mask.iloc[1:]
    with pytest.raises(ValueError, match="open_observed"):
        simulate_next_open(
            prices,
            targets,
            make_settings(),
            open_prices=opens,
            sessions=sessions,
            open_observed=mask,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("transaction_cost_bps", np.inf),
        ("transaction_cost_bps", 10000.0),
        ("transaction_cost_bps", -1.0),
        ("initial_capital", np.inf),
        ("initial_capital", 0.0),
        ("risk_free_rate", np.inf),
        ("risk_free_rate", -1.0),
    ],
)
def test_invalid_financing_configuration_is_rejected(field: str, value: float) -> None:
    prices, opens, sessions = market(3)
    targets = pd.DataFrame({"SPY": [1.0]}, index=prices.index[:1])
    settings = make_settings()
    settings = settings.model_copy(
        update={"backtest": settings.backtest.model_copy(update={field: value})}
    )
    with pytest.raises(ValueError, match=r"cost rate|initial_capital|risk_free_rate"):
        simulate_next_open(prices, targets, settings, open_prices=opens, sessions=sessions)


def test_at_least_two_evaluated_sessions_are_required() -> None:
    prices, opens, sessions = market(3)
    targets = pd.DataFrame({"SPY": [1.0]}, index=prices.index[1:2])
    with pytest.raises(ValueError, match="two evaluated sessions"):
        simulate_next_open(prices, targets, make_settings(), open_prices=opens, sessions=sessions)


def test_expired_aligned_prices_cannot_create_silent_zero_returns() -> None:
    prices, opens, sessions = market(10)
    frames = {
        symbol: pd.DataFrame(
            {
                "open": opens[symbol],
                "high": prices[symbol],
                "low": prices[symbol],
                "close": prices[symbol],
            },
            index=prices.index,
        )
        for symbol in prices.columns
    }
    frames["GLD"].iloc[2:] = np.nan
    aligned = align_panel(panel_from_symbol_frames(frames), anchor="SPY")
    targets = pd.DataFrame({"GLD": [1.0]}, index=prices.index[:1])
    with pytest.raises(ValueError, match=r"opening prices.*GLD.*2024-01-11"):
        simulate_next_open(
            aligned.close(),
            targets,
            make_settings(),
            open_prices=aligned.field("open"),
            sessions=sessions,
        )


def test_unrepresentable_returns_fail_instead_of_creating_invalid_metrics() -> None:
    prices, opens, sessions = market(4)
    prices["GLD"] = [100.0, 1e-200, 1e200, 1e200]
    opens["GLD"] = [100.0, 1000.0, 1e-200, 1e200]
    targets = pd.DataFrame({"GLD": [1.0]}, index=prices.index[:1])
    with pytest.raises(ValueError, match="unrepresentable portfolio returns"):
        simulate_next_open(prices, targets, make_settings(), open_prices=opens, sessions=sessions)
