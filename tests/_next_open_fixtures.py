"""Small hand-specified inputs, with no provider or exchange-network access."""

from __future__ import annotations

import pandas as pd

from regimelab.settings import BacktestSettings, DataSettings, Settings


def make_settings(*, capital: float = 1000.0, cost: float = 0.0, rate: float = 0.0) -> Settings:
    return Settings(
        _env_file=None,
        data=DataSettings(provider="synthetic", risk_assets=("SPY", "GLD")),
        backtest=BacktestSettings(
            regime_weights={},
            initial_capital=capital,
            transaction_cost_bps=cost,
            risk_free_rate=rate,
        ),
    )


def market(
    rows: int = 5, *, dates: list[str] | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    index = pd.DatetimeIndex(dates) if dates else pd.bdate_range("2024-01-02", periods=rows)
    prices = pd.DataFrame(100.0, index=index, columns=["SPY", "GLD"])
    opens = prices.copy()
    sessions = pd.DataFrame(
        {
            "market_open": (index + pd.Timedelta(hours=9, minutes=30))
            .tz_localize("America/New_York")
            .tz_convert("UTC"),
            "market_close": (index + pd.Timedelta(hours=16))
            .tz_localize("America/New_York")
            .tz_convert("UTC"),
        },
        index=index,
    )
    return prices, opens, sessions
