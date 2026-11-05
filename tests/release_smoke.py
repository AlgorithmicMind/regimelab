"""Run with an installed wheel, outside the checkout, using python -I."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import subprocess
import sys
import sysconfig
from pathlib import Path

import pandas as pd

import regimelab
from regimelab.backtest import NextOpenResult, simulate_next_open, simulate_weights
from regimelab.settings import BacktestSettings, DataSettings, Settings


def main() -> None:  # noqa: PLR0915
    """Verify installed identity, both execution APIs and core CLI entry points."""
    prefix = Path(sys.prefix).resolve()
    site = Path(sysconfig.get_path("purelib")).resolve()
    module_path = Path(regimelab.__file__).resolve()
    assert prefix != Path(sys.base_prefix).resolve(), "smoke requires an isolated venv"
    configuration = (prefix / "pyvenv.cfg").read_text(encoding="utf-8").lower()
    assert "include-system-site-packages = false" in configuration
    assert module_path.is_relative_to(site), (module_path, site)
    assert regimelab.__version__ == "0.5.0"
    assert importlib.metadata.version("regimelab") == "0.5.0"
    distribution = importlib.metadata.distribution("regimelab")
    direct_url = distribution.read_text("direct_url.json")
    if direct_url:
        assert not json.loads(direct_url).get("dir_info", {}).get("editable", False)
    assert importlib.util.find_spec("textual") is None, "verify the core install without TUI extras"

    dates = pd.date_range("2024-01-02", periods=3)
    prices = pd.DataFrame({"SPY": [100.0, 104.0, 104.0], "GLD": 100.0}, index=dates)
    opens = pd.DataFrame({"SPY": [100.0, 103.0, 104.0], "GLD": 100.0}, index=dates)
    sessions = pd.DataFrame(
        {
            "market_open": (dates + pd.Timedelta(hours=9, minutes=30)).tz_localize(
                "America/New_York"
            ),
            "market_close": (dates + pd.Timedelta(hours=16)).tz_localize("America/New_York"),
        },
        index=dates,
    )
    settings = Settings(
        _env_file=None,
        data=DataSettings(provider="synthetic", risk_assets=("SPY", "GLD")),
        backtest=BacktestSettings(
            regime_weights={},
            initial_capital=1000.0,
            transaction_cost_bps=0.0,
            risk_free_rate=0.0,
        ),
    )
    targets = pd.DataFrame({"SPY": [1.0]}, index=dates[:1])
    result = simulate_next_open(prices, targets, settings, open_prices=opens, sessions=sessions)
    assert isinstance(result, NextOpenResult)
    gap_return = float(result.net_returns.iloc[0])
    assert abs(gap_return - 1 / 103) <= 1e-12
    assert result.to_frame()["has_fills"].tolist() == [True, False]
    assert result.rebalance_policy == "target_change"

    mixed_closes = pd.DataFrame(
        {"SPY": [100.0, 120.0, 132.0], "GLD": [100.0, 100.0, 90.0]}, index=dates
    )
    mixed_opens = pd.DataFrame(
        {"SPY": [100.0, 100.0, 120.0], "GLD": [100.0, 100.0, 100.0]}, index=dates
    )
    mixed_targets = pd.DataFrame({"SPY": 0.5, "GLD": 0.5}, index=dates[:2])
    mixed_settings = settings.model_copy(
        update={"backtest": settings.backtest.model_copy(update={"initial_capital": 200.0})}
    )
    default = simulate_next_open(
        mixed_closes, mixed_targets, mixed_settings, open_prices=mixed_opens, sessions=sessions
    )
    explicit = simulate_next_open(
        mixed_closes,
        mixed_targets,
        mixed_settings,
        open_prices=mixed_opens,
        sessions=sessions,
        rebalance_policy="each_instruction",
    )
    carried = simulate_next_open(
        mixed_closes,
        mixed_targets.iloc[:1],
        mixed_settings,
        open_prices=mixed_opens,
        sessions=sessions,
        rebalance_policy="each_instruction",
    )
    assert explicit.rebalance_policy == "each_instruction"
    assert abs(float(default.nav.iloc[-1]) - 222.0) <= 1e-10
    assert abs(float(explicit.nav.iloc[-1]) - 220.0) <= 1e-10
    assert abs(float(carried.nav.iloc[-1]) - 222.0) <= 1e-10
    assert abs(float(explicit.traded_values["SPY"].iloc[-1]) + 10.0) <= 1e-10
    assert abs(float(explicit.traded_values["GLD"].iloc[-1]) - 10.0) <= 1e-10
    pd.testing.assert_series_equal(default.benchmark_returns, explicit.benchmark_returns)

    legacy_prices = prices.copy()
    legacy_prices.loc[dates[-1], "GLD"] = float("nan")
    legacy_targets = pd.DataFrame({"GLD": [1.0]}, index=dates[:1])
    try:
        simulate_weights(legacy_prices, legacy_targets, settings)
    except ValueError as exc:
        legacy_error = str(exc)
        assert "GLD" in legacy_error
        assert "2024-01-04" in legacy_error
    else:
        raise AssertionError("installed legacy API silently accepted an exposed price gap")

    module_help = subprocess.run(
        [sys.executable, "-I", "-m", "regimelab", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "regimelab" in module_help.stdout.lower()
    script_name = "regimelab.exe" if sys.platform == "win32" else "regimelab"
    console_script = Path(sysconfig.get_path("scripts")) / script_name
    assert console_script.is_file()
    console_help = subprocess.run(
        [str(console_script), "--help"], check=True, capture_output=True, text=True
    )
    assert "regimelab" in console_help.stdout.lower()
    print(
        json.dumps(
            {
                "version": regimelab.__version__,
                "distribution_version": importlib.metadata.version("regimelab"),
                "module_path": str(module_path),
                "site_packages": str(site),
                "python": sys.executable,
                "system_site_packages": False,
                "textual_installed": False,
                "next_open_gap_return": gap_return,
                "default_policy_final_nav": float(default.nav.iloc[-1]),
                "explicit_policy_final_nav": float(explicit.nav.iloc[-1]),
                "carried_policy_final_nav": float(carried.nav.iloc[-1]),
                "legacy_gap_error": legacy_error,
                "module_cli_help": "passed",
                "console_cli_help": "passed",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
