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


def main() -> None:
    """Verify installed identity, both execution APIs and core CLI entry points."""
    prefix = Path(sys.prefix).resolve()
    site = Path(sysconfig.get_path("purelib")).resolve()
    module_path = Path(regimelab.__file__).resolve()
    assert prefix != Path(sys.base_prefix).resolve(), "smoke requires an isolated venv"
    configuration = (prefix / "pyvenv.cfg").read_text(encoding="utf-8").lower()
    assert "include-system-site-packages = false" in configuration
    assert module_path.is_relative_to(site), (module_path, site)
    assert regimelab.__version__ == "0.4.0"
    assert importlib.metadata.version("regimelab") == "0.4.0"
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
                "legacy_gap_error": legacy_error,
                "module_cli_help": "passed",
                "console_cli_help": "passed",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
