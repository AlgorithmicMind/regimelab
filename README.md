<div align="center">

# 🏛️ RegimeLab

**Institutional-grade market regime detection, systemic risk telemetry, and walk-forward asset allocation engine in Python.**

[![PyPI version](https://img.shields.io/pypi/v/regimelab.svg?color=blue)](https://pypi.org/project/regimelab/)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Type Checked with mypy](https://img.shields.io/badge/mypy-strict-blue)](https://mypy-lang.org/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

</div>

---

## ⚡ Why RegimeLab?

Most open-source regime detection scripts fit a Hidden Markov Model (HMM) on the entire in-sample dataset and claim predictive power. In production quantitative finance, this fails due to three fatal flaws:

1. **Look-Ahead Bias:** Training filters without strict point-in-time (`asof`) truncation leaks future distribution moments into past states.
2. **Label Switching:** HMM/GMM state indices are mathematically interchangeable between refits. Without canonical sorting, "State 0" randomly alternates between Bull and Bear across rolling windows.
3. **Calendar Desynchronization:** Multi-asset cross-sections suffer from holiday mismatches, halted assets, and survivorship bias.

**RegimeLab** solves these operational hurdles, providing a turnkey, causal quantitative engine for systematic asset allocation and macro risk monitoring.

---

### 📊 Feature Matrix: Raw Tooling vs. RegimeLab

| Challenge | Raw `hmmlearn` / `statsmodels` | RegimeLab Framework |
| :--- | :--- | :--- |
| **State Labeling** | Unordered integer states (permutes on refit) | **Deterministic Canonical Sorting** — stress score over emission means by default for HMM/Hamilton (configurable via `ranking_method` / `stress_signs`); Sharpe and volatility ordering still available |
| **Temporal Clock** | In-sample full-sample fitting (Look-ahead) | **Strict Point-in-Time (`asof`) cursor** & expanding-window walk-forward |
| **Systemic Risk** | None | **Kritzman Absorption Ratio (PCA)**, VIX Term Spread & Sector Breadth |
| **Execution Reality** | Pure theoretical classification | **Walk-Forward Backtester** with transaction costs (bps) and confidence floors |
| **Data Ingestion** | Expects clean 2D NumPy array | **Multi-Asset PIT Alignment**, staleness budgets, synthetic & Parquet providers |
| **Reporting** | Matplotlib static plot | **Interactive Plotly HTML reports** + Textual TUI Terminal Dashboard |

---

## 🚀 Quickstart

### 1. Installation

```bash
# Install the latest version available on PyPI
pip install regimelab

# Or install with interactive TUI support
pip install "regimelab[tui]"
```

Tagged wheels and source archives are also available on [GitHub Releases](https://github.com/AlgorithmicMind/regimelab/releases). GitHub and PyPI are published separately; if a version is not on PyPI yet, download its wheel and install it with `pip install path/to/wheel.whl`.

### 2. Python API Usage

#### Current Market Regime Detection (3 lines)

```python
from regimelab import Settings
from regimelab.pipeline import run_single_asof

# Run point-in-time regime inference for any historical or current date.
# refit_policy="walkforward" (the default) reproduces the exact posterior the
# walk-forward backtest produced for that date — the live signal is the
# track record. Pass refit_policy="daily" to refit through the as-of date.
settings = Settings(data={"provider": "yfinance"})
payload = run_single_asof(settings, asof="2024-12-31")

print(f"Detected Regime: {payload.regime.value}")
print(f"Confidence: {payload.probabilities.confidence:.2%}")
print(f"Target Allocation: {payload.target_weights.weights if payload.target_weights else {}}")

# Data quality is explicit: is_stale is True when the posterior was computed
# on an earlier session, and warnings explain why.
print(f"Computed on: {payload.observation_date} (last session {payload.session_date})")
print(f"Stale: {payload.is_stale}")
for warning in payload.warnings:
    print(f"  warning: {warning}")
```

`asof_not_in_panel` means absent panel coverage, not a confirmed market
closure. Check an exchange calendar before publication; `is_stale=False`
alone does not establish freshness.

#### Regime Statistics

`RegimeStatistics` entries are available through
`TelemetryPayload.state_statistics` and `WalkForwardFold.statistics`. Use
`payload.statistics_for(regime)` to look up one regime's statistics.

#### Extract Causal Systemic Risk Features

```python
from regimelab import Settings
from regimelab.data.fetcher import load_aligned_panel
from regimelab.features import build_feature_matrix

settings = Settings(data={"provider": "yfinance"})
panel = load_aligned_panel(settings)
features = build_feature_matrix(panel, settings)

# Inspect causal feature matrix
print(features[["absorption_ratio", "absorption_delta", "vix_term_spread", "breadth"]].tail())
```

#### Walk-Forward Backtesting Engine

```python
from regimelab import Settings
from regimelab.pipeline import run_pipeline

settings = Settings(
    data={"provider": "synthetic"}, # Fully offline, reproducible dataset
    model={"classifier": "hmm", "n_states": 4},
    backtest={"transaction_cost_bps": 5.0, "confidence_floor": 0.5},
)

result = run_pipeline(settings, command="backtest_run")
metrics = result.backtest.metrics

print(f"Strategy CAGR: {metrics.cagr:.2%}")
print(f"Sharpe Ratio:  {metrics.sharpe:.2f}")
print(f"Max Drawdown:  {metrics.max_drawdown:.2%}")
```

#### Backtest your own weights

Any target-weight schedule — not only the regime map — can be simulated under
the same execution assumptions (signal lag, event-driven rebalancing, costs):

```python
from regimelab.backtest import simulate_weights
from regimelab.data.fetcher import load_aligned_panel

prices = load_aligned_panel(settings).close()
# A decision on day t is held from t + signal_lag_days until the next one;
# CASH is the residual (here 0 — SPY/TLT are fully invested).
targets = pd.DataFrame({"SPY": 0.6, "TLT": 0.4}, index=prices.index[::21])
result = simulate_weights(prices, targets, settings)
print(f"60/40 Sharpe: {result.metrics.sharpe:.2f}")
```

This is the legacy close-only overlay: it uses integer-lag,
constant-weight and cash-inclusive-turnover conventions. It is not an
opening fill simulator. Missing, nonfinite or nonpositive prices needed
for an exposed return or the benchmark raise `ValueError`.

#### Next-open position ledger

`simulate_next_open` retains actual adjusted units between instructions, pays
costs on risky buys/sells (not cash transfers), and requires a caller-supplied
exchange schedule.

This fully offline example buys at 103 after the previous close of 100. It
earns only the subsequent move from 103 to 104, not the overnight gap:

```python
import pandas as pd
from regimelab import Settings
from regimelab.backtest import simulate_next_open

dates = pd.date_range("2024-01-02", periods=3)
closes = pd.DataFrame({"SPY": [100.0, 104.0, 104.0]}, index=dates)
opens = pd.DataFrame({"SPY": [100.0, 103.0, 104.0]}, index=dates)
sessions = pd.DataFrame({
    "market_open": (dates + pd.Timedelta(hours=9, minutes=30)).tz_localize("America/New_York"),
    "market_close": (dates + pd.Timedelta(hours=16)).tz_localize("America/New_York"),
}, index=dates)
settings = Settings(
    _env_file=None,
    data={"risk_assets": ("SPY",)},
    backtest={"regime_weights": {}, "initial_capital": 1000.0,
              "transaction_cost_bps": 0.0, "risk_free_rate": 0.0},
)
targets = pd.DataFrame({"SPY": [1.0]}, index=dates[:1])
result = simulate_next_open(
    closes, targets, settings, open_prices=opens, sessions=sessions
)
print(result.net_returns.iloc[0])  # 1 / 103, approximately 0.00970874
print(result.to_frame())          # units, cash, fees, timing and opening/closing weights
```

Without `published_at`, availability at the decision close is an explicit
research assumption. Supply actual timezone-aware publication timestamps for
an observed record. Instructions at/after an opening cannot receive an earlier
fill; an older delayed observation cannot override a newer one.
Use genuinely observed, consistently adjusted OHLC and the optional
`open_observed` / `close_observed` provenance masks. The library does not infer
freshness from a finite carried price or guarantee subscriber auction fills.
See [the execution specification](docs/next_open_execution.md) for cash accrual,
self-financing costs, benchmark treatment and compatibility boundaries.

### 3. CLI & Terminal Dashboard

RegimeLab ships with a powerful Typer CLI:

```bash
# 1. Run full walk-forward pipeline and generate interactive HTML report
regimelab run --report market_report.html

# 2. Inspect point-in-time telemetry for a specific date (JSON output)
regimelab asof 2023-10-15

# 3. Launch the full interactive Textual Terminal Dashboard
regimelab tui
```

Additional CLI commands include `regimelab report OUTPUT` for direct HTML generation and `regimelab asof YYYY-MM-DD --output telemetry.json` for persisted JSON payloads.

---

## 🧠 Core Methodology & Architecture

```text
                       ┌──────────────────────────────────────────────┐
                       │     Data Layer (PIT Alignment & Caching)     │
                       └──────────────────────┬───────────────────────┘
                                              │
                       ┌──────────────────────▼───────────────────────┐
                       │   Causal Features (PCA Absorption, Spread)   │
                       └──────────────────────┬───────────────────────┘
                                              │
                       ┌──────────────────────▼───────────────────────┐
                       │ Models: Hamilton / HMM / GMM + Anti-Switch   │
                       └──────────────────────┬───────────────────────┘
                                              │
               ┌──────────────────────────────┴──────────────────────────────┐
               ▼                                                             ▼
  ┌─────────────────────────┐                                   ┌─────────────────────────┐
  │   Walk-Forward Engine   │                                   │  Telemetry, HTML & TUI  │
  │  (Dynamic Allocations)  │                                   │ (Interactive Artifacts) │
  └─────────────────────────┘                                   └─────────────────────────┘
```

### 1. Canonical State Labeling (`regimelab.models.labeling`)

To eliminate label switching, RegimeLab fits the underlying statistical model (Gaussian HMM, Hamilton Markov Switching, or GMM) and evaluates the conditional distribution parameters of each state. HMM and Hamilton sort states by a **stress score** — the mean signed standardised emission mean over features with a known stress sign (`stress_signs` makes the sign table configurable) — while the GMM control orders by conditional volatility. Sharpe ($\frac{\mu}{\sigma}$) ordering remains available via `ranking_method`. States are then mapped deterministically to:

- **BULL_TREND** (High return, low volatility)
- **NEUTRAL_TRANSITION** (Moderate return, mean-reverting)
- **HIGH_VOL_BEAR** (Negative drift, elevated variance)
- **RISK_OFF** (Severe drawdown regime)

### 2. Kritzman Absorption Ratio (`regimelab.features.absorption`)

Quantifies market fragility via Principal Component Analysis (PCA) over rolling multi-asset return covariance matrices:

$$
\text{Absorption Ratio} = \frac{\sum_{i=1}^{k} \sigma^2_{PC_i}}{\sum_{j=1}^{N} \sigma^2_j}
$$

A rapid spike in the absorption ratio ($\Delta \text{AR} > 1.5$) indicates tightening cross-asset coupling, signaling systemic vulnerability prior to market crashes.

---

## ⚙️ Configuration (`regimelab.toml`)

Customize execution parameters via `regimelab.toml`, environment variables (`REGIMELAB_DATA__PROVIDER=yfinance`), or Python kwargs:

```toml
[data]
provider = "yfinance"          # "yfinance", "synthetic", or "parquet"
start = "2005-01-01"
benchmark = "SPY"
calendar_anchor = "SPY"

[model]
classifier = "hmm"             # "hmm", "gmm", or "hamilton"
n_states = 4
covariance_type = "diag"
min_train_observations = 756

[backtest]
transaction_cost_bps = 5.0
confidence_floor = 0.5
refit_frequency_days = 63
```

- **Signal and investable universes:** `data.correlation_assets` defines correlation inputs independently of investable `data.risk_assets`.
- **Volatility sources:** `data.volatility_source` configures CBOE/FRED sourcing for `^VIX` and `^VIX3M`, with provider fallback.
- **Panel caching:** `load_aligned_panel` uses `data.cache_dir` and `data.cache_ttl_hours`, anchored on `data.calendar_anchor`; `force_refresh=True` bypasses the cache.

---

## 🧪 Testing & Formal Verification

RegimeLab is built with property-based testing (hypothesis) to mathematically guarantee absence of look-ahead leakage:

```bash
# Run test suite with causality property tests
uv run pytest -q

# Run strict mypy type checking
uv run mypy src/regimelab

# Lint with ruff
uv run ruff check .
```

---

## Documentation

- [Architecture and methodology](docs/architecture.md)
- [Next-open execution contract](docs/next_open_execution.md)
- [Changelog and migration notes](CHANGELOG.md)
- [Release notes and downloads](https://github.com/AlgorithmicMind/regimelab/releases)

---

## 📄 License

MIT License. Developed for quantitative researchers, portfolio managers, and systematic trading engineers.
