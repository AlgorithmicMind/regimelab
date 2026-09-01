# regimelab

`regimelab` is a point-in-time, institutional market-regime detection engine. It
fetches or synthesises multi-asset panels, builds a causal feature matrix,
classifies regimes with HMM / GMM / Hamilton filters, and runs an event-driven
walk-forward backtest. The end products are a self-contained HTML report, a
`TelemetryPayload` boundary object, and a Textual dashboard.

The design objective is causality: no layer is allowed to read data from beyond
the current as-of date. This is enforced in `regimelab.data.alignment`, relied
upon by the feature builders, and verified by a dedicated property-test suite.

## Installation

Requires Python 3.12 or later.

```bash
uv sync
```

For the optional Textual dashboard:

```bash
uv pip install "regimelab[tui]"
```

## Quick start

Run the pipeline on synthetic data and write a report:

```bash
uv run python -m regimelab run --report report.html
```

The default provider is `yfinance`; set the provider explicitly in a
`regimelab.toml` file:

```toml
[data]
provider = "synthetic"

[model]
n_init = 3
max_iter = 100
covariance_type = "diag"

[backtest]
transaction_cost_bps = 5.0
confidence_floor = 0.5
```

Then run the full backtest with the report:

```bash
uv run python -m regimelab run --report .artifacts/report.html --config regimelab.toml
```

## CLI commands

- `regimelab run --report PATH` — full walk-forward pipeline and backtest.
- `regimelab asof YYYY-MM-DD` — telemetry payload for a historical date.
- `regimelab report OUTPUT` — run pipeline and write an HTML report.
- `regimelab tui` — run pipeline and open the Textual dashboard.

## Library usage

```python
from regimelab.pipeline import run_pipeline, latest_telemetry
from regimelab.report.html import write_html_report
from regimelab.settings import Settings

settings = Settings(
    data={"provider": "synthetic"},
    model={"n_init": 2, "covariance_type": "diag"},
    backtest={"transaction_cost_bps": 5.0},
)
result = run_pipeline(settings, command="demo")
write_html_report(result, "report.html")

payload = latest_telemetry(result)
print(payload.regime)
```

## Architecture

- `regimelab.data` — provider abstraction, point-in-time alignment, and the
  local panel cache.
- `regimelab.features` — absorption ratio, realised volatility, market breadth,
  and cross-asset correlation, all causal and point-in-time.
- `regimelab.models` — HMM, GMM, and Hamilton regime classifiers with canonical
  labelling to fix label switching.
- `regimelab.backtest` — event-driven, regime-conditional backtest with
  confidence floor and transaction-cost accounting.
- `regimelab.pipeline` — orchestration, manifest capture, and the
  `TelemetryPayload` boundary object.
- `regimelab.report` — self-contained Plotly HTML reports.
- `regimelab.tui` — optional Textual dashboard.
- `regimelab.cli` — Typer entry point.

## Causality and point-in-time discipline

The three rules that protect every run from look-ahead are:

1. **One master calendar** (`align_panel`) so a missing session in one symbol
   cannot be forward-filled into an invented bar.
2. **Forward-fill with a staleness budget** so a stale value degrades to `NaN`
   instead of silently leaking.
3. **As-of truncation** (`truncate_asof`) is the single primitive on which every
   PIT cut is made; there is no other way to slice time.

These are exercised by the `tests/test_causality.py` suite, which includes
Hypothesis property checks and walk-forward boundary tests.

## Running the test suite

```bash
uv run pytest
```

The suite includes causality property tests, backtest invariant tests, and an
end-to-end walk-forward smoke test on synthetic data.

## License

MIT.
