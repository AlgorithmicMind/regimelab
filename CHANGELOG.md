# Changelog

## 0.4.0 — 2026-10-20

- Fix: HMM convergence metadata describes the selected highest-likelihood restart, not whether any restart converged. Model selection and fitted numerical parameters are unchanged.
- Added: `simulate_next_open` and `NextOpenResult`, an explicit next-open position ledger with publication ordering, observation masks, actual unit drift, self-financing per-risky-leg costs and an identically accounted benchmark. The legacy overlay and pipeline defaults remain unchanged.
- Added: optional initial-equity baselines for drawdown/Calmar; their default `None` preserves legacy metrics. The new ledger includes inception losses relative to capital.
- The API/accounting contract and assumptions are documented in `docs/next_open_execution.md`; next-open execution remains opt-in and the default pipeline, model settings and historical start date remain unchanged.
- Fix: the legacy overlay rejects missing, nonfinite or nonpositive closes needed for an exposed return or its benchmark instead of silently reporting zero/skipping dates. Valid-input golden outputs are preserved; only the first benchmark interval may lack a preceding input observation.
- Breaking warning-code change: `non_session_asof` is replaced by `asof_not_in_panel`. Missing panel coverage is not inferred to be a market holiday; `is_stale` still compares the posterior date with the last supplied session, not the expected exchange calendar.

## 0.3.0 — 2026-10-20

- `RegimeStatistics`: `LabelAssignment.to_statistics()` exposes per-regime training-window statistics — one entry per regime, so states sharing a regime (the positive-drift guardrail under VOLATILITY/SHARPE) are combined with occupancy-weighted moments, mirroring how `relabel_probabilities` sums their posteriors; `raw_states: tuple[int, ...]` lists every member state, and `ann_return`/`ann_volatility`/`sharpe` are `None` for a regime without weight. `TelemetryPayload.state_statistics` carries them and `statistics_for(regime)` looks them up by regime
- `LabelAssignment.describe()` keeps one row per raw state (as in 0.2.0), sorted by ladder position
- `WalkForwardFold.statistics` holds the fold model's per-regime statistics
- `latest_telemetry` signs the payload with the model and statistics of the fold whose `[test_start, test_end]` contains `observation_date`, instead of always the last fold
- `simulate_weights` simulates any hand-supplied target-weight schedule under the same execution assumptions; `run_backtest` is reimplemented on it with numerically identical output (golden regression test)
- Breaking: `BacktestResult.regimes` and `regime_metrics` are optional (`None` when a weight path rather than a regime signal was simulated) — a breaking change for type consumers
- Fix: leading sessions whose confidence is below the floor now hold 100% CASH and earn the risk-free rate, instead of holding zero weights and earning nothing
- Breaking: the pipeline data path goes through `load_aligned_panel` — the panel cache moves from `artifacts_dir/cache` to `data.cache_dir`, panels are reused within `data.cache_ttl_hours` (the TTL is no longer forced to 0) and alignment anchors on `data.calendar_anchor`; `force_refresh=True` restores always-refetch
- `RankingMethod.SHARPE` docstring now states it is no longer any classifier's default (`hmm`/`hamilton` → `STRESS`, `gmm` → `VOLATILITY`; see `DEFAULT_RANKING_METHODS`)

## 0.2.0 — 2026-10-07

- Official CBOE/FRED sourcing for ^VIX and ^VIX3M with provider fallback
- Breaking: HMM and Hamilton models rank states by stress features; positive-drift guardrail no longer applies under STRESS
- Fix: walk-forward look-ahead — each fold now labels only the interval strictly
  after its own training window, filtering the full history from the stationary
  distribution with frozen parameters instead of carrying the previous model's
  posterior forward.
- TelemetryPayload.observation_date, session_date, is_stale and populated warnings (non-session asof reported separately from stale data)
- DataSettings.correlation_assets decouples signal universe from investable universe
- Breaking: run_single_asof defaults to refit_policy="walkforward", reproducing the walk-forward signal exactly (train_end = last refit strictly before the observation date); refit_policy="daily" keeps the previous behaviour
- ModelSettings.stress_signs makes STRESS ranking signs configurable (replaces STRESS_SIGNS entirely; validated against model.features when STRESS is in effect)
