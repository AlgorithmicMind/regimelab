# Changelog

## 0.2.0

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
