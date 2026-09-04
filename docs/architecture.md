# RegimeLab — Complete Technical X-Ray

> Document generated for full visibility into the code surface, configuration, data structures, and test coverage.

---

## 1. Complete Configuration Schema

Configuration lives in `regimelab.settings` and uses `pydantic-settings`. Precedence order (highest to lowest) is:

1. Explicit keyword arguments to `Settings(...)`.
2. Environment variables with the `REGIMELAB_` prefix and `__` separator (e.g. `REGIMELAB_MODEL__N_STATES=3`).
3. `.env` file.
4. `regimelab.toml` in the working directory.
5. The documented defaults below.

All sections are immutable (`frozen=True`) and ignore unmapped foreign environment variables (`extra="ignore"`) to ensure seamless integration when imported inside host applications with their own environment configurations (e.g. LLM or database API keys).

### 1.1 `DataSettings`

| Field | Type | Default | Description / Validation |
|-------|------|---------|------------------------|
| `provider` | `Literal["parquet", "synthetic", "yfinance"]` | `"yfinance"` | Data provider. |
| `risk_assets` | `tuple[str, ...]` | `("SPY", "TLT", "GLD")` | Symbols eligible for allocation in the backtest. |
| `benchmark` | `str` | `"SPY"` | Buy-and-hold benchmark. |
| `vix_symbol` | `str` | `"^VIX"` | 30-day implied volatility index. |
| `vix3m_symbol` | `str` | `"^VIX3M"` | 93-day implied volatility index. |
| `cross_section` | `tuple[str, ...]` | 9 sector SPDRs (`XLB`…`XLY`) | Cross-section for absorption and breadth. |
| `start` | `dt.date` | `2005-01-01` | Start date for data ingestion. |
| `end` | `dt.date \| None` | `None` | End date (`None` means latest available). |
| `cache_dir` | `Path` | `platformdirs.user_cache_dir("regimelab")` | Local cache directory. |
| `cache_ttl_hours` | `float` | `12.0` | Cache TTL for open-ended network requests. |
| `calendar_anchor` | `str` | `"SPY"` | Symbol that defines the master calendar. |

**Public properties**

- `all_symbols: tuple[str, ...]` — ordered union of `risk_assets`, `cross_section`, `benchmark`, `vix_symbol`, `vix3m_symbol`, `calendar_anchor`.

**Validations**

- `len(cross_section) >= 3` (the PCA covariance matrix needs at least 3 assets).
- If `end` is not `None`, it must be later than `start`.

### 1.2 `FeatureSettings`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `absorption_window` | `int` | `500` | Covariance window for the absorption ratio. `>= 60`. |
| `absorption_eigen_fraction` | `float` | `0.2` | Fraction of leading eigenvectors. `(0, 1]`. |
| `absorption_delta_short` | `int` | `15` | Short averaging window for the AR delta. `>= 2`. |
| `absorption_delta_long` | `int` | `252` | Long averaging/std window for the AR delta. `>= 20`. |
| `correlation_window` | `int` | `63` | Rolling mean pairwise-correlation window. `>= 20`. |
| `breadth_sma` | `int` | `200` | SMA for the participation ratio. |
| `breadth_momentum_window` | `int` | `21` | Breadth momentum window. |
| `volatility_window` | `int` | `21` | Window for volatility estimators. `>= 5`. |
| `trailing_return_window` | `int` | `21` | Cumulative return window. `>= 5`. |
| `trading_days_per_year` | `int` | `252` | Trading days per year. `[200, 260]`. |

**Public property**

- `warmup: int` — maximum of `absorption_window + absorption_delta_long`, `breadth_sma + breadth_momentum_window`, `correlation_window`, `volatility_window`, `trailing_return_window`. This is the number of sessions required before every feature is defined.

### 1.3 `ModelSettings`

| Field | Type | Default | Description / Validation |
|-------|------|---------|--------------------------|
| `classifier` | `Literal["hmm", "gmm", "hamilton"]` | `"hmm"` | Regime classifier. |
| `n_states` | `int` | `4` | Number of latent states. `[2, 4]`. |
| `seed` | `int` | `20260831` | Master seed for reproducibility. |
| `n_init` | `int` | `10` | EM restarts. `>= 1`. |
| `max_iter` | `int` | `500` | Maximum iterations. `>= 10`. |
| `tolerance` | `float` | `1e-4` | Convergence tolerance. `> 0`. |
| `covariance_type` | `Literal["full", "diag", "tied", "spherical"]` | `"diag"` | Covariance parameterization. |
| `verbose` | `bool` | `False` | Estimator logging. |
| `min_train_observations` | `int` | `756` | Minimum training observations. `>= 252`. |
| `min_state_occupancy` | `float` | `0.03` | Minimum state occupancy. `[0, 0.5)`. |
| `features` | `tuple[str, ...]` | `("trailing_return", "realised_vol", "vix_term_spread", "absorption_delta", "mean_correlation")` | Observation vector; all entries must be fields of `FeatureSnapshot`. |

### 1.4 `BacktestSettings`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `signal_lag_days` | `int` | `1` | Days between observing a regime and trading it. `>= 1`. |
| `refit_frequency_days` | `int` | `21` | Walk-forward refit frequency. `>= 1`. |
| `transaction_cost_bps` | `float` | `5.0` | Transaction cost in bp per unit of turnover. `>= 0`. |
| `initial_capital` | `float` | `1_000_000.0` | Initial capital. `> 0`. |
| `risk_free_rate` | `float` | `0.02` | Annual risk-free rate. `>= 0`. |
| `regime_weights` | `dict[MarketRegime, dict[str, float]]` | Regime table (see defaults) | Target weights per regime. |
| `confidence_floor` | `float` | `0.0` | Minimum confidence to act. `[0, 1]`. |

**`regime_weights` defaults**

- `BULL_TREND`: `{"SPY": 0.90, "TLT": 0.10, "GLD": 0.00}`
- `NEUTRAL_TRANSITION`: `{"SPY": 0.50, "TLT": 0.25, "GLD": 0.10}`
- `HIGH_VOL_BEAR`: `{"SPY": 0.20, "TLT": 0.40, "GLD": 0.15}`
- `RISK_OFF`: `{"SPY": 0.00, "TLT": 0.30, "GLD": 0.20}`

**Validations**

- For each regime, the sum of weights may not exceed `1.0`.
- Negative weights are rejected (no shorting).

### 1.5 `Settings` (root)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `data` | `DataSettings` | `DataSettings()` | Data section. |
| `features` | `FeatureSettings` | `FeatureSettings()` | Feature section. |
| `model` | `ModelSettings` | `ModelSettings()` | Model section. |
| `backtest` | `BacktestSettings` | `BacktestSettings()` | Backtest section. |
| `artifacts_dir` | `Path` | `Path("artifacts")` | Artifact directory. |

**Cross-section validations**

- `model.features` must be a subset of the fields of `FeatureSnapshot`.
- The symbols in `backtest.regime_weights` must be in `data.risk_assets`.

---

## 2. Public Module and Function Inventory

### 2.1 `regimelab` (root)

Exports the stable API from `src/regimelab/__init__.py`:

- `MarketRegime`
- `RegimeProbabilities`
- `Settings`
- `TelemetryPayload`
- `__version__`

Submodules are reachable but considered reorganizable internals.

### 2.2 `regimelab.cli` — Typer CLI

- `app: typer.Typer` — root instance, `name="regimelab"`, `no_args_is_help=True`.
- `main() -> None` — console script entry point.
- `run(report: Path | None, tui: bool, config: Path | None) -> None` — full pipeline.
- `asof(date: str, config: Path | None, output: Path | None) -> None` — historical payload.
- `report(output: Path, config: Path | None) -> None` — direct HTML.
- `tui(config: Path | None) -> None` — Textual dashboard.

### 2.3 `regimelab.__main__`

- Imports `main` from `cli` to enable `python -m regimelab`.

### 2.4 `regimelab.data` — Ingestion and Alignment

#### 2.4.1 `regimelab.data.providers` — Providers

- `PriceProvider` (Protocol):
  - `name: str`
  - `is_point_in_time: bool`
  - `fetch(symbols, start, end) -> pd.DataFrame`
- `ProviderError` — operational exception.
- `OHLCV_FIELDS: tuple[str, ...] = ("open", "high", "low", "close", "volume")`
- `resolve_provider(settings: Settings) -> PriceProvider` — instantiates the configured provider.
- `validate_panel(panel: pd.DataFrame) -> None` — validates the panel contract.
- `empty_panel(symbols=()) -> pd.DataFrame` — valid empty panel.
- `normalise_index(frame) -> pd.DataFrame` — cleans the index (tz, duplicates, order).
- `panel_from_symbol_frames(frames: dict[str, pd.DataFrame]) -> pd.DataFrame` — assembles a panel.

**Provider classes**

| Class | `__init__` parameters | `is_point_in_time` | Notes |
|-------|------------------------|--------------------|-------|
| `YahooProvider` | `auto_adjust=True`, `timeout=30` | `False` | yfinance download, adjusted by default. |
| `SyntheticProvider` | `seed=20260831` | `True` | Deterministic synthetic generator with known latent states; `latent_states(start, end) -> pd.Series[int]`. |
| `ParquetProvider` | `root`, `strict=True` | `False` | Reads `root/<symbol>.parquet` (`^` normalized to `_`). |

- `write_snapshot(panel, root) -> tuple[Path, ...]` — persists a panel as a snapshot.
- `symbol_to_filename(symbol) -> str` — maps a ticker to a safe file name.

#### 2.4.2 `regimelab.data.alignment`

- `AlignmentError(ValueError)` — alignment failure.
- `DEFAULT_MAX_STALENESS: int = 5` — default staleness budget (not a `DataSettings` field; parameter of `align_panel`).
- `truncate_asof(frame, asof) -> pd.DataFrame | pd.Series` — PIT inclusive truncation.
- `trailing_window(frame, asof, length) -> pd.DataFrame | pd.Series` — last `length` rows up to `asof`.
- `align_panel(panel, *, anchor, max_staleness=5, start=None, end=None) -> AlignedPanel` — aligns a panel to a master calendar.

**`AlignedPanel`** (dataclass)

- `panel: pd.DataFrame`, `anchor: str`, `staleness: pd.DataFrame`, `max_staleness: int`, `dropped_symbols: tuple[str, ...]`
- Properties: `calendar`, `symbols`, `start`, `end`
- Methods: `field(field, symbols=None) -> pd.DataFrame`, `close(symbols=None) -> pd.DataFrame`, `log_returns(symbols=None) -> pd.DataFrame`, `asof(asof) -> AlignedPanel`, `fresh_symbols(asof=None, tolerance=0) -> tuple[str, ...]`, `digest() -> str`, `summary() -> dict`

#### 2.4.3 `regimelab.data.fetcher`

- `_CacheEntry` (internal dataclass)
- `PanelFetcher(provider, cache_dir, ttl_hours=12.0)`:
  - `fetch(symbols, start, end, *, force_refresh=False) -> pd.DataFrame`
  - Content-addressed caching (`sha256` of provider, symbols, range).
  - Closed ranges (`end` fixed) cache indefinitely; open ranges use `ttl_hours`.
- `load_aligned_panel(settings, *, end=None, force_refresh=False) -> AlignedPanel` — entry point combining provider, fetcher, and `align_panel`.

### 2.5 `regimelab.features` — Feature Engineering

Canonical feature-matrix columns (`FEATURE_COLUMNS` = fields of `FeatureSnapshot`):

`absorption_ratio`, `absorption_delta`, `vix_term_spread`, `realised_vol`, `breadth`, `breadth_momentum`, `mean_correlation`, `trailing_return`.

Also includes diagnostic columns: `parkinson_vol`, `defensive_correlation`.

**Public in `regimelab.features.__init__`**

- `build_feature_matrix(panel: AlignedPanel, settings: Settings) -> pd.DataFrame`
- `snapshot_at(matrix: pd.DataFrame, asof: dt.date | pd.Timestamp) -> FeatureSnapshot`

**Feature families**

`regimelab.features.absorption`

- `leading_eigenvector_count(n_assets, fraction=0.2) -> int`
- `absorption_ratio_series(returns, *, window=500, eigen_fraction=0.2, min_assets=3) -> pd.Series`
- `absorption_delta_series(absorption_ratio, *, short_window=15, long_window=252) -> pd.Series`
- `absorption_features(returns, *, window=500, eigen_fraction=0.2, short_window=15, long_window=252, symbols=None) -> pd.DataFrame`

`regimelab.features.breadth`

- `percent_above_sma(prices, *, window=200, min_assets=3) -> pd.Series`
- `breadth_momentum_series(breadth, *, window=21) -> pd.Series`
- `breadth_thrust(breadth, *, fast=10, slow=63) -> pd.Series`
- `breadth_features(prices, *, sma_window=200, momentum_window=21, min_assets=3) -> pd.DataFrame`

`regimelab.features.correlation`

- `mean_pairwise_correlation(returns, *, window=63, min_assets=3) -> pd.Series`
- `pairwise_correlation(left, right, *, window=63) -> pd.Series`
- `rolling_correlation_matrix(returns, asof, *, window=63) -> pd.DataFrame`
- `correlation_features(returns, *, window=63, risk_asset=None, defensive_assets=(), min_assets=3) -> pd.DataFrame`

`regimelab.features.volatility`

- `parkinson_volatility(high, low, *, window=21, trading_days=252) -> pd.Series`
- `garman_klass_volatility(open_, high, low, close, *, window=21, trading_days=252) -> pd.Series`
- `close_to_close_volatility(close, *, window=21, trading_days=252) -> pd.Series`
- `vix_term_spread(vix, vix3m) -> pd.Series`
- `trailing_log_return(close, *, window=21) -> pd.Series`
- `volatility_features(ohlc, vix=None, vix3m=None, *, window=21, return_window=21, trading_days=252) -> pd.DataFrame`

`regimelab.features._rolling` (internal but used by all families)

- `usable_columns(frame, *, min_coverage=0.95) -> list[str]`
- `rolling_matrix_reduce(frame, window, reducer, *, min_columns=3, name="value") -> pd.Series`
- `mean_offdiagonal(matrix) -> float`
- `standardised_shift(series, short_window, long_window, *, name) -> pd.Series`
- `annualisation_factor(trading_days) -> float`

### 2.6 `regimelab.models` — Regime Classifiers

#### 2.6.1 `regimelab.models.types`

- `MarketRegime(StrEnum)`:
  - `BULL_TREND`, `NEUTRAL_TRANSITION`, `HIGH_VOL_BEAR`, `RISK_OFF`
  - Properties: `risk_rank`, `is_risk_on`
- `REGIME_LADDER: tuple[MarketRegime, ...]` — canonical risk-on to risk-off order.
- `regime_ladder(n_states) -> tuple[MarketRegime, ...]` — returns the sub-ladder for 2, 3, or 4 states.
- `RegimeProbabilities`:
  - `probabilities: dict[MarketRegime, float]`
  - Properties: `argmax -> MarketRegime`, `confidence -> float`, `entropy -> float`, `normalised_entropy -> float`, `risk_off_mass -> float`
- `FeatureSnapshot`:
  - Fields: `absorption_ratio`, `absorption_delta`, `vix_term_spread`, `realised_vol`, `breadth`, `breadth_momentum`, `mean_correlation`, `trailing_return` (all `float | None`)
  - Methods: `is_complete() -> bool`, `missing() -> tuple[str, ...]`
- `AllocationWeights`:
  - `weights: dict[str, float]` `[0, 1]`
  - Properties: `invested -> float`, `cash -> float`
  - Method: `normalise(symbols=None) -> dict[str, float]`
- `ModelMetadata`:
  - Fields: `name`, `n_states`, `seed`, `train_start`, `train_end`, `n_train_observations`, `converged`, `log_likelihood`, `feature_names`
- `TelemetryPayload`:
  - Fields: `asof`, `regime`, `probabilities`, `features`, `model`, `target_weights`, `warnings`
  - Validators: `model.train_end <= asof` (last line of defence against look-ahead), regime consistency vs. probabilities.

#### 2.6.2 `regimelab.models.base`

- `InsufficientDataError(ValueError)`
- `CausalStandardiser` (dataclass):
  - `fit(frame) -> Self` — moments from the training window.
  - `transform(frame) -> np.ndarray` — applies z-score with frozen moments.
- `EmissionParameters` (dataclass): `means`, `covariances`; `log_density(matrix) -> np.ndarray`
- `StateEstimate` (dataclass): `posteriors`, `emission`, `transition`, `log_likelihood`, `converged`
- `FittedRegime` (dataclass):
  - `classifier`, `estimate`, `assignment`, `standardiser`, `metadata`
  - Methods: `probabilities(features, initial=None) -> pd.DataFrame`, `labels(features) -> pd.Series`, `probabilities_at(features, asof) -> RegimeProbabilities`, `transition_matrix() -> pd.DataFrame`, `expected_durations() -> pd.Series`
- `RegimeModel(ABC)`:
  - `name: ClassVar[str]`, `ranking_method: ClassVar[RankingMethod]`
  - `__init__(settings: ModelSettings)`
  - `fit(features, reference_returns, *, train_end=None, feature_names=None) -> FittedRegime`
  - Abstract method `_estimate(matrix, seed) -> StateEstimate`

#### 2.6.3 `regimelab.models.hmm`

- `HiddenMarkovRegimeModel(RegimeModel)`
  - `name = "hmm"`, `ranking_method = RankingMethod.SHARPE`
  - `_estimate` uses `hmmlearn.hmm.GaussianHMM` with `n_init` deterministic restarts.

#### 2.6.4 `regimelab.models.gmm`

- `GaussianMixtureRegimeModel(RegimeModel)`
  - `name = "gmm"`, `ranking_method = RankingMethod.VOLATILITY`
  - `_estimate` uses `sklearn.mixture.GaussianMixture`; `transition = None`.

#### 2.6.5 `regimelab.models.hamilton`

- `MarkovSwitchingRegimeModel(RegimeModel)`
  - `name = "hamilton"`, `ranking_method = RankingMethod.SHARPE`
  - `__init__(settings, *, dependent_index=0, switching_variance=True)`
  - `_estimate` uses `statsmodels.tsa.regime_switching.markov_regression.MarkovRegression`.

#### 2.6.6 `regimelab.models.registry`

- `CLASSIFIERS: dict[str, type[RegimeModel]]` — `{hmm, gmm, hamilton}`
- `build_classifier(settings, name=None) -> RegimeModel`

#### 2.6.7 `regimelab.models.labeling`

- `RankingMethod(StrEnum)`: `SHARPE`, `MEAN_RETURN`, `VOLATILITY`
- `OccupancyError(ValueError)`
- `StateStatistics` (dataclass): `state`, `occupancy`, `mean_return`, `volatility`, `n_observations`; property `sharpe`
- `LabelAssignment` (dataclass): `mapping`, `statistics`, `method`; property `inverse`; methods `regime_of(state)`, `describe() -> pd.DataFrame`
- `summarise_states(reference_returns, posteriors, *, trading_days=252) -> tuple[StateStatistics, ...]`
- `assign_labels(statistics, *, method=SHARPE, min_occupancy=0.0) -> LabelAssignment`
- `relabel_probabilities(posteriors, assignment) -> pd.DataFrame`

#### 2.6.8 `regimelab.models.filters`

- `FilterResult(NamedTuple)`: `filtered`, `predicted`, `log_likelihood`
- `gaussian_log_density(observations, means, covariances) -> np.ndarray`
- `stationary_distribution(transition) -> np.ndarray`
- `forward_filter(log_emissions, transition, initial=None) -> FilterResult` — Hamilton forward filter in log space.

### 2.7 `regimelab.backtest` — Backtesting and Metrics

#### 2.7.1 `regimelab.backtest.engine`

- `CASH_COLUMN = "CASH"`
- `BacktestResult` (dataclass):
  - Fields: `regimes`, `target_weights`, `held_weights`, `gross_returns`, `net_returns`, `traded_notional`, `costs`, `equity`, `benchmark_returns`, `benchmark_equity`, `metrics`, `benchmark_metrics`, `regime_metrics`
  - Properties: `drawdown`, `benchmark_drawdown`, `excess_return`
  - Methods: `comparison_table() -> pd.DataFrame`, `to_frame() -> pd.DataFrame`
- `target_weight_frame(regimes, settings, symbols) -> pd.DataFrame` — converts a regime path into target weights.
- `run_backtest(prices, regimes, settings, *, confidence=None, start=None) -> BacktestResult`

**`run_backtest` logic**

- Event-driven rebalancing (only when the regime changes).
- `signal_lag_days` delay between signal observation and position.
- `confidence_floor`: if modal confidence is below the floor, the previous allocation is retained.
- Costs = `transaction_cost_bps / 1e4 * traded_notional`.
- Unallocated cash earns the daily `risk_free_rate`.
- No leverage, no shorting.

#### 2.7.2 `regimelab.backtest.metrics`

- `equity_curve(returns, initial=1.0) -> pd.Series`
- `drawdown_series(equity) -> pd.Series`
- `PerformanceMetrics` (Pydantic):
  - Fields: `n_observations`, `start`, `end`, `total_return`, `cagr`, `annual_volatility`, `sharpe`, `sortino`, `max_drawdown`, `calmar`, `hit_rate`, `skewness`, `excess_kurtosis`, `var_95`, `cvar_95`, `average_exposure`, `annual_turnover`, `total_cost`
  - Method: `summary_rows() -> list[tuple[str, str]]`
- `compute_metrics(returns, *, trading_days=252, risk_free_rate=0.0, weights=None, turnover=None, costs=None) -> PerformanceMetrics`
- `regime_conditional_metrics(returns, regimes, *, trading_days=252, order=None) -> pd.DataFrame`

#### 2.7.3 `regimelab.backtest.walkforward`

- `WalkForwardFold` (dataclass): `fold_index`, `train_start`, `train_end`, `test_start`, `test_end`, `n_train`, `n_test`, `model: ModelMetadata`, `assignment: pd.DataFrame | None`
- `WalkForwardResult` (dataclass): `labels`, `probabilities`, `confidence`, `folds`, `features`, `backtest: BacktestResult | None`
  - Property `regimes` (alias of `labels`)
  - Method `regime_probabilities(asof) -> RegimeProbabilities`
- `run_walk_forward(panel, features, settings, *, run_backtest=True) -> WalkForwardResult`

**Walk-forward logic**

- Expanding window: each refit uses all history up to `train_end`.
- Refit dates determined by `min_train_observations` and `refit_frequency_days`.
- Between refits the model is frozen; the filter advances causally.
- The filtered posterior at the end of one segment is reused as the prior for the next to preserve Markov continuity.
- The last segment reaches the end of the panel.

### 2.8 `regimelab.pipeline` — Orchestration

- `PipelineResult` (dataclass): `settings`, `panel`, `features`, `walkforward`, `manifest`, `store`
  - Property `backtest -> BacktestResult | None`
- `latest_telemetry(result) -> TelemetryPayload`
- `run_pipeline(settings, *, command="run", run_backtest=True) -> PipelineResult`
- `run_single_asof(settings, asof, *, command="asof") -> TelemetryPayload`
- `RunCommand` (callable): wrapper for CLI/tests.

**Artifacts written by `run_pipeline` in the `RunStore`**

- `manifest.json`
- `features.parquet`
- `regime_probabilities.parquet`
- `regime_labels.parquet`
- `equity.parquet` (only if `run_backtest=True`)

### 2.9 `regimelab.report` — Reports

- `HTMLReport(result)`:
  - `render() -> str` — self-contained HTML with inline CSS and Plotly charts via CDN.
- `write_html_report(result, path) -> Path`

Report sections:

1. Performance summary (cards: strategy/benchmark CAGR, excess return, max drawdown, Sharpe, annual turnover).
2. Regime-conditional metrics (table per regime).
3. Latest feature snapshot (feature table).
4. Charts: equity, drawdown, regime probabilities (stacked area).

### 2.10 `regimelab.tui` — Textual Dashboard

- `RegimeLabApp = _LazyApp()` — lazy proxy that imports `textual` only when installed.
- `RegimeLabApp(result).run()` — launches the dashboard.
- Screens: backtest metrics, feature snapshot, regime probabilities, target weights.

### 2.11 `regimelab.store` — Provenance and Artifacts

- `GitState` (Pydantic): `commit`, `branch`, `dirty`
- `RunManifest` (Pydantic):
  - `run_id`, `created_at`, `command`, `asof`, `git`, `python_version`, `platform`, `packages`, `seed`, `settings`, `input_digest`
  - Properties: `is_reproducible`, `provenance_line()`
- `RunStore(root, manifest)`:
  - `directory`, `manifest`
  - `write_manifest() -> Path`
  - `write_frame(frame, name) -> Path`
  - `write_json(payload, name) -> Path`
  - `write_text(content, name) -> Path`
- `digest_frame(frame) -> str` — SHA-256 of the DataFrame/Series content.

---

## 3. Output Data Structures

### 3.1 `TelemetryPayload`

```python
TelemetryPayload(
    asof=dt.date,
    regime=MarketRegime,
    probabilities=RegimeProbabilities,
    features=FeatureSnapshot,
    model=ModelMetadata,
    target_weights=AllocationWeights | None,
    warnings=tuple[str, ...],
)
```

- `asof`: decision date.
- `regime`: modal regime.
- `probabilities`: full posterior distribution (dict `MarketRegime -> float`).
- `features`: causal systemic-risk snapshot.
- `model`: fit metadata (name, seed, training range, etc.).
- `target_weights`: current-regime target weights, or `None` if no backtest is configured.
- `warnings`: operational warnings.

**Validators**

- `model.train_end <= asof` (last line of defence against look-ahead).
- `regime` must appear in `probabilities`.

### 3.2 `PipelineResult`

```python
PipelineResult(
    settings=Settings,
    panel=AlignedPanel,
    features=pd.DataFrame,
    walkforward=WalkForwardResult,
    manifest=RunManifest,
    store=RunStore,
)
```

- `settings`: resolved configuration.
- `panel`: PIT-aligned panel.
- `features`: full causal feature matrix.
- `walkforward`: walk-forward loop and optional backtest result.
- `manifest`: run provenance.
- `store`: artifact directory accessor.
- Property `backtest`: `result.walkforward.backtest`.

### 3.3 `BacktestResult`

```python
BacktestResult(
    regimes=pd.Series,
    target_weights=pd.DataFrame,
    held_weights=pd.DataFrame,
    gross_returns=pd.Series,
    net_returns=pd.Series,
    traded_notional=pd.Series,
    costs=pd.Series,
    equity=pd.Series,
    benchmark_returns=pd.Series,
    benchmark_equity=pd.Series,
    metrics=PerformanceMetrics,
    benchmark_metrics=PerformanceMetrics,
    regime_metrics=pd.DataFrame,
)
```

- `target_weights`: theoretical per-session weights.
- `held_weights`: weights retained after lag and confidence floor.
- `gross_returns`: return before costs.
- `net_returns`: return after costs (used for metrics).
- `traded_notional`: turnover per session.
- `costs`: transaction cost per session.
- `metrics` vs `benchmark_metrics`: strategy and buy-and-hold metrics.
- `regime_metrics`: benchmark conditional statistics per regime.

### 3.4 `WalkForwardResult`

```python
WalkForwardResult(
    labels=pd.Series,          # modal regime per session
    probabilities=pd.DataFrame, # posterior per regime
    confidence=pd.Series,       # modal confidence
    folds=tuple[WalkForwardFold, ...],
    features=pd.DataFrame,
    backtest=BacktestResult | None,
)
```

### 3.5 `WalkForwardFold`

```python
WalkForwardFold(
    fold_index=int,
    train_start=dt.date,
    train_end=dt.date,
    test_start=dt.date,
    test_end=dt.date,
    n_train=int,
    n_test=int,
    model=ModelMetadata,
    assignment=pd.DataFrame | None,
)
```

### 3.6 `manifest.json` Schema

Example fields (Pydantic serialized to JSON):

```json
{
  "run_id": "a1b2c3d4e5f6",
  "created_at": "2026-09-01T18:30:00Z",
  "command": "run",
  "asof": null,
  "git": {
    "commit": "abc123...",
    "branch": "main",
    "dirty": false
  },
  "python_version": "3.12.4",
  "platform": "Windows-11...",
  "packages": {
    "regimelab": "0.1.4",
    "numpy": "2.5.2",
    "pandas": "3.0.5",
    ...
  },
  "seed": 20260831,
  "settings": { /* Data, Feature, Model, Backtest, artifacts_dir */ },
  "input_digest": "sha256..."
}
```

**Meaning**

- `run_id`: unique run identifier.
- `command`: CLI command that originated the run.
- `asof`: as-of date for the `asof` command; `null` for `run`.
- `git`: repository state.
- `packages`: versions of packages that affect numerical results.
- `settings`: resolved configuration; allows reproducing the run without the original environment.
- `input_digest`: hash of the input panel (`digest_frame(panel.close())`).

### 3.7 Parquet Files in `.artifacts/`

Artifact directory layout:

```text
.artifacts/
  <YYYYMMDD-HHMMSS>-<run_id>/
    manifest.json
    features.parquet
    regime_probabilities.parquet
    regime_labels.parquet
    equity.parquet  # only if there is a backtest
```

- `features.parquet`: `pd.DataFrame` with a `date` index and feature-matrix columns.
- `regime_probabilities.parquet`: posteriors per canonical regime; columns `BULL_TREND`, etc.
- `regime_labels.parquet`: modal label per session.
- `equity.parquet`: strategy equity curve.

In addition, the panel cache (`settings.data.cache_dir / panels / <sha16>.parquet`) persists downloaded panels to avoid repeated network calls.

---

## 4. Test List

Files: `tests/conftest.py`, `tests/test_backtest.py`, `tests/test_causality.py`.

### 4.1 Shared fixtures (`conftest.py`)

- `test_settings: Settings` — test configuration using the `synthetic` provider, dates 2015-2020, and shortened windows for speed.
- `pipeline_result: PipelineResult` — full pipeline generated once per session.

### 4.2 `tests/test_backtest.py`

1. **`test_held_weights_use_only_past_targets`**  
   Verifies that each held allocation at `t` matches a target weight observed on or before `t - signal_lag_days`.

2. **`test_transaction_costs_reduce_returns`**  
   `gross_returns >= net_returns` session by session.

3. **`test_weights_sum_to_one`**  
   Both `target_weights` and `held_weights` sum to 1.0 in every session.

4. **`test_confidence_floor_preserves_previous_allocation`**  
   When modal confidence is below `confidence_floor`, the held allocation is identical to the previous day's.

### 4.3 `tests/test_causality.py`

**`TestPointInTimeBoundary`**

5. **`test_truncate_asof_does_not_exceed_cursor`**  
   For every date in the calendar, `truncate_asof` returns an index whose maximum is exactly that date.

6. **`test_truncate_asof_property` (@given 20 random dates, Hypothesis)**  
   Property: the truncated frame contains no rows after the cursor.

**`TestFeaturesAndBacktest`**

7. **`test_feature_snapshot_exists_and_is_single_row`**  
   `snapshot_at` returns a valid `FeatureSnapshot` for a middle date.

8. **`test_backtest_held_weights_use_only_past_targets`**  
   Causal version of the backtest test: held weights come only from past targets.

**`TestWalkForward`**

9. **`test_folds_are_sequential`**  
   For each fold: `train_start <= train_end <= test_end` and `test_start <= test_end`.

10. **`test_test_windows_advance`**  
    Test windows cover the whole sample without missing overlaps between consecutive folds.

**`TestAsofPayload`**

Parametrized for three dates: `2020-03-16`, `2020-06-15`, `2020-12-31`. Counts as 3 test cases.

11-13. **`test_asof_payload_uses_only_history_before_cursor`**  
    `run_single_asof` returns a `TelemetryPayload` whose `asof` matches the requested date and whose `features` are available on that date. Ensures the as-of endpoint does not look past the cursor.

**Total: 13 test cases** (4 in `test_backtest` + 2 + 2 + 2 + 3 parametrized cases in `test_causality`).

---

## 5. End-to-End Data Flow

```text
[Provider] -> [PanelFetcher + cache] -> [align_panel]
    -> [build_feature_matrix]
    -> [run_walk_forward: fit / filter / backtest]
    -> [PipelineResult]
    -> [latest_telemetry | HTMLReport | RegimeLabApp | CLI output]
```

Every box in the flow produces an immutable artifact or Pydantic object, so the final decision point is always a `TelemetryPayload` validated against look-ahead.
