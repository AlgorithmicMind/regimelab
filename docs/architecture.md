# RegimeLab — Radiografía Técnica Completa

> Documento generado para visibilidad total de la superficie de código, la configuración, las estructuras de datos y la cobertura de tests.

---

## 1. Esquema Completo de Configuración

La configuración vive en `regimelab.settings` y utiliza `pydantic-settings`. El orden de precedencia (mayor a menor) es:

1. Kwargs explícitos a `Settings(...)`.
2. Variables de entorno con prefijo `REGIMELAB_` y separador `__` (ej. `REGIMELAB_MODEL__N_STATES=3`).
3. Archivo `.env`.
4. Archivo `regimelab.toml` en el directorio de trabajo.
5. Valores por defecto documentados a continuación.

Todas las secciones son inmutables (`frozen=True`) y rechazan campos extra (`extra="forbid"`).

### 1.1 `DataSettings`

| Campo | Tipo | Valor por defecto | Descripción / Validación |
|-------|------|-------------------|--------------------------|
| `provider` | `Literal["parquet", "synthetic", "yfinance"]` | `"yfinance"` | Proveedor de datos. |
| `risk_assets` | `tuple[str, ...]` | `("SPY", "TLT", "GLD")` | Símbolos asignables en el backtest. |
| `benchmark` | `str` | `"SPY"` | Benchmark buy-and-hold. |
| `vix_symbol` | `str` | `"^VIX"` | Índice de volatilidad 30 días. |
| `vix3m_symbol` | `str` | `"^VIX3M"` | Índice de volatilidad 93 días. |
| `cross_section` | `tuple[str, ...]` | 9 sector SPDRs (`XLB`…`XLY`) | Cross-section para absorpción y breadth. |
| `start` | `dt.date` | `2005-01-01` | Fecha de inicio de ingestión. |
| `end` | `dt.date \| None` | `None` | Fecha final (`None` = última disponible). |
| `cache_dir` | `Path` | `platformdirs.user_cache_dir("regimelab")` | Cache local. |
| `cache_ttl_hours` | `float` | `12.0` | TTL de cache para peticiones abiertas. |
| `calendar_anchor` | `str` | `"SPY"` | Símbolo que define el calendario maestro. |

**Propiedades públicas**

- `all_symbols: tuple[str, ...]` — unión ordenada de `risk_assets`, `cross_section`, `benchmark`, `vix_symbol`, `vix3m_symbol`, `calendar_anchor`.

**Validaciones**

- `len(cross_section) >= 3` (la matriz de covarianza para PCA necesita al menos 3 activos).
- Si `end` no es `None`, debe ser posterior a `start`.

### 1.2 `FeatureSettings`

| Campo | Tipo | Valor por defecto | Descripción |
|-------|------|-------------------|-------------|
| `absorption_window` | `int` | `500` | Ventana de covarianza para el absorption ratio. `>= 60`. |
| `absorption_eigen_fraction` | `float` | `0.2` | Fracción de eigenvectores líderes. `(0, 1]`. |
| `absorption_delta_short` | `int` | `15` | Media corta del delta AR. `>= 2`. |
| `absorption_delta_long` | `int` | `252` | Media/desv. larga del delta AR. `>= 20`. |
| `correlation_window` | `int` | `63` | Ventana correlación media por pares. `>= 20`. |
| `breadth_sma` | `int` | `200` | SMA para el ratio de participación. |
| `breadth_momentum_window` | `int` | `21` | Ventana del momentum de breadth. |
| `volatility_window` | `int` | `21` | Ventana para estimadores de volatilidad. `>= 5`. |
| `trailing_return_window` | `int` | `21` | Ventana del retorno acumulado. `>= 5`. |
| `trading_days_per_year` | `int` | `252` | Días de negociación anual. `[200, 260]`. |

**Propiedad pública**

- `warmup: int` — máximo de `absorption_window + absorption_delta_long`, `breadth_sma + breadth_momentum_window`, `correlation_window`, `volatility_window`, `trailing_return_window`. Es la cantidad de sesiones necesarias antes de que cualquier feature esté definida.

### 1.3 `ModelSettings`

| Campo | Tipo | Valor por defecto | Descripción / Validación |
|-------|------|-------------------|--------------------------|
| `classifier` | `Literal["hmm", "gmm", "hamilton"]` | `"hmm"` | Clasificador de regímenes. |
| `n_states` | `int` | `4` | Número de estados latentes. `[2, 4]`. |
| `seed` | `int` | `20260831` | Semilla maestra para reproducibilidad. |
| `n_init` | `int` | `10` | Restarts del EM. `>= 1`. |
| `max_iter` | `int` | `500` | Iteraciones máximas. `>= 10`. |
| `tolerance` | `float` | `1e-4` | Tolerancia de convergencia. `> 0`. |
| `covariance_type` | `Literal["full", "diag", "tied", "spherical"]` | `"diag"` | Parametrización de covarianza. |
| `verbose` | `bool` | `False` | Logs del estimador. |
| `min_train_observations` | `int` | `756` | Mínimo de observaciones de entrenamiento. `>= 252`. |
| `min_state_occupancy` | `float` | `0.03` | Mínima ocupación de un estado. `[0, 0.5)`. |
| `features` | `tuple[str, ...]` | `("trailing_return", "realised_vol", "vix_term_spread", "absorption_delta", "mean_correlation")` | Vector de observación; todos deben ser campos de `FeatureSnapshot`. |

### 1.4 `BacktestSettings`

| Campo | Tipo | Valor por defecto | Descripción |
|-------|------|-------------------|-------------|
| `signal_lag_days` | `int` | `1` | Días entre observar un régimen y operar. `>= 1`. |
| `refit_frequency_days` | `int` | `21` | Frecuencia de re-entrenamiento walk-forward. `>= 1`. |
| `transaction_cost_bps` | `float` | `5.0` | Coste de transacción en bp por unidad de turnover. `>= 0`. |
| `initial_capital` | `float` | `1_000_000.0` | Capital inicial. `> 0`. |
| `risk_free_rate` | `float` | `0.02` | Tipo libre de riesgo anual. `>= 0`. |
| `regime_weights` | `dict[MarketRegime, dict[str, float]]` | Tabla por régimen (ver defaults) | Pesos objetivo por régimen. |
| `confidence_floor` | `float` | `0.0` | Confianza mínima para actuar. `[0, 1]`. |

**Defaults de `regime_weights`**

- `BULL_TREND`: `{"SPY": 0.90, "TLT": 0.10, "GLD": 0.00}`
- `NEUTRAL_TRANSITION`: `{"SPY": 0.50, "TLT": 0.25, "GLD": 0.10}`
- `HIGH_VOL_BEAR`: `{"SPY": 0.20, "TLT": 0.40, "GLD": 0.15}`
- `RISK_OFF`: `{"SPY": 0.00, "TLT": 0.30, "GLD": 0.20}`

**Validaciones**

- Para cada régimen, la suma de pesos no puede superar `1.0`.
- Pesos negativos rechazados (no hay short).

### 1.5 `Settings` (raíz)

| Campo | Tipo | Valor por defecto | Descripción |
|-------|------|-------------------|-------------|
| `data` | `DataSettings` | `DataSettings()` | Sección de datos. |
| `features` | `FeatureSettings` | `FeatureSettings()` | Sección de features. |
| `model` | `ModelSettings` | `ModelSettings()` | Sección de modelo. |
| `backtest` | `BacktestSettings` | `BacktestSettings()` | Sección de backtest. |
| `artifacts_dir` | `Path` | `Path("artifacts")` | Directorio de artefactos. |

**Validaciones cruzadas**

- `model.features` debe ser subconjunto de los campos de `FeatureSnapshot`.
- Los símbolos en `backtest.regime_weights` deben estar en `data.risk_assets`.

---

## 2. Inventario de Módulos y Funciones Públicas

### 2.1 `regimelab` (raíz)

Exporta la API estable desde `src/regimelab/__init__.py`:

- `MarketRegime`
- `RegimeProbabilities`
- `Settings`
- `TelemetryPayload`
- `__version__`

Submódulos son accesibles pero se consideran internos reorganizables.

### 2.2 `regimelab.cli` — CLI Typer

- `app: typer.Typer` — instancia raíz, `name="regimelab"`, `no_args_is_help=True`.
- `main() -> None` — entry point del console script.
- `run(report: Path | None, tui: bool, config: Path | None) -> None` — pipeline completo.
- `asof(date: str, config: Path | None, output: Path | None) -> None` — payload histórico.
- `report(output: Path, config: Path | None) -> None` — HTML directo.
- `tui(config: Path | None) -> None` — dashboard Textual.

### 2.3 `regimelab.__main__`

- Importa `main` de `cli` para permitir `python -m regimelab`.

### 2.4 `regimelab.data` — Ingestión y Alineación

#### 2.4.1 `regimelab.data.providers` — Providers

- `PriceProvider` (Protocol):
  - `name: str`
  - `is_point_in_time: bool`
  - `fetch(symbols, start, end) -> pd.DataFrame`
- `ProviderError` — excepción operacional.
- `OHLCV_FIELDS: tuple[str, ...] = ("open", "high", "low", "close", "volume")`
- `resolve_provider(settings: Settings) -> PriceProvider` — instancia el provider configurado.
- `validate_panel(panel: pd.DataFrame) -> None` — valida el contrato de panel.
- `empty_panel(symbols=()) -> pd.DataFrame` — panel vacío válido.
- `normalise_index(frame) -> pd.DataFrame` — limpia índice (tz, duplicados, orden).
- `panel_from_symbol_frames(frames: dict[str, pd.DataFrame]) -> pd.DataFrame` — ensambla panel.

**Clases provider**

| Clase | Parámetros `__init__` | `is_point_in_time` | Notas |
|-------|------------------------|--------------------|-------|
| `YahooProvider` | `auto_adjust=True`, `timeout=30` | `False` | Descarga yfinance, ajustada por defecto. |
| `SyntheticProvider` | `seed=20260831` | `True` | Generador sintético con estados latentes conocidos; `latent_states(start, end) -> pd.Series[int]`. |
| `ParquetProvider` | `root`, `strict=True` | `False` | Lee `root/<symbol>.parquet` (el `^` se normaliza a `_`). |

- `write_snapshot(panel, root) -> tuple[Path, ...]` — persiste panel como snapshot.
- `symbol_to_filename(symbol) -> str` — mapea ticker a nombre de fichero seguro.

#### 2.4.2 `regimelab.data.alignment`

- `AlignmentError(ValueError)` — fallo de alineación.
- `DEFAULT_MAX_STALENESS: int = 5` — presupuesto de staleness por defecto (no es configurable en `DataSettings`, es parámetro de `align_panel`).
- `truncate_asof(frame, asof) -> pd.DataFrame | pd.Series` — recorte PIT inclusivo.
- `trailing_window(frame, asof, length) -> pd.DataFrame | pd.Series` — últimas `length` filas hasta `asof`.
- `align_panel(panel, *, anchor, max_staleness=5, start=None, end=None) -> AlignedPanel` — alinea panel a calendario maestro.

**`AlignedPanel`** (dataclass)

- `panel: pd.DataFrame`, `anchor: str`, `staleness: pd.DataFrame`, `max_staleness: int`, `dropped_symbols: tuple[str, ...]`
- Propiedades: `calendar`, `symbols`, `start`, `end`
- Métodos: `field(field, symbols=None) -> pd.DataFrame`, `close(symbols=None) -> pd.DataFrame`, `log_returns(symbols=None) -> pd.DataFrame`, `asof(asof) -> AlignedPanel`, `fresh_symbols(asof=None, tolerance=0) -> tuple[str, ...]`, `digest() -> str`, `summary() -> dict`

#### 2.4.3 `regimelab.data.fetcher`

- `_CacheEntry` (dataclass interno)
- `PanelFetcher(provider, cache_dir, ttl_hours=12.0)`:
  - `fetch(symbols, start, end, *, force_refresh=False) -> pd.DataFrame`
  - Cacheo por contenido (`sha256` de provider, símbolos, rango).
  - Rangos cerrados (`end` fijo) cachean indefinidamente; rangos abiertos usan `ttl_hours`.
- `load_aligned_panel(settings, *, end=None, force_refresh=False) -> AlignedPanel` — entry point que combina provider, fetcher y `align_panel`.

### 2.5 `regimelab.features` — Feature Engineering

Columnas canónicas del feature matrix (`FEATURE_COLUMNS` = campos de `FeatureSnapshot`):

`absorption_ratio`, `absorption_delta`, `vix_term_spread`, `realised_vol`, `breadth`, `breadth_momentum`, `mean_correlation`, `trailing_return`.

Además incluye columnas diagnóstico: `parkinson_vol`, `defensive_correlation`.

**Públicas en `regimelab.features.__init__`**

- `build_feature_matrix(panel: AlignedPanel, settings: Settings) -> pd.DataFrame`
- `snapshot_at(matrix: pd.DataFrame, asof: dt.date | pd.Timestamp) -> FeatureSnapshot`

**Familias de features**

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

`regimelab.features._rolling` (interno pero usado por todas las familias)

- `usable_columns(frame, *, min_coverage=0.95) -> list[str]`
- `rolling_matrix_reduce(frame, window, reducer, *, min_columns=3, name="value") -> pd.Series`
- `mean_offdiagonal(matrix) -> float`
- `standardised_shift(series, short_window, long_window, *, name) -> pd.Series`
- `annualisation_factor(trading_days) -> float`

### 2.6 `regimelab.models` — Clasificadores de Régimen

#### 2.6.1 `regimelab.models.types`

- `MarketRegime(StrEnum)`:
  - `BULL_TREND`, `NEUTRAL_TRANSITION`, `HIGH_VOL_BEAR`, `RISK_OFF`
  - Propiedades: `risk_rank`, `is_risk_on`
- `REGIME_LADDER: tuple[MarketRegime, ...]` — orden canónico de risk-on a risk-off.
- `regime_ladder(n_states) -> tuple[MarketRegime, ...]` — devuelve sub-escalera para 2, 3 o 4 estados.
- `RegimeProbabilities`:
  - `probabilities: dict[MarketRegime, float]`
  - Propiedades: `argmax -> MarketRegime`, `confidence -> float`, `entropy -> float`, `normalised_entropy -> float`, `risk_off_mass -> float`
- `FeatureSnapshot`:
  - Campos: `absorption_ratio`, `absorption_delta`, `vix_term_spread`, `realised_vol`, `breadth`, `breadth_momentum`, `mean_correlation`, `trailing_return` (todos `float | None`)
  - Métodos: `is_complete() -> bool`, `missing() -> tuple[str, ...]`
- `AllocationWeights`:
  - `weights: dict[str, float]` `[0, 1]`
  - Propiedades: `invested -> float`, `cash -> float`
  - Método: `normalise(symbols=None) -> dict[str, float]`
- `ModelMetadata`:
  - Campos: `name`, `n_states`, `seed`, `train_start`, `train_end`, `n_train_observations`, `converged`, `log_likelihood`, `feature_names`
- `TelemetryPayload`:
  - Campos: `asof`, `regime`, `probabilities`, `features`, `model`, `target_weights`, `warnings`
  - Validadores: `train_end <= asof` (look-ahead), consistencia régimen vs. probabilidades.

#### 2.6.2 `regimelab.models.base`

- `InsufficientDataError(ValueError)`
- `CausalStandardiser` (dataclass):
  - `fit(frame) -> Self` — momentos de la ventana de entrenamiento.
  - `transform(frame) -> np.ndarray` — aplica z-score con momentos congelados.
- `EmissionParameters` (dataclass): `means`, `covariances`; `log_density(matrix) -> np.ndarray`
- `StateEstimate` (dataclass): `posteriors`, `emission`, `transition`, `log_likelihood`, `converged`
- `FittedRegime` (dataclass):
  - `classifier`, `estimate`, `assignment`, `standardiser`, `metadata`
  - Métodos: `probabilities(features, initial=None) -> pd.DataFrame`, `labels(features) -> pd.Series`, `probabilities_at(features, asof) -> RegimeProbabilities`, `transition_matrix() -> pd.DataFrame`, `expected_durations() -> pd.Series`
- `RegimeModel(ABC)`:
  - `name: ClassVar[str]`, `ranking_method: ClassVar[RankingMethod]`
  - `__init__(settings: ModelSettings)`
  - `fit(features, reference_returns, *, train_end=None, feature_names=None) -> FittedRegime`
  - Método abstracto `_estimate(matrix, seed) -> StateEstimate`

#### 2.6.3 `regimelab.models.hmm`

- `HiddenMarkovRegimeModel(RegimeModel)`
  - `name = "hmm"`, `ranking_method = RankingMethod.SHARPE`
  - `_estimate` usa `hmmlearn.hmm.GaussianHMM` con `n_init` restarts deterministas.

#### 2.6.4 `regimelab.models.gmm`

- `GaussianMixtureRegimeModel(RegimeModel)`
  - `name = "gmm"`, `ranking_method = RankingMethod.VOLATILITY`
  - `_estimate` usa `sklearn.mixture.GaussianMixture`; `transition = None`.

#### 2.6.5 `regimelab.models.hamilton`

- `MarkovSwitchingRegimeModel(RegimeModel)`
  - `name = "hamilton"`, `ranking_method = RankingMethod.SHARPE`
  - `__init__(settings, *, dependent_index=0, switching_variance=True)`
  - `_estimate` usa `statsmodels.tsa.regime_switching.markov_regression.MarkovRegression`.

#### 2.6.6 `regimelab.models.registry`

- `CLASSIFIERS: dict[str, type[RegimeModel]]` — `{hmm, gmm, hamilton}`
- `build_classifier(settings, name=None) -> RegimeModel`

#### 2.6.7 `regimelab.models.labeling`

- `RankingMethod(StrEnum)`: `SHARPE`, `MEAN_RETURN`, `VOLATILITY`
- `OccupancyError(ValueError)`
- `StateStatistics` (dataclass): `state`, `occupancy`, `mean_return`, `volatility`, `n_observations`; propiedad `sharpe`
- `LabelAssignment` (dataclass): `mapping`, `statistics`, `method`; propiedad `inverse`; métodos `regime_of(state)`, `describe() -> pd.DataFrame`
- `summarise_states(reference_returns, posteriors, *, trading_days=252) -> tuple[StateStatistics, ...]`
- `assign_labels(statistics, *, method=SHARPE, min_occupancy=0.0) -> LabelAssignment`
- `relabel_probabilities(posteriors, assignment) -> pd.DataFrame`

#### 2.6.8 `regimelab.models.filters`

- `FilterResult(NamedTuple)`: `filtered`, `predicted`, `log_likelihood`
- `gaussian_log_density(observations, means, covariances) -> np.ndarray`
- `stationary_distribution(transition) -> np.ndarray`
- `forward_filter(log_emissions, transition, initial=None) -> FilterResult` — filtro de Hamilton en log-space.

### 2.7 `regimelab.backtest` — Backtesting y Métricas

#### 2.7.1 `regimelab.backtest.engine`

- `CASH_COLUMN = "CASH"`
- `BacktestResult` (dataclass):
  - Campos: `regimes`, `target_weights`, `held_weights`, `gross_returns`, `net_returns`, `traded_notional`, `costs`, `equity`, `benchmark_returns`, `benchmark_equity`, `metrics`, `benchmark_metrics`, `regime_metrics`
  - Propiedades: `drawdown`, `benchmark_drawdown`, `excess_return`
  - Métodos: `comparison_table() -> pd.DataFrame`, `to_frame() -> pd.DataFrame`
- `target_weight_frame(regimes, settings, symbols) -> pd.DataFrame` — convierte régimen a pesos objetivo.
- `run_backtest(prices, regimes, settings, *, confidence=None, start=None) -> BacktestResult`

**Lógica de `run_backtest`**

- Rebalanceo por evento (solo cuando cambia el régimen).
- `signal_lag_days` de retardo entre señal y posición.
- `confidence_floor`: si la confianza modal es menor, se mantiene la asignación previa.
- Costes = `transaction_cost_bps / 1e4 * traded_notional`.
- El cash no asignado devenga `risk_free_rate` diario.
- No apalancamiento, no shorting.

#### 2.7.2 `regimelab.backtest.metrics`

- `equity_curve(returns, initial=1.0) -> pd.Series`
- `drawdown_series(equity) -> pd.Series`
- `PerformanceMetrics` (Pydantic):
  - Campos: `n_observations`, `start`, `end`, `total_return`, `cagr`, `annual_volatility`, `sharpe`, `sortino`, `max_drawdown`, `calmar`, `hit_rate`, `skewness`, `excess_kurtosis`, `var_95`, `cvar_95`, `average_exposure`, `annual_turnover`, `total_cost`
  - Método: `summary_rows() -> list[tuple[str, str]]`
- `compute_metrics(returns, *, trading_days=252, risk_free_rate=0.0, weights=None, turnover=None, costs=None) -> PerformanceMetrics`
- `regime_conditional_metrics(returns, regimes, *, trading_days=252, order=None) -> pd.DataFrame`

#### 2.7.3 `regimelab.backtest.walkforward`

- `WalkForwardFold` (dataclass): `fold_index`, `train_start`, `train_end`, `test_start`, `test_end`, `n_train`, `n_test`, `model: ModelMetadata`, `assignment: pd.DataFrame | None`
- `WalkForwardResult` (dataclass): `labels`, `probabilities`, `confidence`, `folds`, `features`, `backtest: BacktestResult | None`
  - Propiedad `regimes` (alias de `labels`)
  - Método `regime_probabilities(asof) -> RegimeProbabilities`
- `run_walk_forward(panel, features, settings, *, run_backtest=True) -> WalkForwardResult`

**Lógica walk-forward**

- Ventana expandiente: cada refit usa todo el historial hasta `train_end`.
- Refit en fechas determinadas por `min_train_observations` y `refit_frequency_days`.
- Entre refits, el modelo está congelado; se avanza el filtro causal.
- Se reutiliza el posterior filtrado del final de un segmento como prior del siguiente para preservar continuidad de Markov.
- El último segmento alcanza el final del panel.

### 2.8 `regimelab.pipeline` — Orquestación

- `PipelineResult` (dataclass): `settings`, `panel`, `features`, `walkforward`, `manifest`, `store`
  - Propiedad `backtest -> BacktestResult | None`
- `latest_telemetry(result) -> TelemetryPayload`
- `run_pipeline(settings, *, command="run", run_backtest=True) -> PipelineResult`
- `run_single_asof(settings, asof, *, command="asof") -> TelemetryPayload`
- `RunCommand` (callable): wrapper para CLI/tests.

**Artefactos escritos por `run_pipeline` en el `RunStore`**

- `manifest.json`
- `features.parquet`
- `regime_probabilities.parquet`
- `regime_labels.parquet`
- `equity.parquet` (solo si `run_backtest=True`)

### 2.9 `regimelab.report` — Reportes

- `HTMLReport(result)`:
  - `render() -> str` — HTML autocontenido con CSS inline y gráficos Plotly vía CDN.
- `write_html_report(result, path) -> Path`

Secciones del reporte:

1. Performance summary (tarjetas: CAGR estrategia/benchmark, excess return, max drawdown, Sharpe, annual turnover).
2. Regime-conditional metrics (tabla por régimen).
3. Latest feature snapshot (tabla de features).
4. Gráficos: equity, drawdown, regime probabilities (stacked area).

### 2.10 `regimelab.tui` — Dashboard Textual

- `RegimeLabApp = _LazyApp()` — proxy lazy que importa `textual` solo si está instalado.
- `RegimeLabApp(result).run()` — arranca el dashboard.
- Pantallas: métricas del backtest, feature snapshot, probabilidades de régimen, target weights.

### 2.11 `regimelab.store` — Provenance y Artefactos

- `GitState` (Pydantic): `commit`, `branch`, `dirty`
- `RunManifest` (Pydantic):
  - `run_id`, `created_at`, `command`, `asof`, `git`, `python_version`, `platform`, `packages`, `seed`, `settings`, `input_digest`
  - Propiedades: `is_reproducible`, `provenance_line()`
- `RunStore(root, manifest)`:
  - `directory`, `manifest`
  - `write_manifest() -> Path`
  - `write_frame(frame, name) -> Path`
  - `write_json(payload, name) -> Path`
  - `write_text(content, name) -> Path`
- `digest_frame(frame) -> str` — SHA-256 del contenido del DataFrame/Series.

---

## 3. Estructuras de Datos de Salida

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

- `asof`: fecha de decisión.
- `regime`: régimen modal.
- `probabilities`: distribución posterior completa (dict `MarketRegime -> float`).
- `features`: snapshot causal de riesgo sistémico.
- `model`: metadatos del fit (nombre, semilla, rango de entrenamiento, etc.).
- `target_weights`: pesos objetivo del régimen actual, o `None` si no hay backtest configurado.
- `warnings`: advertencias operacionales.

**Validadores**

- `model.train_end <= asof` (última línea de defensa contra look-ahead).
- `regime` debe aparecer en `probabilities`.

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

- `settings`: configuración resuelta.
- `panel`: panel alineado PIT.
- `features`: matriz de features causal completa.
- `walkforward`: resultado del bucle walk-forward y backtest opcional.
- `manifest`: provenance del run.
- `store`: directorio de artefactos (accesor).
- Propiedad `backtest`: `result.walkforward.backtest`.

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

- `target_weights`: pesos teóricos por sesión.
- `held_weights`: pesos retenidos después del lag y del confidence floor.
- `gross_returns`: retorno bruto antes de costes.
- `net_returns`: retorno neto (métricas).
- `traded_notional`: turnover en cada sesión.
- `costs`: coste de transacción por sesión.
- `metrics` vs `benchmark_metrics`: métricas de la estrategia y del buy-and-hold.
- `regime_metrics`: estadísticas condicionales por régimen del benchmark.

### 3.4 `WalkForwardResult`

```python
WalkForwardResult(
    labels=pd.Series,          # régimen modal por sesión
    probabilities=pd.DataFrame, # posteriors por régimen
    confidence=pd.Series,       # confianza modal
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

### 3.6 Esquema del `manifest.json`

Ejemplo de campos (Pydantic serializado a JSON):

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
    "regimelab": "0.1.0",
    "numpy": "2.5.2",
    "pandas": "3.0.5",
    ...
  },
  "seed": 20260831,
  "settings": { /* Data, Feature, Model, Backtest, artifacts_dir */ },
  "input_digest": "sha256..."
}
```

**Significado**

- `run_id`: identificador único del run.
- `command`: comando CLI que originó el run.
- `asof`: fecha as-of para el comando `asof`; `null` en `run`.
- `git`: estado del repositorio.
- `packages`: versiones de paquetes que afectan resultados numéricos.
- `settings`: configuración resuelta; permite reproducir el run sin el entorno original.
- `input_digest`: hash del panel de entrada (`digest_frame(panel.close())`).

### 3.7 Archivos Parquet en `.artifacts/`

Layout del directorio de artefactos:

```text
.artifacts/
  <YYYYMMDD-HHMMSS>-<run_id>/
    manifest.json
    features.parquet
    regime_probabilities.parquet
    regime_labels.parquet
    equity.parquet  # solo si hay backtest
```

- `features.parquet`: `pd.DataFrame` con índice `date` y columnas del feature matrix.
- `regime_probabilities.parquet`: posteriores por régimen canónico; columnas `BULL_TREND`, etc.
- `regime_labels.parquet`: etiqueta modal por sesión.
- `equity.parquet`: curva de equity de la estrategia.

Además, el cache de paneles (`settings.data.cache_dir / panels / <sha16>.parquet`) persiste paneles descargados para evitar redes repetidas.

---

## 4. Lista de Tests

Ficheros: `tests/conftest.py`, `tests/test_backtest.py`, `tests/test_causality.py`.

### 4.1 Fixtures compartidas (`conftest.py`)

- `test_settings: Settings` — configuración de test con provider `synthetic`, fechas 2015-2020, y ventanas acortadas para rapidez.
- `pipeline_result: PipelineResult` — pipeline completo generado una vez por sesión.

### 4.2 `tests/test_backtest.py`

1. **`test_held_weights_use_only_past_targets`**  
   Verifica que cada asignación retenida en `t` coincide con un peso objetivo de una sesión anterior o igual a `t - signal_lag_days`.

2. **`test_transaction_costs_reduce_returns`**  
   `gross_returns >= net_returns` sesión a sesión.

3. **`test_weights_sum_to_one`**  
   Tanto `target_weights` como `held_weights` suman 1.0 en cada sesión.

4. **`test_confidence_floor_preserves_previous_allocation`**  
   Cuando la confianza modal está por debajo de `confidence_floor`, la asignación retenida es idéntica a la del día anterior.

### 4.3 `tests/test_causality.py`

**`TestPointInTimeBoundary`**

5. **`test_truncate_asof_does_not_exceed_cursor`**  
   Para cada fecha del calendario, `truncate_asof` devuelve un índice cuyo máximo es exactamente esa fecha.

6. **`test_truncate_asof_property` (@given 20 fechas aleatorias, Hypothesis)**  
   Propiedad: el frame truncado no contiene filas posteriores al cursor.

**`TestFeaturesAndBacktest`**

7. **`test_feature_snapshot_exists_and_is_single_row`**  
   `snapshot_at` devuelve un `FeatureSnapshot` válido para una fecha intermedia.

8. **`test_backtest_held_weights_use_only_past_targets`**  
   Versión causal del test de backtest: los held weights provienen exclusivamente de targets pasados.

**`TestWalkForward`**

9. **`test_folds_are_sequential`**  
   Para cada fold: `train_start <= train_end <= test_end` y `test_start <= test_end`.

10. **`test_test_windows_advance`**  
    Las ventanas de test cubren la totalidad de la muestra sin solapamientos perdidos entre folds consecutivos.

**`TestAsofPayload`**

Parametrizado para tres fechas: `2020-03-16`, `2020-06-15`, `2020-12-31`. Cuenta como 3 casos de test.

11-13. **`test_asof_payload_uses_only_history_before_cursor`**  
    `run_single_asof` devuelve un `TelemetryPayload` cuyo `asof` coincide con el solicitado y cuyo `features` está disponible en esa fecha. Asegura que el endpoint as-of no mira más allá del cursor.

**Total: 13 casos de test** (4 en `test_backtest` + 2 + 2 + 2 + 3 parametrizados en `test_causality`).

---

## 5. Flujo de Datos End-to-End

```text
[Provider] -> [PanelFetcher + cache] -> [align_panel]
    -> [build_feature_matrix]
    -> [run_walk_forward: fit / filter / backtest]
    -> [PipelineResult]
    -> [latest_telemetry | HTMLReport | RegimeLabApp | CLI output]
```

Cada caja de texto negrita del flujo produce un artefacto o un Pydantic object inmutable, de modo que el punto de decisión final es siempre un `TelemetryPayload` validado contra look-ahead.
