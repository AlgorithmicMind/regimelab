# Next-open execution: implementation specification

Status: **Implemented on the local development branch; not released or
integrated into Aegis.** The selected-HMM-restart metadata fix is independent
of this API.

This specification introduces an explicit, opt-in position ledger. It does not
change the existing `simulate_weights` / `run_backtest` numeric convention,
replace their golden evidence, select an allocation, or authorize a release.
The existing integer-lag overlay is not a substitute for this ledger.

## 1. Public API and compatibility

`NextOpenResult` and `simulate_next_open` are defined in
`regimelab.backtest.portfolio` and exported from `regimelab.backtest`:

```python
def simulate_next_open(
    prices: pd.DataFrame,
    target_weights: pd.DataFrame,
    settings: Settings,
    *,
    open_prices: pd.DataFrame,
    sessions: pd.DataFrame,
    published_at: pd.Series | None = None,
    open_observed: pd.DataFrame | None = None,
    close_observed: pd.DataFrame | None = None,
    start: dt.date | None = None,
) -> NextOpenResult:
    ...
```

- `prices` contains consistently adjusted closes, not returns. `open_prices`
  uses the same adjustment convention. Fractional adjusted units are research
  units, not a claim about physical share counts or actual fills.
- `settings.data.risk_assets` defines allowable risky sleeves; the benchmark is
  `settings.data.benchmark`. Residual `CASH` is not a traded instrument.
- Target rows are close-of-session decisions. Reuse the legacy weight
  validation where possible: finite long-only weights, no leverage, known
  columns and residual cash. Clip only the existing 1e-9 bookkeeping tolerance;
  if the risk sum exceeds one within tolerance, normalize that residue to one.
- An entirely unpriced risk asset is allowed only when it never has positive
  target weight. Unused zero-weight assets must not contaminate valuations.
- The function is explicitly next-open. Reject
  `settings.backtest.signal_lag_days != 1` rather than silently interpreting an
  old lag-two configuration as next-open execution.
- Allocation logic, confidence gating and regime inference belong to the
  caller. The simulator neither fits a model nor substitutes regime weights.
- Keep the old API and its result type intact. The new result has separate
  opening/closing weights because no single vector represents an entire day
  containing a trade.

## 2. Calendar and publication ordering

All three price/schedule indexes are identical, ascending, unique, normalized
timezone-naive session labels. No extra, omitted or duplicate sessions are
silently repaired. Target dates follow the same ordering/uniqueness rules and
must be sessions in this calendar.

`sessions` has timezone-aware `market_open` and `market_close` columns, with
nonmissing timestamps, open before close, and each close before the next open.
The caller supplies the actual exchange schedule, including holidays and
shortened sessions. The simulator does not infer a calendar from weekdays,
download one, or invent a missing exchange session from price availability.

When supplied, `published_at` is a timezone-aware Series indexed exactly like
the target decisions. Each timestamp must be at or after that decision
session's close. For a historical research run with no recorded publication
times, default availability is that close; disclose that this is an assumption,
not observed historical publication evidence.

At each opening, use the newest observation-date decision whose availability
is strictly before that opening. A decision published exactly at the opening
does not receive that auction price. A delayed older observation never rolls
the portfolio back after a newer observation has become available. No
instruction means retain positions, not liquidate to cash.

This is a reference-price eligibility rule, not a guarantee of access to an
opening auction. Production callers must also enforce the broker's earlier
order-submission deadline and operational delivery cutoff.

When `start` is absent, simulation begins on the first scheduled session after
the earliest decision session, independently of target content or publication
latency. For an empty decision frame it begins at the first price session.
An explicit `start` selects the first session on/after that date and defines
portfolio inception, not a retrospective performance crop. Decisions before
inception may be acted on at its opening if already available. Require at
least two evaluated sessions. Initial capital is cash at the previous close.

## 3. Prices and observation masks

Never forward-fill prices or returns inside this API. Required valuation or
execution prices must be finite and strictly positive. Include symbol/session
in errors. A sold asset needs an opening price, but not a later closing price
if no units remain; the benchmark still needs its own valuation prices.

Optional `open_observed` and `close_observed` are boolean, nonmissing frames on
the same session index. False marks a carried/unobserved value as unusable even
when its numeric price is finite. Missing mask columns mean unobserved.
Check masks only where the corresponding price is used. Without masks, the
caller vouches that supplied nonmissing prices are genuine observations.
Production must provide provenance-derived masks or genuinely unfilled inputs;
numeric prices alone cannot prove the absence of upstream forward-filling.

## 4. Cash, costs and self-financing rebalances

Use `initial_capital`, `risk_free_rate` and `trading_days_per_year` from settings.
Require finite positive initial capital and a finite nonnegative cash rate.
Credit previous-close cash once before each evaluated opening at
`daily_rf = risk_free_rate / trading_days_per_year`, including the inception
session. Cash produced by today's trade first earns interest next session.
This is a constant-rate, trading-session approximation, not an actual money
market yield or calendar-day accrual model.

For this API, `transaction_cost_bps / 10000` is an all-in proportional cash
charge per unit of actual ETF purchase/sale notional at the reference open.
It may represent combined fees/spread/slippage; do not add an undisclosed
second slippage adjustment. Both risky buys and risky sells count. Residual
cash movements do not. Require a finite cost rate `0 <= c < 1`.
The numeric default 5 bps is an assumption, not an empirical estimate. This
cost basis differs from legacy cash-inclusive turnover and must be disclosed.

Let `V` be opening NAV before fees, `B_i` the existing risky position values
at that open, and `w_i` the new risky target fractions of post-fee NAV.
Solve the self-financing equation for post-fee capital `N`:

```text
N + c * sum(abs(w_i * N - B_i)) = V
```

For long-only weights summing to at most one and `c < 1`, the left side is
strictly increasing. The root is bracketed by zero and `V`. A deterministic
bisection (80 iterations, return the lower bracket) is sufficient; zero cost
returns `V` directly. These numerical choices are not strategy parameters.

Set new risky units to `w_i * N / open_i`. Charge fees on actual differences
between new and old opening position values. Compute cash as the original
cash minus signed risky purchases and fees, not as a leveraged borrowing.
Reconcile post-trade NAV, fees, units and cash; tolerate only floating-point
residue, never a real negative cash balance or nonfinite/invalid NAV.

Adopt a new target only if it differs from the last adopted target by more
than the existing 1e-9 weight threshold. Compare targets, not actual drifted
weights. Repeated identical instructions do not authorize daily rebalancing.
Keep units unchanged between adoption events, even when weights drift.

## 5. Result and benchmark

`NextOpenResult` is an immutable dataclass container with:

- `target_weights`: normalized input decisions, not fictitious daily orders.
- `decision_at_open`, `execution_times`: eligible observation dates and the
  scheduled openings; distinguish eligible decisions from actual nonzero fills.
- `positions`, `cash_balance`, `cash_interest`: post-trade adjusted units and
  closing cash, plus interest credited in currency.
- `nav_before_trade`, `nav_after_trade`, `nav`: opening values before/after
  fees and closing portfolio value, all in currency.
- `opening_weights`, `closing_weights`: actual weights at those marks.
- `traded_units`, `traded_values`: signed ETF fill quantities/notionals.
- `fees`: currency costs; `costs`: fees divided by previous closing NAV.
- `net_returns`: closing NAV divided by previous closing NAV minus one;
  inception uses initial capital as denominator. `equity = nav / initial_capital`.
- `traded_notional`: gross risky fill notional divided by pre-trade opening
  NAV. This is gross traded exposure, not half-turnover.
- `metrics`, `benchmark_returns`, `benchmark_equity`, `benchmark_metrics`.

Calculate metrics from the net ledger using existing metric helpers. For
exposure reporting use actual opening weights and disclose that convention.
`to_frame()` must expose the ledger and decision/fill timing for audit. Do not
invent a single all-day `held_weights` series or an unexplained no-cost
counterfactual gross-return series.

Drawdowns include initial equity 1.0 as a possible peak, so an inception
fee/loss is not hidden by starting the peak at the first marked NAV.
`drawdown_series` accepts optional keyword `initial: float | None = None` and
`compute_metrics` accepts `initial_equity: float | None = None`; legacy
behavior is preserved when these are absent. The new ledger uses `initial=1.0` and
`initial_equity=1.0`, including when calculating Calmar.

The benchmark buys the configured benchmark asset at the first evaluated
opening, uses the same initial cash interest and per-leg cost convention,
then holds units without daily rebalancing. It uses the same ledger, not a
separate accounting implementation. Benchmark and strategy share inception
and evaluated sessions. Static allocation comparators must also use this
ledger; full-period mean weights remain an ex-post diagnostic.

## 6. Acceptance and validation

Use deterministic synthetic inputs; freeze their expected answers before
writing implementation. No market-data downloads or E0-E4 reruns in this step.

- Cash through close 100, buy at open 103, close 104: return `104/103 - 1`.
- Hold one unit at close 100, sell at open 90, close 80: closing cash 90.
- Two units, one in each asset at 100; no order, closes 120/100: NAV 220,
  weights `6/11` and `5/11`, no restorative rebalancing.
- With 1% synthetic costs, cash 1000 into a fully invested target at price 100:
  post-fee NAV `1000/1.01`, fees `1000 - 1000/1.01`, cash zero.
- Sell a position worth 1000 with 1% costs: cash 990, fees 10.
- Rebalance existing values 600/400 to 40/60 with 1% costs:
  post-fee NAV `998/1.002`; verify the equation, actual fees and zero cash.
- Pure cash has zero risky trades/fees and compounds the stated session rate.
- With flat prices and an inception fee, drawdown includes the loss relative
  to initial capital; it is not zero just because later prices remain flat.
- Validate sparse/no decisions, duplicate/unsorted indexes, invalid weights,
  inconsistent schedules, naive/missing publication timestamps, late and
  out-of-order publications, and calendar gaps/holidays/short sessions.
- Test missing/zero/negative/infinite used prices and false observation masks;
  an unused missing price must not poison valid P&L.
- The ledger must conserve value under nonzero fees, keep cash/units
  nonnegative, avoid a forced terminal liquidation, and leave inputs unchanged.
- Verify exact legacy golden preservation and selected-fit HMM convergence
  metadata independently. Neither fix selects a model or allocation.

The existing Aegis experiment starts ingestion in 2008; the library/daily
adapter defaults to 2005. Neither date is selected here as a single production
start. Initial dates, allocation rules, HMM/GMM evaluation, production retries
and publication rollout remain separate decisions.
