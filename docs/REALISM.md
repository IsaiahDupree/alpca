# Realism: how close is the sim to real Alpaca?

Two independent adversarial audits (mining the production `TradingBot` platform +
reviewing this code) scored the *original* sim **34/100** and ranked 18 gaps. This
doc tracks what's now modeled, what's deliberately deferred, and why.

## The headline: friction matters

Same strategy (Donchian), same 600 synthetic bars, three cost models:

| cost model | total return |
|---|---|
| frictionless (0 bps) | **+17.9%** |
| flat 2 bps slippage | **+4.3%** |
| realistic (spread + sqrt-impact + 10% volume cap + SEC/TAF fees) | **+2.9%** |

A frictionless backtest overstates this strategy's return by ~6×. The realism
layer exists so the number you see is the number you could actually trade.

## ✅ Modeled now

| area | what | where |
|---|---|---|
| **No look-ahead** | a signal from bar *i* fills at bar *i+1*'s **open**, never bar *i*'s own close (one-bar `pending` deferral) | `backtest/engine.py` |
| **Bid/ask spread** | side-aware half-spread: buy at +half-spread, sell at −half-spread | `execution/fills.py` |
| **Market impact** | square-root law: `impact_bps = coef · √(qty / bar_volume)` — bigger orders pay more | `execution/fills.py` |
| **Volume cap / partial fills** | an order can take at most `participation_cap` of bar volume; excess is a partial fill | `execution/fills.py` |
| **Tick rounding** | fills round to the $0.01 legal tick | `execution/fills.py` |
| **Alpaca fees** | $0 commission + **SEC §31** (sell notional) + **FINRA TAF** (per-share sold, capped $8.30) | `execution/fees.py` |
| **Costed EOD liquidation** | end-of-data liquidation pays the same spread/impact + fees as any exit (no free unwind) | `backtest/engine.py` |
| **Split/dividend adjustment** | `fetch_alpaca_bars(adjustment=...)` defaults to `"all"` (split+dividend) for signal/return continuity; `"raw"` for live-fill parity; recorded on every bar | `data/bars.py` |
| **Calendar enforcement** | opt-in `require_regular_hours`: fills happen only on tradeable NYSE bars; a signal landing off-session carries forward to the next regular open (no off-hours fills). Runner skips off-session submissions | `backtest/engine.py`, `runtime/runner.py` |
| **Position averaging** | a 2nd BUY weighted-averages into the existing position (shares added, cost basis blended); partial SELL keeps the remainder at cost — not overwrite | `runtime/runner.py` |
| **Dividend cash flows** | held position crossing a cash-dividend ex-date is credited qty×amount (use with raw/split bars; adjusted bars already include it) | `data/corporate_actions.py`, `backtest/engine.py` |
| **Limit through-trade fills** | a LIMIT fills only if the bar's price traded through it (buy: low≤limit; sell: high≥limit); gap-through gives price improvement; volume-cap partials; no-fill leaves it resting | `execution/fills.py` (`fill_limit`), `execution/adapters/sim.py` |
| **T+1 settlement** | opt-in cash account: sale proceeds are unsettled until the next trading session; a BUY can only use settled cash | `runtime/account.py` (`SettlementLedger`), `runtime/runner.py` |
| **PDT day-trade guard** | opt-in: a <$25k account is blocked from a 4th day-trade in any rolling 5 sessions; inactive ≥$25k | `runtime/account.py` (`PdtGuard`), `runtime/runner.py` |
| **Calibrated sim latency** | `SimAdapter.paper_calibrated()` preset = measured ~248ms submit→ack from real paper (vs 8ms toy default) | `execution/adapters/sim.py` |
| **Order lifecycle** | `OpenOrderBook`: LIMIT/STOP/STOP_LIMIT rest across bars; DAY orders expire at next session open, GTC persist; stops trigger on touch then fill; partial fills leave remainder resting; cancel + cancel-replace | `execution/open_orders.py` |
| **Short selling** | signed positions (long/short) via one verified `apply_fill` (open/add/reduce/close/flip); risk gate blocks shorts unless `allow_short`; signed concentration; daily short-borrow fee; symmetric long/short z-score strategy (`zscore-ls`) | `runtime/position_math.py`, `risk/risk_engine.py`, `runtime/account.py` (`BorrowFeeLedger`), `runtime/runner.py` |
| **Resting orders wired to the loop** | strategies emit LIMIT/STOP intents (`Signal.order_type`); `LiveRunner` rests them in the book, advances them intrabar each bar (trigger/fill/expire → accounting + ledger + `strategy.on_fill`), and cancel-replaces as the level moves. Donchian `entry="stop"` rests a buy-stop at the channel high (turtle execution) | `strategies/base.py`, `strategies/breakout.py`, `runtime/runner.py` |
| **Runner-driven backtest analytics** | `backtest_resting()` runs a strategy through the full runner + order book and returns a `BacktestResult` (total_return / Sharpe / maxDD / win_rate / n_trades + per-bar equity curve). Resting-order strategies now get the same analytics as the next-open `run_backtest`, reusing the tested book integration rather than duplicating it | `backtest/runner_backtest.py`, `runtime/runner.py` (`to_result`) |
| **Buying power** | a BUY whose notional exceeds available cash is rejected (cash-account, no margin) | `risk/risk_engine.py` |
| **Pre-trade risk** | notional cap, daily-loss auto-halt, concentration, max positions, orders/min rate limit | `risk/risk_engine.py` |
| **Market calendar** | NYSE session classifier (REGULAR/PRE/AFTER/CLOSED + holidays + half-days, 2024–2027) | `data/calendar.py` |
| **Tamper-evident audit** | SHA-256 hash-chained order ledger | `execution/order_event_log.py` |
| **Lifecycle latency** | signal→submit→ack→fill timestamps + slippage, p50/p95/p99 | `execution/order.py`, `metrics/latency.py` |

## ⚠️ Deferred (known gaps, ranked)

These are documented, not silently ignored. None blocks offline research; each
matters before trusting live size/timing.

**High value**
- ~~Limit-order through-trade test~~ — DONE (`fill_limit`).
- ~~Order lifecycle (resting / DAY-GTC expiry / cancel-replace / stop triggers)~~
  — DONE (`OpenOrderBook`).
- ~~Wire the order book into the trading loop~~ — DONE: `LiveRunner` now accepts an
  `open_order_book`, advances it each bar, and routes `Signal` LIMIT/STOP intents
  into it (Donchian `entry="stop"`).
- ~~Order book in the backtester / rich analytics for resting strategies~~ — DONE
  via `backtest_resting()` (runs through the runner + book, returns a full
  `BacktestResult`). The procedural `run_backtest` intentionally keeps the simpler
  next-open market model; resting-order backtests go through `backtest_resting`.
- **Live dividend fetch wiring** — `fetch_alpaca_dividends` is implemented
  against Alpaca's corporate-actions endpoint but unverified against a live
  account (no creds yet); the backtester/runner don't auto-fetch it.

**Medium / low**
- Sub-penny guard is in the fill model (`min_tick`), but MOC/auction-imbalance
  slippage on close orders is not modeled.
- LULD halts, missing/zero-volume bars, opening/closing auctions.
- Latency that *moves the fill price* (currently latency is measured but a fixed
  delay doesn't re-price the fill against a later snapshot).
- Short selling is now modeled (signed positions + borrow fee), but short-sale
  **locate / easy-to-borrow (ETB) checks and SSR** (short-sale restriction on a
  −10% day) are not — any symbol is assumed freely shortable.
- Self-cross / wash-trade detection.
- Survivorship / point-in-time universe (matters for multi-symbol studies).

## Calibration note

The fill-model coefficients (`half_spread_bps`, `impact_coef_bps`,
`participation_cap`) are sensible defaults, **not** calibrated to a specific
symbol. Once real Alpaca paper fills are collected, fit them against the
`parity` report's realized-vs-modeled slippage so the sim matches the names you
actually trade.
