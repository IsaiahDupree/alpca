# Alpca architecture

A focused Alpaca **paper**-trading bot whose defining feature is that **execution
latency is a measured quantity**. Every order carries its full lifecycle as
wall-clock timestamps, and the bot reports per-stage latency percentiles plus
slippage vs. the backtest's modeled fill.

## The latency model (the point of the project)

Every `Order` (`alpca/execution/order.py`) records four lifecycle timestamps:

```
signal_ts ──▶ submit_ts ──▶ ack_ts ──▶ fill_ts
   │             │            │           │
   └ strategy    └ we called  └ broker    └ terminal
     emitted       the API      accepted    filled
```

From these we derive, per order:

| metric                | meaning                                            |
|-----------------------|----------------------------------------------------|
| `signal_to_submit_ms` | our own overhead: signal → API call                |
| `submit_to_ack_ms`    | network + broker acceptance round trip             |
| `ack_to_fill_ms`      | time resting in the book until filled              |
| `signal_to_fill_ms`   | end-to-end: idea → done                            |
| `slippage_bps`        | signed fill vs. the price the strategy intended    |

`alpca/metrics/latency.py` aggregates these across all orders into
mean/p50/p95/p99/max tables. The same `slippage_bps` lets you compare a live
paper fill against the backtest's modeled fill (the backtest records the exact
`slippage_bps`/`commission_bps` it assumed).

## Flow

```
        bars (Alpaca / synthetic)
               │
          Strategy.on_bar(bar) ──▶ Signal(side, strength, price, ...)
               │   (signal_ts stamped here, intended_price = price)
               ▼
        ExecutionRouter.submit()              ← single chokepoint
           1. idempotency (dedupe client_order_id)
           2. log SIGNAL
           3. RiskEngine.check()  ──▶ deny → log RISK_BLOCK, reject
           4. submit_ts; perf_counter start; log SUBMIT
           5. adapter.submit() ─┬─ SimAdapter   (offline: injected latency+slippage)
                                └─ AlpacaAdapter (real paper: alpaca-py via to_thread)
           6. poll until terminal / timeout; record rtt_ms
           7. log ACK / FILL / REJECT
               │
               ▼
        OrderEventLog (hash-chained JSONL)  +  LatencyReport
```

## Components

| module | role |
|--------|------|
| `alpca/config.py` | env-driven config; **PAPER by default**, live double-gated |
| `alpca/execution/order.py` | `Order`/`Fill` + lifecycle timestamps + latency/slippage props |
| `alpca/execution/order_event_log.py` | append-only **SHA-256 hash-chained** ledger; `verify_chain()` |
| `alpca/execution/router.py` | risk-gated `submit()`; times every stage; `rtt_stats()` |
| `alpca/execution/adapters/base.py` | thin async adapter contract |
| `alpca/execution/adapters/sim.py` | offline fills with injected latency + slippage (no creds) |
| `alpca/execution/adapters/alpaca.py` | real Alpaca paper via `alpaca-py`, run off-loop with `to_thread` |
| `alpca/risk/risk_engine.py` | pre-trade gate: notional/daily-loss/concentration/positions/rate |
| `alpca/strategies/` | `Strategy.on_bar` contract + Donchian, ORB, Z-score |
| `alpca/backtest/engine.py` | event-driven backtester with cost model + metrics |
| `alpca/metrics/latency.py` | latency + slippage percentile reports |
| `alpca/data/bars.py` | Alpaca historical bars + deterministic synthetic bars |

## Design choices

- **One chokepoint.** All orders go through `ExecutionRouter.submit()`, so risk,
  idempotency, latency timing, and the audit ledger live in exactly one place —
  no strategy or adapter has to think about them.
- **Adapters are async + thin.** The Alpaca SDK is synchronous, so its calls run
  via `asyncio.to_thread` to keep the loop responsive (which keeps latency
  numbers honest).
- **Sim mirrors paper.** `SimAdapter` implements the same `BaseAdapter` contract,
  so the full pipeline — including the latency ledger and metrics — runs in CI
  with zero credentials; flipping to real paper is one adapter swap.
- **Tamper-evident audit.** The order ledger is hash-chained: any edit to a past
  event breaks the chain at the next event (`verify_chain()` reports where).

## rtt vs. lifecycle latency

Two complementary measurements:

- **`rtt_ms`** (router `_rtt_samples` / `rtt_stats()`) — wall-clock from just
  before `adapter.submit()` to terminal status. Only counts orders that actually
  reached the broker (a risk-blocked order has no round trip). This is the
  "how fast did the broker turn my order around" number.
- **Lifecycle latencies** (on each `Order`) — the finer `signal→submit→ack→fill`
  breakdown, including our own pre-submit overhead and book-resting time.

## Ported from

The strategy logic, risk gates, hash-chained ledger pattern, router-rtt idea, and
Alpaca adapter shape are distilled from the local `../TradingBot` "Star
Algorithm" platform (and its sibling `../TradingBot2`), trimmed to just the
trading loop + measurement — no users/funding/dashboard/SaaS layers.
