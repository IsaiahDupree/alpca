# Alpca — Alpaca paper-trading bot with latency metrics

A focused, runnable paper-trading bot for **Alpaca**, built to make **execution
latency a first-class, measured quantity**. Every order carries timestamps for
its full lifecycle — **signal → order → ack → fill** — and the bot reports
per-stage latency percentiles (p50/p95/p99/max) and **slippage vs. the
backtest's modeled fill**.

It ports the proven, verified pieces of the local `TradingBot` "Star Algorithm"
platform (strategy contract, risk-gated execution router, Alpaca adapter,
hash-chained order ledger, backtester/metrics) into a single clean package — no
SaaS layers (no users/funding/dashboard), just the trading loop and the
measurement.

## Why latency-sensitive strategies

The first strategy set is deliberately latency-sensitive — breakout/ORB and
fast-bar mean-reversion — because those are the strategies where the gap between
*backtest fill* and *real paper fill* actually shows up. That gap is exactly what
the latency + slippage harness measures.

## Layout

```
alpca/
  config.py              # env-driven config; PAPER by default, live double-gated
  envfile.py             # load creds from an external .env into this process only
  strategies/            # Signal contract + Donchian, ORB, Z-score (+ registry)
  execution/
    order.py             # Order/Fill + lifecycle timestamps + latency/slippage
    order_event_log.py   # append-only hash-chained latency ledger
    router.py            # single risk-gated submit() chokepoint; times every stage
    adapters/            # BaseAdapter -> AlpacaAdapter (paper), SimAdapter (offline)
  data/bars.py           # Alpaca historical bars + deterministic synthetic bars
  risk/risk_engine.py    # pre-trade gate (notional/loss/concentration/positions/rate)
  backtest/engine.py     # event-driven backtester w/ cost model + metrics
  metrics/latency.py     # PnL + latency + slippage reporting
scripts/
  probe_alpaca.py        # live paper connectivity + REST latency (no orders)
  smoke_alpaca_paper.py  # place ONE tiny paper order, measure full lifecycle latency
  demo_offline.py        # full pipeline on SimAdapter — NO credentials needed
tests/                   # offline tests (SimAdapter; no credentials)
```

## Setup

```bash
cd Alpca
python3 -m venv .venv && source .venv/bin/activate
pip install -e .            # or: pip install -r requirements.txt
cp .env.example .env        # fill in ALPACA_API_KEY / ALPACA_SECRET_KEY (paper)
```

## Run

```bash
# Verify config + credentials (prints NO secret values)
python -m alpca.cli config

# Offline, no credentials: full latency+slippage pipeline on the sim broker
python scripts/demo_offline.py --strategy donchian --bars 500

# Offline backtest of a strategy on synthetic bars
python -m alpca.cli backtest --strategy donchian --offline

# Real Alpaca paper: connectivity + REST latency (no orders placed)
python scripts/probe_alpaca.py            # or --env-file /path/to/.env

# Real Alpaca paper: one tiny order, full signal->fill latency report
python scripts/smoke_alpaca_paper.py
```

## Safety

- **PAPER by default.** Live requires `ALPACA_PAPER=0` **and**
  `ALPACA_LIVE_CONFIRMED=I_UNDERSTAND`.
- Every order passes the pre-trade `RiskEngine` (notional cap, daily-loss
  auto-halt, concentration cap, max positions, orders/min rate limit) before any
  broker call.
- Credentials are read from the environment only and never logged.

## Status

Foundations complete and unit-tested offline (order lifecycle, hash-chained
ledger, risk engine, sim-broker round trip, strategies, backtester). The real
Alpaca paper adapter is implemented; running it requires paper API credentials
(none are committed). See `docs/ARCHITECTURE.md`.
