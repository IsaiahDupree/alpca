# Real Alpaca paper baseline

First live measurements against the Alpaca **paper** endpoint
(`paper-api.alpaca.markets`), captured 2026-05-31 (market closed — a Sunday).
Reproduce with `python scripts/probe_alpaca.py` and `python scripts/smoke_alpaca_paper.py`.

## Account
- status ACTIVE, cash $100,000, buying power $200,000, paper mode.

## REST round-trip latency (5 samples each)
| call | p50 | p95 | min | max |
|---|---|---|---|---|
| `get_account` | 168.4 ms | 198.2 ms | 158.1 ms | 210.6 ms |
| `get_clock` | 142.7 ms | 155.0 ms | 138.4 ms | 161.2 ms |
| `latest_quote(SPY)` | 121.3 ms | 144.9 ms | 118.6 ms | 158.9 ms |

## Order lifecycle latency (one resting limit, market closed)
SPY quote bid 585.60 / ask 585.78 / mid 585.69; BUY 1 LIMIT, ACCEPTED, then canceled.

| stage | latency |
|---|---|
| signal → submit (our overhead) | 0.2 ms |
| submit → ack (network + broker accept) | **247.8 ms** |
| router rtt (submit → terminal) | 372.1 ms |
| ack → fill | n/a (market closed; order rested then canceled) |

Hash-chained ledger valid (3 events: SIGNAL/SUBMIT/ACK). Account left clean
(0 open orders, 0 positions).

## What this calibrates
- The **submit→ack ~248 ms** is the real network + broker acceptance cost. The
  SimAdapter's default injected latencies (submit 5 ms, ack 8 ms) are far
  optimistic vs. real REST. **DONE:** use `SimAdapter.paper_calibrated()` — a
  preset with submit 120 ms + ack 128 ms = ~248 ms matching this measurement
  (fill_latency is still a ~60 ms placeholder pending a market-hours fill).
- **ack→fill** and **slippage_bps** still need a MARKET-HOURS run (a marketable
  order that actually fills) to calibrate the FillModel coefficients
  (`half_spread_bps`, `impact_coef_bps`) against real fills via the parity report.

## Real backtest (live data, split+dividend adjusted)
Donchian on 207 daily SPY bars (`adjustment="all"`):
- frictionless: +4.87% (8 trades, win 37.5%, maxDD -5.9%)
- realistic (spread + √-impact + 10% volume cap + SEC/TAF fees): **+3.12%**
- `dividend_income` = $0 (correct: adjusted bars already embed dividends — the
  engine does not double-credit).
