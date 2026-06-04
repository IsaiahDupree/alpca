# Market-hours calibration — runbook

Fit Alpca's offline fill model to **real Alpaca paper fills** so backtests use the
slippage/impact/latency the account actually experiences. Everything is built and
verified offline; the only thing pending is the market being open.

## Status (built & verified)
- ✅ Readiness preflight passes against real paper (`calibration_ready.py` → READY).
- ✅ Account ACTIVE, paper, ~$100k equity. Next open: **Monday 2026-06-01 09:30 ET**
  (Memorial Day was 2026-05-25).
- ✅ Fitter recovers known params from synthetic fills (10 unit tests) and from a
  seeded store end-to-end (half-spread 2.508 vs true 2.5, impact 18.008 vs 18.0).
- ✅ Pipeline ran a real parity backtest on 121 live SPY bars (priors vs calibrated).

## The pieces
| component | what it does |
|---|---|
| `alpca/calibration/records.py` | `CalibrationRecord` (intended/fill/latency/volume) + append-only JSONL `CalibrationStore` |
| `alpca/calibration/fit.py` | `calibrate(records)` → `CalibrationResult`: half-spread (median slippage at smallest size), sqrt-impact coef (OLS on √participation), latency preset. `to_fill_model()` / `.save()` |
| `scripts/calibration_ready.py` | preflight — imports, creds, account, clock, quote, fitter, write paths. Safe anytime, **no orders** |
| `scripts/calibrate_paper.py` | LIVE collector — tiny round-trip BUY/SELL orders across sizes during RTH; records real fills; flattens on exit |
| `scripts/run_calibration_pipeline.py` | one command: collect → fit → write `data/calibration.json` → parity vs priors |

## Run it (Monday at the open)

```bash
cd Alpca
# 1) preflight (anytime; should say READY, market OPEN)
.venv/bin/python scripts/calibration_ready.py --symbol SPY

# 2) full pipeline — collects real fills, fits, writes calibration.json, parity
.venv/bin/python scripts/run_calibration_pipeline.py --symbol SPY --cycles 16 --sizes 1,2,3
```

Credentials load from `Alpca/.env` automatically (paper key only). `--env-file`
is optional and now falls back gracefully if the path is missing.

### Safety
- PAPER only (config refuses live). Tiny qty; per-cycle notional cap (`--max-notional`).
- Each cycle buys then sells the same shares back; on exit it cancels working
  orders and **flattens the symbol**. If a flatten ever fails it prints a loud
  `CHECK THE PAPER ACCOUNT`.

## Use the result in backtests

```python
import json
from alpca.execution.fills import FillModel
c = json.load(open("data/calibration.json"))
fm = FillModel(half_spread_bps=c["half_spread_bps"],
               impact_coef_bps=c["impact_coef_bps"],
               participation_cap=0.10, min_tick=0.01)
# run_backtest(strategy, bars, fill_model=fm)  — or backtest_resting(..., fill_model=fm)
```

The latency block in `calibration.json` feeds `SimAdapter` (submit/ack/fill ms)
so offline latency matches the measured paper round trip (~248 ms submit→ack,
recorded in `docs/BASELINE.md`).

## Schedule it (optional)
`run_calibration_pipeline.py --print-cron` prints a cron/launchd line. Or use the
`/schedule` skill / Alpca's scheduler to fire it once at the open. Machine clock
should be ET (or adjust the cron hour).

## `--fit-only`
Re-fit whatever fills are already stored (no new orders) — useful to re-run the
fit after collecting more data across several days:
```bash
.venv/bin/python scripts/run_calibration_pipeline.py --fit-only
```

---

## Phase 4 — microstructure deadband fit (offline, data-gated)

`scripts/analyze_microstructure.py` fits the microprice gate `k` and OFI
`entry`/`exit` deadbands from the **raw tick-level** `data/cache/<sym>_quotes.jsonl`
stream (NOT the 1-min qbars — `attach_quotes_to_bars` broadcasts each sparse quote
forward, so 1,200 "quoted" bars collapse to ~12 unique books). Output:
`data/microstructure_deadbands.json` + a per-symbol table.

```bash
.venv/bin/python scripts/analyze_microstructure.py --symbols SPY,QQQ,AAPL
```

First real fit (cached stream, 2026-06-01 pull):

| sym | clean ticks | uniq books | span | spread p50/p90 (bps) | k (p75 \|tilt\|) | OFI in/out |
|-----|------------:|-----------:|-----:|---------------------:|-----------------:|-----------:|
| SPY | 59,995 | 19,009 | 1.6h | 2.80 / 12.67 | 0.385 | 0.100 / 0.041 |
| QQQ | 59,882 | 18,516 | 0.9h | 6.84 / 16.29 | 0.333 | 0.141 / 0.050 |
| AAPL| 53,659 | 11,673 | 0.2h | 2.58 / 4.84 | 0.429 | 0.117 / 0.043 |

**Findings vs shipped defaults:**
- OFI `entry` fits to **0.10–0.14** vs shipped **0.15** → default is too high
  (confirms the order_flow.py CALIBRATION NOTE).
- `MicropriceGate` `k` fits to **0.33–0.43** (75th pct of |tilt|) vs shipped **0.1**
  → far too low; the gate currently confirms on near-flat books.

**Caveats (the data gate, still open):**
1. Coverage is a single contiguous **<1 session** window (60k-tick cap; AAPL only
   0.2h). Fits are intraday-regime-specific, **not** multi-day-robust. Re-run on a
   multi-session quote pull before treating any value as a shippable constant.
2. The `k` (per-snapshot tilt) transfers to any feed. The OFI numbers are a
   **tick-level** reference — the deployed `L1OFI` consumes 1-min bars (20-bar
   window = 20 min, not 20 ticks = seconds), so **bar-level OFI calibration stays
   data-gated** until per-bar-close NBBO history is cached.
3. Values are written to JSON, not hardcoded into the strategy classes — load them
   per-symbol at construction; do not ship as universal constants.
