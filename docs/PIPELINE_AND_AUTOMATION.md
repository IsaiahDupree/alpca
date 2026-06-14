# Alpca — Data Pipeline, Edges & Automation

Operational awareness doc: what runs automatically, what data feeds it, where each edge stands,
and the concrete path to a second deployable edge. Companion to `EDGE_CASE_STUDIES.md` (the full
13→16 case studies), `COMBINATIONS_AND_LOOPS.md` (stacking math + honest ROI), and
`RESEARCH_CANDIDATES.md` (the to-test register).

---

## Edge status (one screen)

| Edge | Status | Headline |
|------|--------|----------|
| **Cointegration-pairs market-neutral basket** | ✅ **Validated** | OOS Sharpe ~0.5, −3% DD — the one proven edge |
| **PEAD** (post-earnings drift, dollar-neutral) | 🟢 **Strong candidate** | 5yr walk-forward Sharpe 0.6–0.8, IS+OOS both +, **DSR 0.92** (just under 0.95) |
| rsi-mr / supertrend / ema-momentum | ⚠️ Risk-reduced **beta** | Deployed live (swing job) as honest lower-drawdown long |
| Seasonality (turn-of-month, pre-FOMC) | ⚙️ Uncorrelated **overlay** | Weak alone; ρ≈0 leg for the combiner |
| Portfolio combiner | ⚙️ **Ready** | Works mechanically; edge-supply-limited (needs a 2nd validated leg) |
| Everything else | ❌ Rejected | single-asset beta, naive/ADF/Kalman pairs, crypto (daily+hourly), A-S MM + inventory-skew, PCA stat-arb (overfit), TSMOM (illusory), funding tilt, news, overnight (artifact), VIX (tail trap), cross-crypto (dead) |

**The plan:** push PEAD's DSR over 0.95 (more symbols + shorting realism) → it becomes the
combiner's second uncorrelated leg (event-clock, ρ≈0 to the pairs basket) → first time we can
genuinely *stack* two validated edges and lift the combined Sharpe.

---

## Data pipeline

| Dataset | Location | Source | Notes |
|---------|----------|--------|-------|
| Daily equity bars (195 sym, 5yr) | `…/AlpcaData/cache/*_1day_bars.jsonl` | Alpaca | the price universe |
| 1-min bars (SPY/QQQ/AAPL, 3yr) | `…/AlpcaData/cache/*_1min_*` | Alpaca | intraday/microstructure |
| Crypto bars (daily + hourly) | `…/AlpcaData/crypto`, `…/crypto_hourly` | Alpaca | rejected for edge, kept |
| **Earnings surprise (deep, 20+yr)** | `…/AlpcaData/earnings_av/*_earnings.json` | **Alpha Vantage** | PEAD; expanding daily |
| Earnings surprise (~1yr) | `…/AlpcaData/earnings/` | Nasdaq (free, no key) | first-pass fallback |
| Perp funding (BTC/ETH, ~1yr) | (fetched live) | Kraken Futures | funding tilt (weak) |

**Earnings source preference** (`alpca/data/earnings.py`): Alpha Vantage (20+yr, needs
`ALPHAVANTAGE_API_KEY` in gitignored `.env`) → Finnhub (needs key) → Nasdaq (~1yr, no key).
Alpha Vantage free tier = **25 requests/day**, soft-throttles ~1 req/sec → use `delay ≥ 2.5s`.
`download_alphavantage_earnings` caches per-symbol, skips cached, and stops cleanly on the daily
cap (`RateLimited`) so batches resume the next day. **Keys live only in `.env` (never committed).**

---

## Scheduled automation (launchd — 5 jobs)

All wrappers live in `~/Library/Application Support/Alpca/` (so launchd can open them without
TCC issues) and reach `~/Documents` via Full Disk Access on `/bin/zsh`. Each uses a **unique
FDA-probe filename + a 5× retry** (fixes the 2026-06-09 concurrency race; see
`scripts/test_launcher_probe.sh`). Logs: `~/Library/Application Support/Alpca/<job>.log`.

| Job | Schedule (ET) | What it does | Orders? |
|-----|---------------|--------------|---------|
| `com.alpca.calibration` | Mon–Fri 09:36 | Fit the fill model to real paper fills | tiny round-trips |
| `com.alpca.livesession` | Mon–Fri 09:50 | Live PAPER donchian-adx on SPY, flatten at 15:50 | yes (PAPER) |
| `com.alpca.swing` | Mon–Fri 09:50 | Risk-reduced basket on QQQ, holds overnight | yes (PAPER) |
| `com.alpca.discovery` | Fri 17:30 | Re-run cointegration-pairs discovery | no |
| **`com.alpca.forwardtrack`** | **Mon–Fri 17:15** | **Shadow forward paper-track — TWO sleeves now: (1) pairs basket, (2) mid-cap vol-managed momentum (borrow-free long/index-hedge, Case 45 ~0.23, uncorrelated ρ=−0.03). Refresh bars → mark prior book → log sized target → accumulate live OOS curve** | **no (shadow)** |
| **`com.alpca.avearnings`** | **Daily 18:15** | **Pull next 23 symbols' AV earnings until universe full (~8 days), then no-op** | **no (data)** |

**Control:** `launchctl list \| grep alpca` (status) · `launchctl bootout gui/$(id -u)/com.alpca.<job>`
(disable) · `launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.alpca.<job>.plist` (enable).

### The AV earnings expander (`scripts/expand_av_earnings.py`)

Fires daily; each run caches the next 23 uncached symbols' 20-yr earnings surprise, respecting
the 25/day quota, and logs progress (`24/195 cached, 171 to go`). When all 195 are cached it
logs `COMPLETE` and no-ops. Manual run: `python scripts/expand_av_earnings.py --batch 23`.

---

## PEAD — how it works and the path to validation

`alpca/backtest/pead.py`: for each earnings event with |surprise| > threshold, long (beat) /
short (miss) the stock for `hold` trading days starting the day after the report; the **long,
short, and dollar-neutral legs are judged separately**. Events outside a symbol's price window
are skipped (a correctness fix — else pre-history events pile in at bar 0). Judged by the harness
incl. **Deflated Sharpe Ratio** (`evaluation.deflated_sharpe_ratio`) to account for trial count.

**Current (23 symbols / 5yr window):** dollar-neutral Sharpe 0.60–0.78, IS & OOS both positive
(0.59/0.63, 0.81/0.73), −15% maxDD, PSR 0.96, **DSR 0.92**. The edge is the cross-sectional
*spread* (high-surprise outperforms low-surprise); the short leg alone is negative (shorting in a
bull loses), so the alpha is long-minus-short.

**Shorting realism (modeled):** `backtest_pead(borrow_apr=…, no_borrow={…})` charges a daily
borrow fee on the short notional and drops un-locatable names. Stress: dollar-neutral Sharpe
0.61 → 0.58 at realistic large-cap GC borrow (~1%/yr) → 0.34 only under a 10% HTB stress. The
edge **survives realistic borrow** (our universe is liquid large-cap GC); shorting cost is *not*
the binding constraint.

**To validate (DSR > 0.95):**
1. **Breadth (the binding constraint)** — the daily `avearnings` job fills the full 195-symbol
   universe (~8 days), tightening the cross-sectional deciles and shrinking the Sharpe SE.
2. **Re-run** `scripts/test_pead.py` as breadth grows; watch DSR cross 0.95.
3. **Stack** — once validated, add PEAD as the combiner's second leg (`scripts/test_combine.py`);
   being event-clock, it's ρ≈0 to the pairs basket, so the combined Sharpe should genuinely lift.

---

## The honest scoreboard (not daily ROI)

Targets are **combined OOS Sharpe (DSR-deflated)** and **max drawdown** — *not* percent-per-day.
At an honest combined Sharpe ~0.9 / 8% vol, the daily expected return is ~3 bps under ~50 bps of
daily noise; the edge is invisible day-to-day and only exists over hundreds of days. A book at
Sharpe ~1.0 with <8% drawdown is a genuinely strong paper result. See `COMBINATIONS_AND_LOOPS.md`.
