# Alpca — Research Candidate Edges (second deep scout pass)

A ranked register of **untested** candidate edges surfaced by a deep survey of open-source
quant repos / papers, each screened against our venue (Alpaca PAPER, ~1.2s taker fills,
top-of-book, ~2bps equity / ~10bps crypto, daily/low-turnover) **and** our harness
(beat buy-and-hold, Sharpe t-stat significance, regime stability, select-IS/report-OOS).

This is a *to-test* list, not a results document — see `EDGE_CASE_STUDIES.md` for what has
already been built and rejected. Hard venue fact that kills several otherwise-attractive
ideas: **Alpaca spot crypto is BTC/USD + ETH/USD only, long-only, no shorting** — so any
cross-sectional crypto long-short basket is infeasible here regardless of published Sharpe.

## Register

| # | Candidate | Data (free?) | Feasibility | Honest expectation |
|---|-----------|--------------|-------------|--------------------|
| 1 | Overnight-vs-intraday premium (buy-close/sell-open) | Alpaca bars ✓ | **MARGINAL→INFEASIBLE** | Largely a bid-ask-bounce / open-print **artifact**; dies to taker cost twice daily |
| 2 | Pre-FOMC announcement drift | SPY + FOMC calendar ✓ | **FEASIBLE** (~8 trades/yr) | Real but **decaying post-2011**; risk-reduced tilt, not standalone alpha |
| 3 | Turn-of-month (TOM) effect | SPY bars ✓ | **FEASIBLE** (~12 rt/yr) | Old, arbitraged; diversifying overlay, likely < B&H absolute return |
| 4 | ~~**PEAD long-short** (earnings surprise)~~ | AlphaVantage ✓ | ❌ **RESOLVED — REJECTED** | Tested 4× (Cases 14, 18, 55, **59**). Real **SUE** *is* the better signal (flips dollar-neutral OOS −0.17→+0.28) but the edge is long-beta-carried, short-leg dies to adverse borrow, and **fails the fresh-symbol holdout** (train OOS +0.03 → fresh-19 OOS −0.71). Not the 3rd leg. |
| 5 | Betting-Against-Beta / low-vol | Alpaca bars ✓ | **MARGINAL** | Un-levered = risk-reduced beta (like rsi-mr), not a beater; needs leverage we lack |
| 6 | Vol-target / risk-parity overlay | Alpaca bars ✓ | **FEASIBLE** as overlay | Drawdown insurance on the beta sleeve; **already half-rejected as alpha** (Kim 2016) |
| 7 | VIX term-structure regime timing (SVXY) | CBOE/Yahoo ^VIX/^VIX3M ✓ | **MARGINAL** | Short-vol tail trap — Sharpe-test actively **misleads**; only with a CVaR/tail gate |
| 8 | ~~**Meta-labeling** (triple-barrier) on our basket~~ | none (runs on signals) ✓ | ❌ **RESOLVED — REJECTED** | Case 58: meta-model OOS AUC **0.379** (anti-predictive, below its shuffle placebo 0.46) = zero skill; the τ=0.60 "lift" was a trade-count-cut mirage. Pairs stays deployed as-is. |
| 9 | ETF relative-value via Box-Tiao / min-half-life | Alpaca sector ETFs ✓ | **MARGINAL** | Same family we rejected; optimizing half-life is a **stronger overfit magnet** |
| 10 | Cross-asset / lead-lag (BTC→ETH, crypto X-section) | Alpaca BTC/ETH ✓ | **INFEASIBLE** | Needs a coin cross-section + shorting; 2 long-only coins too narrow — **dead** |
| 11 | Cross-sectional crypto factor/reversal basket | needs coin X-section ✗ | **INFEASIBLE** | Long-only 2-coin venue cannot implement it — **dead** |

## Top 3 to build next (scout's ranking, by expected value)

1. ~~**PEAD cross-sectional long-short**~~ — ❌ **RESOLVED (Case 59): REJECTED.** Built it at
   63-symbol / 30-yr breadth with **real SUE** (the one untried lever). SUE beats raw surprise,
   but the dollar-neutral edge is thin, long-beta-carried, short-leg dies to adverse borrow, and
   **fails the fresh-symbol holdout** (fresh-19 OOS −0.71). 4th PEAD rejection. ~~Meta-labeling
   (#8)~~ also **RESOLVED (Case 58): REJECTED** (no OOS skill). Both lowest-hanging candidates
   are now closed.

2. **Seasonality overlay basket** = pre-FOMC drift + turn-of-month (+ optional overnight).
   ⚠️ **The decisive test is DATA-GATED (2026-06).** The "disappearing post-2011" claim needs
   deep daily history, but **every daily cache we have is the same 5yr SIP window (2021-06→2026-06)**
   and **AlphaVantage free tier no longer serves `outputsize=full`** (now a premium feature; free =
   last 100 bars only). On the available 2021-2026 window the overlays are **weak and OOS-negative**
   (SPY turn-of-month IS 0.73 / **OOS −0.74**, exposure 34%; pre-FOMC IS 0.61 / **OOS −0.99**;
   QQQ similar; all t-stats insignificant, p 0.2–0.66) — consistent with the decay thesis, but one
   regime can't adjudicate it. **Status: marginal overlay, not a third leg.** To actually test the
   decay split, source deep daily history (Stooq/Tiingo free, or a paid AV/Polygon tier).

3. **Meta-labeling (mlfinlab triple-barrier) on the surviving 84-symbol basket.**
   *Why it might survive:* doesn't need a new edge — it **amplifies** the one we've proven
   OOS (0.54) by filtering low-conviction trades, with purged-CV machinery built to resist
   the overfit that killed Avellaneda-Lee. Lowest-risk build; upside is lifting our sole
   deployable market-neutral strategy. Honest null: fewer trades + estimation noise may wash
   out the precision gain on an already-thin edge.

## Explicitly dead ends (do not spend time)

- **Cross-crypto / lead-lag / crypto-factor baskets** (#10, #11) — need a coin cross-section
  and shorting we don't have on Alpaca spot.
- **Naive-to-fancy ETF cointegration reselection** (#9) — same family already rejected;
  half-life optimization is a stronger overfit magnet.
- **Standalone overnight buy-close/sell-open** (#1) — largely a bid-ask / open-auction-print
  artifact that evaporates at a taker fill with cost. (Correction to an earlier suggestion:
  this is **not** a high-promise pick.)
- **Short-vol VIX harvesting** (#7) — executable via SVXY but its fat negative-skew tail is
  exactly what a Sharpe-significance test over-rewards; only with an explicit tail/CVaR gate.

## Notes on data access (verified)

- **Binance `fapi`** funding/derivatives: **geo-blocked (HTTP 451) from US IPs.** Use Kraken
  Futures (works) for funding.
- **Finnhub free tier** `/calendar/earnings`: EPS estimate + actual + surprise %, US-accessible
  — the free substitute for WRDS/IBES that makes PEAD buildable.
- **CBOE / Yahoo `^VIX`, `^VIX3M`**: free term-structure proxy for VIX-curve signals.
- **AlphaVantage `TIME_SERIES_DAILY` (2026-06):** `outputsize=full` is now a **PREMIUM** feature —
  the free tier returns only the last ~100 bars (compact). This blocks any deep-history (pre-2011)
  daily backtest on the free key. All local daily caches are the 5yr SIP window (2021-06→2026-06).
  For deep daily history use **Stooq** or **Tiingo** (free, deep) instead.
- **AlphaVantage earnings cache** now at **63 train + 19 holdout symbols** (~30yr each); the daily
  launchd job (18:15) trickles ~23/day under the ~25/day free cap. Full 195-universe still filling.
