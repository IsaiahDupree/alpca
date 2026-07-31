# Alpca — Research Candidate Edges (second deep scout pass)

*Superseded 2026-07-30 — kept for history.* This was a hand-curated register from a single deep survey.
It has since been **overtaken by two at-scale harvests**: Case 62 (176 published strategies, 16-agent
fan-out) and Case 64 (217 strategies from the user's private trading-knowledge corpus, cross-checked
against five other codebases) — both converge on **zero new Alpaca-deployable alpha** (see
`docs/STATE_OF_THE_PROGRAM.md` → "Idea supply is exhausted, not the venue's patience"). The program is no
longer candidate-limited; the three open threads are all **forward-tracking probation sleeves** waiting on
live clocks, not new backtests: composite (Case 68), analyst-revision drift (Case 66), hardened
cross-sectional momentum (Case 65) — see `docs/PAPER_EDGE.md` for their live status and
`docs/EDGE_CASE_STUDIES.md` Cases 60-68 for the full write-ups. The register below is left as-is for
reference.

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
   ✅ **DATA GATE RESOLVED (Case 69, 2026-07-30).** `scripts/download_yahoo_deep.py` pulls Yahoo
   Finance's public chart API (no key needed) for **SPY 33.5yr (1993-01-29→2026-07-29)** and
   **QQQ 27.4yr (1999-03-10→2026-07-29)** — verified clean against the existing SIP cache — plus the
   FOMC calendar extended back to 1994 (200 dates, sourced from the Fed's own historical-materials
   pages). The pre/post-2011 decay split turned out to be **leg-specific, not a blanket decay**:
   pre-FOMC drift decayed hard (SPY Sharpe 0.71→0.11, t 2.99→0.44, p 0.003→0.66) but turn-of-month
   did **not** decay — if anything it got stronger post-2011 (SPY 0.48→0.59, QQQ 0.15→0.63).
   Adversarial review found real caveats on top of that split: the pre-FOMC "decay" is partly a
   **2007–2009 crisis-regime artifact** (that window alone drove 5x the per-event effect of the other
   14 pre-2011 years), and TOM's "post-2011 strength" is **stale** — the 2021–2026 window alone is
   statistically insignificant for both symbols. **Status: overlay is real but noisier and more
   regime-dependent than a clean decay story implies — still a marginal diversification leg, not a
   third fundable edge.** Full writeup: Case 69 in `docs/EDGE_CASE_STUDIES.md`.

3. ~~**Meta-labeling (mlfinlab triple-barrier) on the surviving 84-symbol basket.**~~ — ❌
   **RESOLVED (Case 58): REJECTED.** Same candidate as #8 above. OOS AUC **0.379**, *below* its
   own shuffle-placebo (0.46) — anti-predictive, zero skill. The τ=0.60 "lift" that looked
   promising was a trade-count-cut mirage. Pairs stays deployed as-is, unfiltered.

**All three "top 3 to build next" items are now resolved — none survived.** The next-highest-EV work
after this pass was not a 4th candidate from this register; it was validating the *one deployed edge*
harder (Case 60's out-of-regime SIP unlock, Case 67's real-tail test) and then two at-scale harvests
(Case 62: 176 strategies, Case 64: 217-strategy corpus) that exhausted the idea space more thoroughly
than any hand-picked shortlist could. See `docs/STATE_OF_THE_PROGRAM.md` for what's currently live.

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
