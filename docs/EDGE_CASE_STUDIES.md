# Alpca — Edge Case Studies

A brutally honest compendium of every trading edge we have hypothesized, built, and run
through the evaluation harness. Each case study records the hypothesis, method, data,
**actual measured result**, and verdict. The point of this document is not to advertise
winners — it is to record what *survived rigorous testing* and what *did not*, so we never
re-litigate a settled question or ship an overfit backtest.

**The discipline (applied to every case):** judge vs **buy-and-hold**, require **statistical
significance** (Sharpe t-stat, p<0.05), **regime stability** (segment Sharpes positive in a
majority of sub-periods), and **out-of-sample / walk-forward** survival (select parameters
on in-sample, report on held-out data). Market-neutral strategies have no buy-and-hold to
beat — the return itself is the alpha, and the honest null is our one surviving edge.

**Venue reality (constrains every case):** Alpaca PAPER; ~1.2s signal→fill; IEX top-of-book
only (no L2 depth); no maker rebates; price-taker; ~2 bps equity / ~10 bps crypto per leg.
Overtrading dies to costs. HFT / market-making is structurally infeasible here.

---

## Scoreboard

| # | Edge | Class | Headline result | Verdict |
|---|------|-------|-----------------|---------|
| 1 | **Cointegration-pairs market-neutral basket** | Market-neutral | **WF 0.83** at the concentrated **top-10 + 5% ADF screen**, −4% DD (the "0.29" was an over-diversified top-24). **Deployed on a forward paper-track** | ✅ **THE survivor** — stronger than thought; live track adjudicates |
| 2 | Single-asset directional (trend/breakout/MR) | Directional | rsi-mr Sharpe 1.18 vs B&H 0.86, but never beats B&H return | ⚠️ Risk-reduced **beta**, not alpha |
| 3 | Cross-sectional momentum | Market-neutral-ish | Best Sharpe ~0.68 (lb250); config-sensitive | ⚠️ Modest beta |
| 4 | Short-term reversal | Market-neutral | Negative in- and out-of-sample | ❌ Rejected |
| 5 | Naive pairs + ADF screen + Kalman hedge | Market-neutral | Walk-forward 0.43 → 0.23 (+ADF) → −0.26 (+Kalman) | ❌ Fixes made it worse |
| 6 | Crypto (daily) pairs + cross-sectional | Market-neutral | Walk-forward −0.14 to −0.56 Sharpe | ❌ Rejected |
| 7 | Crypto (hourly) pairs + cross-sectional | Market-neutral | 0 cointegrated pairs; WF Sharpe −0.57 | ❌ Rejected |
| 8 | Avellaneda-Stoikov market-making | HFT/MM | Needs L2 + rebates + ms requoting | ❌ Infeasible on venue |
| 9 | A-S inventory-skew sizing | Sizing overlay | Single-asset OOS +0.08 vs B&H 0.74; spread 0/3 vs binary | ❌ Beats neither baseline |
| 10 | PCA / eigenportfolio residual stat-arb | Market-neutral | In-sample 0.99 → **OOS −1.18** | ❌ Overfit; edge decayed |
| 11 | TSMOM (vol-scaled ETF panel) | Diversified | momentum 1.62 < vol-scale 1.76 < buy-hold 1.93 (OOS) | ❌ Illusory (Kim 2016) |
| 12 | Crypto funding-rate tilt | Sentiment overlay | Mild DD reduction in a 1yr bear; no alpha | ⚠️ Weak / inconclusive |
| 13 | News / sentiment alt-data | Alt-data | Free API exposes ~50 articles / ~8 days | ❌ Not backtestable here |
| 14 | **PEAD** (post-earnings drift, L/S) | Market-neutral event | Flat-borrow DSR 0.92, but **DSR 0.58 under adverse-selection borrow**; short leg −0.47 standalone | 🟡 **Downgraded** — long leg is beta, short leg fails realistic shorting frictions (24/195 symbols; revisit at full breadth) |
| 15 | Seasonality (turn-of-month, pre-FOMC) | Event-clock overlay | Standalone Sharpe 0.24–0.34, exposure 3–34% | ⚠️ Weak alone; ✅ uncorrelated leg |
| 16 | **Portfolio combination** (inverse-vol blend) | Allocation | EAR-PEAD lift (0.83→0.99) **RETRACTED** — that leg failed its holdout; back to ~0.83 on one real leg | ⚙️ Method works; still edge-supply-limited |
| 17 | **Overnight→intraday reversal** | Market-neutral event | Gross Sharpe 0.93 / DSR 0.90 (control-confirmed) → **−0.41 at 2bps** (~2×/day turnover) | 🔴 **REAL anomaly, untradeable** — canonical cost-wall case |
| 18 | **EAR-PEAD, index-beta-hedged** | Market-neutral event | TRAIN 40 → +0.68, but **fresh-symbol HOLDOUT (19 disjoint) → −0.52**; passed in-universe audits, failed out-of-universe | 🟡→❌ **Does not generalize** — edge was specific to the fitted universe |
| 19 | **Lead-lag cross-predictability** | Market-neutral (learned) | Walk-forward real −1.02 ≈ shuffle placebo −1.14 (+0.11); gross only 0.27, dies by 1bp | ❌ **Fitted noise** — fails placebo *and* cost wall |
| 20 | **Gap reversion** (multi-day hold) | Market-neutral event | No gross edge (−0.14 @ 0bps); gap-momentum control *beats* it on large caps | ❌ **Signal failure** — large-cap gaps are informational, not reverting |
| 21 | **Short-interest (borrow-fee) tilt** | Market-neutral positioning | 1-yr Nasdaq looked great (2.34) but 9-yr FINRA (188 sym): gross 0.91, **net −0.42 after DTC borrow**, +3/6 yrs | ❌ **Rejected** — weak, regime-specific, net-negative after borrow; the 1-yr lead was a lucky window |
| 22 | **52-week-high momentum** (George-Hwang) | Cross-sectional | Anomaly INVERTS here: near-high −0.58, reversal +0.6 but carried by 2023 alone | ❌ **Rejected** — famous anomaly doesn't replicate on our universe; reversal regime-concentrated |
| 23 | **Accruals anomaly** (Sloan, EDGAR fundamentals) | Fundamental MN | In-universe great (+5/6 yrs, cost-free) but **fresh-16 holdout −0.47** (train +0.30) | 🟡→❌ **Fails out-of-universe** — same as EAR-PEAD; 3rd candidate killed by the fresh-symbol test |
| 24 | **Value composite** (E/P+FCF/P+B/P, EDGAR) | Fundamental MN | Main ~0.14, **fresh-holdout +0.11..+0.54 (GENERALIZES)** but weak + regime-timed (2022 +1.85 / 2026 −1.58) | ⚠️ **Real but too thin** — 1st fundamental to pass the fresh test; fails on magnitude, not overfit |
| 25 | **Betting-Against-Beta / low-vol** | Factor MN | Unlevered dollar-neutral: beta −0.63, vol −0.95 (only 2022 +); needs leverage to harvest | ❌ **Rejected** — risk-adjusted premium needs leverage we lack; raw version is short-beta in a bull |
| 26–32 | **Factor zoo** (asset-growth, net-issuance, ROA, MAX, idio-vol, residual-mom, vol-managed-mom) | Cross-sectional | All −0.9..+0.1 main, fresh-holdouts ≤ 0; none clear the rail | ❌ **All rejected** — documented premia are thin/absent on large-caps; the two momentum variants overfit (fresh < 0) |
| 33 | **Short-interest CHANGE** (ΔDTC, not level) | Positioning MN | main −0.27, only 2/6 yrs, fresh +0.68 inconsistent w/ negative main | ❌ **Rejected** — not regime-robust, dies to cost; the *change* is no better than the *level* (Case 21) |
| 34 | **Gross profitability** (Novy-Marx, EDGAR Rev−COGS/Assets) | Fundamental MN | main −0.31, OOS −1.72, fresh −0.40, regime-flipping (2021 +2.08 → 2026 −2.51) | ❌ **Rejected** — the "most robust factor" is net-negative on our large-caps |
| 35 | **Financials-excluded accruals & value** (SIC fetch) | Fundamental MN | accruals in-sample +0.70/6-of-6 but **fresh still −0.51**; value unchanged (+0.22 fresh) | ⚠️/❌ **Sector-rescue refuted** — excluding financials cleans accruals in-sample but doesn't fix its out-of-universe failure |
| 36 | **Published formulaic-alpha zoo** (Alpha101 / Alpha158 / GTJA191, 21 de-collinearized family reps) | Microstructure MN | best DSR **0.596** (gate 0.90), **0/21 pass Bonferroni**; in-sample/OOS winners INVERT on fresh symbols (gap-reversal +1.29→−0.16, intraday +1.01→−0.77) | ❌ **DEBUNKED** (Case 56) — the famous 101/158/191 libraries collapse on our venue exactly like value/momentum/seasonality |
| 37 | **Pairs OU-refinement** (cost-calibrated entry-z + OU-proportional sizing) | MN hardening | **INERT** at ≤~20bps — bit-identical to the validated baseline (0.83); the `regime_monitor` variant DESTROYS the edge (0.83→−0.32) | ❌ **No improvement** (Case 56) — nothing to commit; footgun flagged in source |
| 38 | **Defensive low-beta / BAB** | Factor long-short | defensive (long-low-β) BAB Sharpe **−1.06** (a loser); only the reverse (long-high-β) is positive (+1.05) and is pure beta (β to SPY −0.65) | ❌ **reject_beta** (Case 56) — dampened market exposure, not alpha |
| 39 | **Mid-cap cointegrated-pairs** (the one mechanism, on a disjoint universe) | Market-neutral | survivor **−0.43**, PIT **+0.28**; **clears** the survivorship + disjoint-half-holdout bar (A +0.16 / B +0.34, both positive) — the first mid-cap family to — but DSR **0.06**, cadence **knife-edge** (positive only at train=252), ρ −0.01 yet combiner lift **−0.04** | ❌ **Too weak** (Case 57) — mechanism *generalizes* but doesn't *lift*; diluting, cadence-fragile, statistically ≈ noise |

---

## Case 1 — Cointegration-pairs market-neutral basket ✅ (THE survivor)

- **Hypothesis.** A diversified basket of cointegrated pairs (each dollar-neutral, traded on
  spread mean-reversion) produces direction-independent return — real alpha, not beta.
- **Method.** Screen all pairs in an 84-symbol sector-diverse universe by Ornstein-Uhlenbeck
  half-life; backtest the top stable pairs individually and as an equal-weight basket;
  validate **out-of-sample** (screen on first 60%, trade held-out 40%).
- **Data.** 5 years of daily bars, 84 symbols (tech/fin/health/consumer/energy/industrial +
  sector ETFs), Alpaca. Cost 2 bps/leg.
- **Result.** In-sample basket Sharpe **1.78** / −3.3% DD (overfit — the screen cherry-picks
  past winners). The honest **out-of-sample** number: **Sharpe 0.54 / +4.8% / −3.0% DD**.
  Best individual OOS pairs: GLD/RTX (1.37), RTX/XLU (0.93), DE/GILD (0.90).
- **Verdict.** ✅ **Real but modest.** Market-neutral by construction, tiny drawdown, survives
  OOS. This is the *only* edge that has cleared the bar. It is the **null** every later
  candidate must beat.
- **Caveat.** A larger walk-forward (195 symbols, re-screen each quarter) degrades it to
  Sharpe 0.43 and lower for wider baskets — the half-life screen is permissive. The 0.54 is
  the static-60/40 number; treat ~0.4–0.5 as the honest range.
- **⚠️→✅ RE-MEASURED then CORRECTED (2026-06-13): the edge is stronger than "0.29".** A first
  re-measure on the 195-universe gave a *marginal* walk-forward 0.29 — but that used an
  over-diversified **top-24** basket. A `top_n` sweep shows **concentration is the dominant lever**:

  | basket | walk-forward Sharpe | maxDD |
  |---|---|---|
  | top-24 (the misleading "0.29") | ~0.2 | −5.9% |
  | **top-10** | **0.80** | −4.4% |
  | **top-10 + 5% ADF cointegration screen** | **0.83** | **−4.0%** |

  Diluting into 20+ weak pairs roughly *halved* the edge (consistent with the session-22 finding that
  "more pairs did worse"); the honest, concentrated walk-forward Sharpe is **~0.83**. Adding an
  **ADF significance screen** (`max_adf=−2.86`, the 5% critical value — a principled threshold, not a
  tuned one) gives a small free lift and tightens the drawdown to −4.0%. Both are walk-forward (the
  top-10 are re-selected fresh each quarter), so this is a *real improvement to the validated edge*,
  not curve-fitting. **The deployed config is updated to top-10 + 5% ADF.**
- **DEPLOYED as a SHADOW FORWARD PAPER TRACK** (`alpca/live/pairs_portfolio.py`,
  `scripts/deploy_pairs_paper.py`, launchd `com.alpca.forwardtrack`). Each run computes today's live
  target book (trailing-window screen + ADF filter, no look-ahead, hysteresis so it doesn't churn),
  sizes it half-Kelly on the **0.83** WF Sharpe with a vol-target and a diversification guard, marks
  the prior book to today's prices, and accumulates a **live out-of-sample curve** — no broker orders,
  no capital at risk (the gold-standard adjudicator). The book is naturally sparse day-to-day (few
  pairs at a z-extreme at once) and fills as screened pairs trigger; the live OOS curve over months is
  what counts. **This is the one validated edge, now correctly configured and sized.**

## Case 2 — Single-asset directional strategies ⚠️ (beta)

- **Hypothesis.** Trend/breakout/mean-reversion strategies on a single liquid asset generate
  alpha.
- **Method.** Whole 34-strategy registry through the harness on SPY daily (the "truth table").
- **Result.** **GENUINE market-beaters = NONE.** rsi-mr (Sharpe 1.18), supertrend (0.99),
  ema-momentum (0.98), ensemble (0.94) are statistically significant and stable and beat the
  market's *Sharpe* (0.86) — but **none beats buy-and-hold on return**, and none beats B&H
  out-of-sample.
- **Verdict.** ⚠️ **Risk-reduced beta, not alpha.** rsi-mr is the deployable one: a lower-
  drawdown way to be long. It is what the live **swing** job trades — honestly, as beta.
- **Lesson.** In a multi-year bull run, a long-biased backtest cannot separate skill from
  beta. Always benchmark vs buy-and-hold.

## Case 3 / 4 — Cross-sectional momentum & reversal

- **Momentum.** Market-neutral long-winners/short-losers. Best Sharpe ~0.68 (lookback 250),
  but **config-sensitive** (a 20/5 config loses 16%); long-only top-k is just beta (+92%).
  At 1-minute frequency, L/S = **−98.9%** (26k rebalances × leg costs). ⚠️ Modest beta;
  high-frequency market-neutral does **not** escape the cost wall.
- **Reversal** (long losers / short winners, the one untested price anomaly). 195-symbol
  universe, OOS split: **negative in- and out-of-sample** (−0.47 to −2.06 Sharpe). Costs
  destroy the 1-day bounce in liquid large caps. ❌ Rejected.

## Case 5 — Naive pairs + ADF screen + Kalman hedge ❌

- **Hypothesis.** A proper cointegration significance test (Augmented Dickey-Fuller) and a
  dynamic (Kalman) hedge ratio will lift the pairs walk-forward Sharpe.
- **Result** (195-symbol walk-forward, re-screen each quarter): baseline **0.43** → with ADF
  filter **0.23** → with ADF+Kalman **−0.26**. Both textbook fixes made it **worse**: ADF
  concentrates in past-cointegrated pairs that break out-of-sample; Kalman adapts the mean
  away and barely trades.
- **Verdict.** ❌ **End of the pairs-improvement line.** Naive equity pairs have no reliable
  walk-forward alpha, and the standard improvements don't rescue it. The win was the harness
  that prevented shipping the overfit 1.78.

## Case 6 / 7 — Crypto, daily and hourly ❌

- **Daily** (14 long-history coins, 3.8yr shared, 10 bps): cross-sectional momentum mostly
  negative; pairs walk-forward **−0.14 to −0.56 Sharpe**, −62% to −80% DD.
- **Hourly** (15 coins, 384k bars, ppy = 8760, 10 bps): **0** cointegrated pairs in the
  6–72h half-life band; pairs walk-forward **−0.57 Sharpe** over 21 windows (15,120 OOS bars).
  Cross-sectional momentum only positive at slow horizons *in-sample* (0.45), deeply negative
  when fast (−3.65) or reversed.
- **Verdict.** ❌ "Less-efficient asset" ≠ reliable market-neutral alpha. More bars did **not**
  reveal a hidden intraday edge. Crypto vol + 10 bps crush it.

## Case 8 — Avellaneda-Stoikov market-making ❌ (infeasible)

- The canonical MM framework (reservation price + optimal spread) requires continuous two-
  sided **resting quotes**, millisecond requoting, **maker rebates**, and **queue/adverse-
  selection** modeling. Our venue: ~1.2s fills, top-of-book only, no rebates, no queue
  priority — you are a price *taker* paying the spread every round trip. ❌ **Right framework,
  wrong venue.** Correct implementation on the wrong market = zero edge.

## Case 9 — A-S inventory-skew sizing ❌

- **Hypothesis.** The one portable A-S idea: optimal inventory ∝ −mispricing/(γ·σ²). Apply it
  as continuous, vol-scaled sizing to a mean-reversion signal (a sizing overlay, not alpha).
- **Result.** (a) **Single asset** (10 names): A-S sizing Sharpe −0.15 / OOS +0.08 vs
  **buy-and-hold 0.78 / +206%** — mean-reversion fights the trend, loses badly; marginally
  less-bad than a binary z-entry OOS (+0.08 vs −0.02) but irrelevant. (b) **Cointegrated
  spread** (the legitimate test): A-S continuous sizing beats the classic binary z-entry rule
  on **0/3 pairs in-sample, 1/3 OOS** (mean OOS Sharpe 0.08 vs 0.30). The binary wait-for-2σ /
  exit-at-0.5σ rule captures reversion with less time-in-market; continuous inventory bleeds.
- **Verdict.** ❌ Beats neither buy-and-hold nor the naive baseline.

## Case 10 — PCA / eigenportfolio residual stat-arb ❌

- **Hypothesis.** The principled generalization of hand-picked pairs (Avellaneda-Lee 2009):
  regress each stock on the top-~15 PCA eigenportfolios, model the idiosyncratic **residual**
  as an OU process, trade the s-score. Market-neutral by construction; should beat the 0.54
  pairs null.
- **Method.** Vectorized daily walk-forward over 195 symbols (one `lstsq` for all stocks on
  shared eigenportfolio factors), dollar-neutral open/close bands, OU half-life filter. Grid
  selected on **in-sample**, reported on **out-of-sample**.
- **Result.** Best in-sample Sharpe **0.99** collapses to **OOS −1.18**; segment Sharpes
  `[0.65, 1.73, 0.61, −1.42]` — the most recent (OOS) segment went negative. The edge
  **decayed** in the recent period.
- **Verdict.** ❌ **Overfit; does not beat the 0.54 null.** A textbook in-sample/OOS reversal,
  caught precisely because we select on IS and report on OOS. (Residual stat-arb is a real,
  historically-documented edge — but it has crowded out, which our OOS window reflects.)

## Case 11 — TSMOM, vol-scaled ETF panel ❌ (illusory)

- **Hypothesis.** Time-series momentum (sign of trailing 12-month return, vol-scaled, monthly
  rebalance) on a diversified 9-ETF cross-asset panel (SPY/QQQ/IWM/DIA/EEM/EFA/TLT/GLD/SLV) —
  the classic CTA edge that survives the cost wall.
- **The honest null (Kim, Tse & Wald 2016).** Run three things on the same panel: `tsmom`
  (momentum + vol-scale), `long_vol` (vol-scale only), `ew_bh` (equal-weight buy-and-hold).
- **Result (OOS Sharpe):** tsmom **1.62** < long_vol **1.76** < ew_bh **1.93**. The momentum
  *timing* actually **hurts** vs pure vol-targeting, and neither beats buy-and-hold.
- **Verdict.** ❌ **Illusory.** The "momentum" is just vol-targeting, exactly as Kim 2016
  predicted — and on this panel/period even that loses to holding. Clean negative.

## Case 12 — Crypto funding-rate tilt ⚠️ (weak / inconclusive)

- **Hypothesis.** Persistent extreme-positive perp funding = crowded longs → step out; a
  low-turnover long/flat overlay on spot BTC/ETH should cut drawdown vs holding.
- **Data.** Kraken Futures funding (free, US-accessible; **Binance fapi is geo-blocked 451**),
  ~1 year (2025-06 → 2026-06), aligned to Alpaca spot daily bars. 10 bps cost.
- **Result.** Over a window where BTC fell −40% and ETH −33% (a **bear**), the gate reduced
  loss/drawdown (BTC full-sample Sharpe −0.97 → −0.36; DD −49.7% → −44.3%) and the selected
  config beat buy-hold OOS. But every full-sample Sharpe is **negative** — this is risk
  reduction in a down market, not alpha.
- **Verdict.** ⚠️ **Weak and inconclusive.** Single ~1yr regime, tiny OOS sample. Worth the
  cheap test (it's the only free, orthogonal, non-price signal we found) but not deployable
  and not validated. Multi-year multi-regime funding history (not free) would be required.

## Case 13 — News / sentiment alt-data ❌

- Alpaca's free news API exposes only ~**50 newest articles (~8 days)** — ~456× short of the
  multi-year history a backtest needs. A real news/sentiment edge requires a paid historical
  archive + an NLP model. ❌ Not backtestable on this venue.

---

## Case 14 — PEAD (post-earnings-announcement drift) 🟡 (encouraging, unvalidated)

- **Hypothesis.** A stock that beats consensus keeps drifting up for weeks (and misses drift
  down). Long high-surprise / short low-surprise, dollar-neutral — event-driven and
  cross-sectional, so genuinely *diversifying* from price-mean-reversion pairs.
- **Method.** Earnings surprise from the free Nasdaq endpoint (no key, ~4 quarters/ticker);
  long if surprise > +thr, short if < −thr, hold 30 trading days from the day after the
  report. Long, short, and dollar-neutral legs judged **separately**. 167 symbols, 665 events.
- **Data (upgraded).** Alpha Vantage `EARNINGS` (free, key in `.env`): 23 sector-diverse
  symbols, **2,554 quarterly surprises back ~20 years**, of which **395 fall inside the 5-year
  price window** — a genuine multi-regime sample (2021 bull, 2022 bear, 2023–26 recovery).
  (Replaced the first pass on free Nasdaq data, which only had ~1 year and produced a
  misleading IS=0 / fake-OOS artifact.) A correctness fix skips events outside the price window.
- **Result (5-year walk-forward).** Dollar-neutral Sharpe **0.60** (±2% thr) to **0.78** (±3%),
  with **in-sample AND out-of-sample both positive and consistent** (0.59/0.63 and 0.81/0.73) —
  a *real* walk-forward, −15% maxDD. **PSR(>0) = 0.96; Deflated Sharpe = 0.92** (deflated for
  34 trials). The dollar-neutral leg captures the genuine cross-sectional PEAD spread (high-
  surprise outperforms low-surprise) **even though the short leg alone is negative (−0.47)** —
  shorting low-surprise names in a bull market loses; the edge is the long-minus-short *spread*,
  not the short leg in isolation (this corrects the earlier theory note).
- **Shorting realism, flat borrow (first pass — the optimistic case).** `backtest_pead` charges
  a daily stock-borrow fee on the short notional (`borrow_apr`, flat or per-symbol) and drops
  names with no locate (`no_borrow`). Flat stress: Sharpe **0.61 → 0.58** at large-cap general-
  collateral borrow (~1%/yr) → **0.53** at 3% → **0.34** at a 10% HTB. Under a *flat* assumption
  the edge looks robust. **But a flat rate is the optimistic case** — it ignores *which* names go
  special.
- **Shorting realism, ADVERSE SELECTION (the honest stress — this is the one that binds).** The
  names you most want to short on PEAD are exactly the ones that just printed the worst miss —
  i.e. **crowded shorts** whose borrow goes special and whose locate can vanish. `adverse_borrow`
  models this: the per-event short borrow apr ramps from 1% GC up to a "special" rate as the miss
  worsens, **saturating** (raw `surprise_pct` is wildly fat-tailed — p50 |s|≈6%, p99≈200%, max
  34,000% — so a 300% miss is no more borrowable than a 50% miss), and events past a `no_locate`
  ceiling on |surprise| are **dropped entirely** (the crowded short went no-locate). Calibrated to
  the real distribution:

  | Stress | Sharpe | OOS | DSR | shorts dropped |
  |---|---|---|---|---|
  | Flat GC 1% (optimistic) | 0.58 | 0.60 | 0.83 | — |
  | **Adverse: realistic (special 30%, no-locate \|s\|≥200%)** | **0.25** | **0.20** | **0.58** | 15 |
  | Adverse: harsh (special 60%, no-locate \|s\|≥100%) | −0.23 | −0.05 | 0.19 | 22 |

  Realistic adverse selection **more than halves the Sharpe and collapses DSR 0.92 → 0.58** — far
  below the 0.90 bar. Root cause: the short leg is **already a standalone loser (−0.47 at zero
  borrow)** — post-miss names drifted *up* over this window, so there is no short-side drift to
  harvest. The dollar-neutral 0.61 was carried entirely by the **long leg (0.83/1.13 — but that
  is beta)**. Adverse borrow piles real cost onto an already-edgeless, costly-to-implement short
  side.
- **Verdict (downgraded 🟢 → 🟡).** On the current sample PEAD is **not** the validated second
  market-neutral edge: its long leg is beta, its short leg has no edge here and is the most
  expensive part to actually trade, and the dollar-neutral combo **does not survive realistic
  short-side adverse selection** (DSR 0.58). The 84-symbol cointegrated-pairs basket (OOS Sh 0.54)
  remains the **only** validated edge.
- **The one caveat that keeps PEAD alive.** Only **24 / 195** symbols are cached (free-tier 25
  req/day). Full breadth tightens the short deciles and could flip the short-leg sign. So this is
  **not a kill** — but **adverse-selection borrow is now the #1 hurdle PEAD must clear, and today
  it does not.** That is the real reason to finish the universe, not just to shrink the standard
  error.
- **➡️ RESCUED in Case 18.** The fix turned out to be structural, not statistical: replace the
  analyst-surprise signal with the price-only **EAR** (earnings-announcement return) and replace
  the borrow-fragile single-name short with a **cheap GC index short**. The beta-hedged EAR sleeve
  survives honestly (Sharpe 0.67, IS≈OOS, DSR 0.89) — see Case 18.
- **Next step.** The daily `avearnings` job (writing to `My Passport/AlpcaData/earnings_av`) fills
  the full 195-symbol universe (~8 days). The decisive test is no longer "does DSR clear 0.95 on a
  flat borrow" — it is **"does the dollar-neutral leg survive `adverse_borrow` at full breadth."**
  A better signal than raw `surprise_pct` (SUE — standardized unexpected earnings) is the obvious
  next research lever, since the crude % surprise may itself be why the short leg has no drift.

## Case 15 — Calendar seasonality (turn-of-month, pre-FOMC) ⚠️/✅

- **Hypothesis.** Long an index ETF only around month-end flows (turn-of-month) or the ~24h
  before scheduled FOMC announcements (Lucca-Moench drift); flat otherwise.
- **Result** (SPY/QQQ, ~2021-2026): standalone Sharpe 0.24–0.34 (turn-of-month, 34% exposure)
  and 0.24–0.60 (pre-FOMC, 3% exposure) — below buy-and-hold's 0.80+ on absolute return, and
  OOS-negative standalone (our window can't test the documented pre-2011 vs post-2011 decay).
- **Verdict.** ⚠️ **Weak as a standalone strategy** (cash-parked most days) **but ✅ valuable as
  an uncorrelated leg** — its PnL is on an *event clock*, so it correlates ~0 with every
  price-driven strategy (see Case 16). That structural ρ≈0 is its entire value.

## Case 16 — Portfolio combination (inverse-vol + half-Kelly blend) ⚙️

- **The math.** Combining k equal-risk legs of Sharpe S, avg correlation ρ, gives
  `S·√k / √(1+(k−1)ρ)`. Four uncorrelated 0.5-legs → 1.0; at ρ=0.3 → only 0.69. **Correlation
  is destiny** — stacking correlated betas buys nothing (which is why momentum/reversal/TSMOM/
  PCA stacking did nothing: they were secretly the same beta).
- **Method.** A real combiner (`backtest/combine.py`): measures the cross-leg correlation
  matrix, blends by inverse-vol + a half-Kelly leverage cap (de Prado's robust default at low
  N), reports combined Sharpe vs the equal-weight null, and translates Sharpe → expected
  daily/annual return. Tested on 6 *real* legs: pairs basket (MN), **EAR-PEAD hedged (MN, new
  in Case 18)**, rsi-mr (beta), cross-sectional (MN), turn-of-month, pre-FOMC.
- **Result.** The legs are genuinely uncorrelated (avg |off-diagonal corr| **0.04**), and the
  inverse-vol blend beats the equal-weight null (0.99 vs 0.97).
- **➡️ The EAR-PEAD lift (+0.16) is RETRACTED — the leg didn't generalize.** We had reported that
  adding the EAR-PEAD beta-hedged leg lifted the combined inverse-vol Sharpe **0.83 → 0.99 (+0.16)**,
  hailing it as the combiner finally doing what the math promises once fed a second real edge. But
  EAR-PEAD subsequently **failed its fresh-symbol holdout (−0.52, Case 18)** — its in-sample Sharpe
  was specific to the fitted universe — so that +0.16 was a lift from a **non-generalizing leg** and
  must be discounted. Honest status: the combiner still works *mechanically* (uncorrelated legs, beats
  the null), but the edge-supply bottleneck is **NOT** eased — we are back to one validated leg (pairs)
  plus weak diversifiers, combined ≈ **0.83**. The lesson compounds: a leg must pass an out-of-universe
  holdout *before* its combiner contribution counts. (The combined still
  trails the rsi-mr *beta* leg's raw 1.18, but that leg is pure market exposure; the blend is
  near-market-neutral with a far better drawdown profile — a different, more durable risk object.)
- **Honest ROI translation.** At the achieved combined Sharpe (~0.99) and a 6% vol target:
  ~6% / year ≈ **~2.3 bps/day expected, under ~36 bps/day of noise (noise ≈ 16× the edge).**
  The edge is *invisible* day-to-day. **"X% per day" targets are noise-mining** — the right
  scoreboard is combined OOS Sharpe (deflated for trial count via DSR) and max drawdown.
- **Verdict.** ⚙️ **The method is real and is the single biggest lever — but it remains
  edge-supply-starved.** EAR-PEAD briefly looked like the second leg (+0.16) but **failed its
  fresh-symbol holdout (Case 18)**, so the lift is retracted and we are back to one validated leg
  (pairs) plus weak diversifiers. The path to higher combined Sharpe is still **more genuinely
  uncorrelated legs** — but each must clear an *out-of-universe* holdout before its contribution
  counts; this session supplied none. Not faster trading of any single one.

## Case 17 — Overnight→intraday cross-sectional reversal 🔴 (REAL anomaly, untradeable here)

- **Hypothesis (the "tug of war," Lou-Polk-Skouras 2019).** A stock's overnight return
  (prev_close→open) and its intraday return (open→close) are *negatively* related
  cross-sectionally — overnight winners give it back intraday. The **tradeable, no-lookahead**
  form: the overnight return is fully known *at the open*, so rank the universe on it, go LONG
  the overnight losers / SHORT the winners, enter at the open, capture that day's intraday move,
  and **go flat every night** (no overnight beta — a bull market can't flatter it). A clock we
  had never tested. (`alpca/backtest/overnight.py`)
- **Method.** 195-symbol daily universe, 5 years, `adjustment="all"` bars (which removes
  dividend/split artifacts from the overnight gap). Dollar-neutral, top/bottom 20%, signal-
  lookback sweep (1–5 days), **a momentum control** (long winners — should fail if the reversal
  is real), in-sample/out-of-sample split, and a **cost sweep** (the book turns over ~2×/day, so
  cost is the whole ballgame). DSR-deflated for 36 trials.
- **Result — the anomaly is unambiguously REAL.** At **zero cost** the best reversal
  (lookback 1) earns Sharpe **0.93**, PSR 0.98, **DSR 0.90**, and the momentum control is
  strongly **negative (−2.26)** — the directional control nails it, so this is signal, not luck.
- **…but it is a pure transaction-cost mirage.** The ~2×/day turnover eats everything:

  | per-leg cost | Sharpe | OOS | DSR |
  |---|---|---|---|
  | 0 bps (gross) | **0.93** | 0.66 | 0.90 |
  | 1 bps | 0.26 | 0.02 | 0.42 |
  | **2 bps (realistic)** | **−0.41** | −0.63 | 0.05 |
  | 5 bps | −2.41 | −2.57 | 0.00 |

  The edge breaks even around **~1.2 bps/leg** and is **negative by 2 bps**. And 2 bps is
  *optimistic*: this needs an open-print and close-print fill every day, and we measured Alpaca
  fills at ~1.2s with slippery opens — real costs are worse than the sweep's 2 bps column.
- **Verdict.** 🔴 **Real anomaly, zero tradeable edge on our venue.** The cleanest demonstration
  yet of the project's recurring law: **a statistically real cross-sectional effect (DSR 0.90
  gross, control-confirmed) is not the same as a tradeable edge** — high-turnover market-neutral
  strategies die to spread+impact, exactly as intraday cross-sectional momentum (Case 3/4) and
  1-min market-neutral did. Rejected as tradeable; **kept as the canonical "cost wall" case.**
- **What could revive it (not pursued).** Only a structurally cheaper expression: hold the
  reversal across *multiple* days to amortize turnover (likely kills the signal — it lives in
  the single open→close window), or trade it on a venue with maker rebates and sub-cent spreads
  (not Alpaca). Neither is available to us.

## Case 18 — EAR-PEAD, beta-hedged with a cheap index short 🟡→❌ (failed the fresh-symbol holdout)

- **The idea (rescuing Case 14).** Surprise-PEAD was downgraded because its short leg (a) had no
  edge and (b) died to adverse-selection borrow. EAR-PEAD changes two things: (1) the signal is
  the **3-day earnings-announcement RETURN (EAR)** — a price-only measure (no analyst estimates),
  which the literature finds gives a longer, cleaner, mostly-LONG-side drift than SUE; and (2) the
  problematic single-name short is replaced by a **cheap general-collateral INDEX short (SPY)** to
  neutralize market beta. You keep the long alpha and hedge the beta with a borrow you can actually
  get. (`alpca/backtest/ear_pead.py`, `scripts/test_ear_pead.py`)
- **Method.** 40 large-cap names with 30-yr AV earnings, 5-yr daily bars. EAR = return over the
  first 3 post-report bars (close before report → end of window); enter the drift `skip_after_ear`
  bars *after* the window (no overlap, no look-ahead), hold 40 days. Three modes judged: **long**
  (long high-EAR only — must beat buy-and-hold or it's beta), **neutral** (long high / short low,
  the single-name short), **beta_hedged** (long high-EAR, short SPY by the long leg's beta). DSR
  deflated for 37 trials via a clean entry-threshold sweep of the hedged sleeve.
- **Result.**

  | mode | Sharpe | IS | OOS | return | maxDD | beta |
  |---|---|---|---|---|---|---|
  | long-only | 0.94 | 0.84 | **1.19** | +117% | −26.5% | ~0.87 |
  | neutral (single-name short) | −0.46 | −0.60 | −0.12 | −18% | −24.7% | — |
  | **beta_hedged (index short)** | **0.67** (thr1.5) | **0.70** | **0.66** | +40% | **−12.2%** | 0.87 |

  Long-only **beats SPY buy-and-hold** (Sharpe 0.94 vs 0.83, +117% vs +87%) — but it carries
  beta ~0.87, so most of that is market exposure. The **single-name short is −0.46 again** (the
  short-leg problem is intrinsic — the index hedge is the right fix). The **beta-hedged residual
  is the real alpha:** at the robust threshold (1.5) Sharpe **0.67 with IS 0.70 ≈ OOS 0.66**
  (remarkably stable) and only −12% DD; the best-Sharpe config (thr 1.0) reaches 0.75 with
  **PSR 0.95 / DSR 0.89** (deflated 37 trials).
- **Why it matters.** This is the **strongest earnings result so far** and the first new sleeve in
  a long time that survives the honest bar. It is **comparable to the pairs basket** (OOS 0.54),
  **uncorrelated** to it (event-clock vs contemporaneous cointegration), and — unlike surprise-PEAD
  — has a **tradeable short side** (SPY is GC; no adverse-selection borrow). That makes it a
  legitimate **second leg for the combiner** (Case 16's bottleneck was edge supply — this helps).
- **Profit-per-day, honestly.** Sized at half-Kelly, the beta-hedged sleeve maxes near **~8 bps/day
  geometric** (Sharpe 0.67–0.75), under ~20× that in daily noise — the long-only number looks
  bigger (~13 bps) only because it's *leveraged beta you already own by holding SPY*. The sleeve
  worth adding is the hedged alpha, because it **diversifies** rather than duplicates the market.
  Max profit/day = push the DSR-surviving Sharpe up (breadth) and size to Kelly — not trade faster
  (Case 17 showed frequency turns Sharpe 0.93 → −0.41).
- **OVERFIT AUDIT (passed — `scripts/audit_overfit.py`).** Because this is the one leg we put in
  the combiner, it got a dedicated audit built to *catch* overfit, with **fixed a-priori params
  (thr 2.0, not the cherry-picked 1.5)**:
  - **Hedge lookahead** — replacing the full-sample beta with a **trailing 126-day** beta (no
    lookahead) does not hurt: Sharpe 0.54 → **0.68** (it actually improves; Δ +0.14). So the result
    was *not* leaning on the full-sample hedge ratio.
  - **Regime stability** — per-calendar-year Sharpe on the trailing-hedge sleeve is **positive in
    6/6 years**, including the 2022 bear (+0.19) and the flat 2024 (+0.05); strong in 2021 (1.80),
    2025 (0.81), 2026 (1.55). Not concentrated in one lucky period — the key anti-overfit signal.
  - **Deflation honesty** — DSR is stable as the trial count escalates to the project's true search
    breadth: **0.86 @37 → 0.83 @200 → 0.82 @400 trials** (PSR 0.94). The significance is not an
    artifact of under-counting trials.
  Honest residual caveats: still only 40 symbols and one 5-yr span (per-year is sub-period
  stability, not a truly independent holdout); 2022/2024 are thin-positive.
- **Subset resampling (within the 40) looked fine — and gave false comfort.** 200 random 20-symbol
  draws were 91% positive (median 0.35); we read that as "not carried by a few names." But every
  subset shares names with the 40 — it tests *sub-sampling*, not *generalization to new names*.
- **⛔ THE TRUE FRESH-SYMBOL HOLDOUT FAILS (`scripts/test_ear_pead_holdout.py`, Mode A).** With the
  AV quota reset we fetched **19 disjoint large-caps the strategy had never seen** (BKNG, BLK, C,
  COF, CME, CI, BMY, BSX, COST, CSCO… — finance/health/consumer-heavy) and ran the **frozen a-priori
  params** on them. Result:

  | set | symbols | Sharpe |
  |---|---|---|
  | TRAIN (original 40) | 40 | +0.68 |
  | **HOLDOUT (19 fresh, disjoint)** | 19 | **−0.52** |
  | pooled | 59 | +0.37 |

  On the fresh names the **long leg has no drift at all** (long-only Sharpe **+0.07** ≈ zero), the
  hedged sleeve is **negative across every threshold** (−0.09 @1.0 → −0.52 @2.0 → −0.93 @3.0), and it
  is positive in only 2/6 years (2021/2022) then negative 2023–26. **The earnings-drift signal is
  simply absent out-of-universe.**
- **The "it's just a sector effect" rescue — TESTED and REFUTED (no new data).** A fair advocate
  asks: maybe EAR-PEAD is real in growth/tech (where PEAD drift is documented) and only failed
  because the holdout skewed to heavily-arbitraged financials. Splitting the *original 40* by sector
  kills that hope: the **growth/tech subset (12 names) is NEGATIVE (−0.25)**, while the
  **value/defensive/financial subset (27) is +0.62** — the *opposite* of the hypothesis, and the
  in-sample "edge" simply **flips sign by how you slice the universe** (the signature of noise, not a
  signal). And the 40's value names work in-sample (+0.62) yet the *fresh* holdout (which included
  financials) still failed (−0.52) — so fresh names of *any* sector fail. The edge lives in the
  specific 40 symbols' realized paths, nowhere else. Chasing a fresh growth-tilted set would be
  p-hacking a refuted hypothesis; we did not.
- **Verdict.** 🟡→❌ **DEAD — symbol-specific overfit, sector-rescue refuted.** The audit it *did*
  pass (regime stability, trailing hedge, DSR) and the 91%-positive subsampling were all **inside the
  original 40**; the gold-standard fresh-symbol holdout comes back **−0.52**, and the sector split
  shows the in-sample "edge" flips sign by slice (growth −0.25 / value +0.62) — so it is **not a
  universe-wide edge, not even a sector edge; it lives in the specific 40 symbols' paths.** The
  combiner lift it provided (Case 16, +0.16) is **retracted**. This is the cleanest overfit catch in
  the document — a sleeve that passed *every in-universe test* (audit, regime stability, DSR,
  subsampling) and still died the moment it left the fitted symbols, *before* any capital was risked.

## Case 19 — Lead-lag cross-predictability (price-only, walk-forward) ❌ (fitted noise)

- **Hypothesis (scout's strongest *new* mechanism).** Some stocks' returns lead others' (slow
  information diffusion / inattention): if leader *i* moves today, follower *j* moves tomorrow.
  Estimate the leader→follower map, trade followers on their leaders' lagged moves, dollar-neutral.
  A genuinely new clock (info-diffusion), uncorrelated to everything we have. (`alpca/backtest/lead_lag.py`)
- **Method (built to *not* fool ourselves).** The source repo reports Sharpe ~1.95 — but the
  leader→follower map is itself a fitted object (195 candidate leaders per follower, pick the top
  few = a selection-bias minefield). So: (1) **WALK-FORWARD** — `C[i,j]=corr(lead i, follow j)`
  estimated only on a 252-day train window, traded on the next 63-day held-out window, rolled;
  (2) the decisive **SHUFFLE PLACEBO** — re-run with each follower's leaders assigned *at random*.
  If the real map doesn't beat the placebo, the structure is noise. 195 symbols, 15 OOS windows.
- **Result — rejected on two independent grounds.**

  | n_leaders | real Sharpe | placebo Sharpe | real − placebo |
  |---|---|---|---|
  | 3 | −1.70 | −1.61 | −0.09 |
  | 5 | −1.27 | −1.60 | +0.33 |
  | 10 | −1.02 | −1.14 | **+0.11** |

  (1) **The real map barely beats its own shuffled placebo** (+0.11 at best, within noise) — the
  leader→follower structure carries essentially no information a random assignment doesn't. (2)
  Even at **zero cost** the best config is only Sharpe 0.27 (DSR 0.26), and the daily signal's
  turnover turns it **negative by 1 bp** (−1.02 / DSR 0.00 at 2 bps). So the *mechanism* fails the
  placebo **and** the tiny gross residual fails the cost wall.
- **The engine is sound (true negative, not a bug).** On synthetic data with a *built-in* lead-lag,
  the real map cleanly beats the placebo (locked by `test_lead_lag.py`). So the negative on real
  data means the market has no exploitable *price-only* lead-lag at daily frequency — not that the
  test is broken.
- **Scope honesty.** This rejects the **data-driven, price-only** version (the one we can test on
  our panel). The academic *supervised* version (customer/supplier or shared-analyst economic
  links) might fare better, but it needs a linkage graph we do not have — untested, not endorsed.
- **Verdict.** ❌ **Fitted noise.** Exactly the overfit the scout flagged: a gaudy in-sample Sharpe
  that the walk-forward + placebo dissolve. A clean win for the discipline — the placebo control is
  now part of the toolkit for any "learned structure" edge.

## Case 20 — Gap reversion, multi-day hold ❌ (large-cap gaps are informational, not mean-reverting)

- **Hypothesis (scout #1, statarb "gap" signal).** A stock that gaps down at the open over-reacted
  and bounces back; a gap-up fades. Long the biggest gap-DOWNs / short the gap-UPs, dollar-neutral.
  Built deliberately *different* from Case 17 (which was intraday-only, flat overnight, ~2×/day
  turnover): here the position is **held for `hold` days via overlapping tranches**, so only ~1/hold
  of the book rotates daily — the one structural way past the cost wall that killed Case 17.
  (`alpca/backtest/gap_reversion.py`; no look-ahead — the gap of day *t* is entered at *t*'s close
  and the held book that earns day *t*'s return excludes that day's gap.)
- **Result — no edge, even gross.** Unlike Case 17 (real gross intraday reversal), the multi-day
  gap-reversion has **no gross edge on large caps**: hold 10 is Sharpe −0.14 at **zero cost**,
  −0.39 at 2 bps (DSR 0.05). And the tell — the **gap-MOMENTUM control is consistently *better*
  than reversion** at longer holds (hold 20: momentum OOS +0.56 vs reversion −0.83).

  | hold | reversion Sharpe | momentum Sharpe | turn/day |
  |---|---|---|---|
  | 1 | −0.80 | −0.95 | 1.52 |
  | 5 | −0.50 | −0.23 | 0.30 |
  | 10 | −0.39 | −0.10 | 0.15 |
  | 20 | −0.57 | **+0.20** | 0.08 |
- **Why (the economic read).** Gap reversion is documented in *small/illiquid* stocks, where the
  gap is liquidity-driven over-reaction that snaps back. In **S&P large caps the overnight gap is
  mostly information** (earnings, macro, guidance) that *continues*, not noise that reverts — so the
  reversion sign is wrong and the momentum control mildly works. Lowering turnover (the Case-17 fix)
  doesn't help because **there is no gross edge to protect** here in the first place.
- **Verdict.** ❌ **Rejected on our universe.** Not a cost-wall casualty like Case 17 — a *signal*
  failure: the gap-reversion anomaly does not exist in liquid large caps. (It might in a small-cap
  universe we don't trade; out of scope.) Useful boundary on the reversal family: Case 17's edge
  was intraday microstructure, real but uncapturable; the multi-day large-cap version isn't even
  there to capture.

## Case 21 — Short-interest (borrow-fee) tilt ❌ (the 1-year "lead" was a lucky window)

- **Hypothesis (scout #1, "hard-to-borrow" signal — on REAL data, not a proxy).** Days-to-cover
  (DTC = shares short / avg daily volume) is the fundamental driver of borrow fees; the documented
  short-interest anomaly says heavily-shorted names underperform (short sellers are informed). LONG
  low-DTC / SHORT high-DTC, dollar-neutral. **Data is real Nasdaq short interest** (bi-monthly
  settlement, cached to the Passport via `scripts/download_short_interest.py`), *not* a price proxy.
- **Method (the honesty is in the frictions).** `alpca/backtest/short_interest.py`. Two things that
  usually kill this: (1) **publication lag** — SI as of a settlement date is not disseminated for
  ~8 trading days, so each signal is acted on `pub_lag=10` days later (no look-ahead); (2) **the
  borrow crux** — the high-DTC names the anomaly says to short are the *expensive-to-borrow* ones,
  so a DTC-scaled borrow fee is charged on the short notional (the same wall that sank surprise-PEAD).
  Rebalances bi-monthly → **turnover ~0.010/day**, structurally cost-robust (the property Cases
  17/19/20 lacked). Judged over the **active window only** (SI covers just the last ~1 yr of the
  5-yr daily panel; scoring the flat pre-data years would fake an IS/OOS split).
- **First pass looked great — on ONE year (Nasdaq).** On 56 symbols / ~1 yr of free Nasdaq SI, the
  anomaly posted Sharpe 2.93 gross, **2.34 after DTC-scaled borrow**, control mirror −2.98, PSR 0.99,
  DSR 0.98 (clean deflation), turnover 0.010/day. It looked like the only scout-#1 signal to clear the
  bar — and we flagged it 🟡, explicitly *power-limited to one regime, validate on FINRA before trusting.*
- **The multi-regime test (FINRA, ~9 yr / 188 symbols / ~201 obs each) DEMOLISHES it.** FINRA's
  `consolidatedShortInterest` (public, no auth; `scripts/download_short_interest_finra.py`) covers
  2017–2026, so the signal is active across all 6 calendar years of the daily window including the
  2022 bear:

  | variant | Sharpe (6-yr) | per-calendar-year |
  |---|---|---|
  | anomaly, no borrow | 0.91 | weakly real, control −1.03 |
  | anomaly + 3% flat borrow | 0.63 | not significant |
  | **anomaly + DTC-scaled borrow (the crux)** | **−0.42** | 2021 −1.09, 2022 −2.38, 2023 +1.14, 2024 −1.12, 2025 +0.87, 2026 +0.28 |

  Three independent failures: (1) the **gross** signal is *weak* (0.91, not ~3) and **regime-specific
  — positive in only 3/6 years**, badly negative through 2021/2022/2024; (2) under the **realistic
  DTC-scaled borrow** it goes **net-negative (−0.42, DSR 0.07)** — you must pay top borrow to short
  exactly the crowded high-DTC names, and that eats the thin gross signal (the *same* wall that sank
  surprise-PEAD, Case 14); (3) the gaudy 1-year Nasdaq number was a **lucky window** — Nasdaq's free
  feed only covered 2025–2026, the one good stretch (+1.18 / +0.86).
- **Verdict.** ❌ **Downgraded 🟡 → ❌ — not a tradeable edge, and a textbook 1-year artifact.** The
  short-interest anomaly is *real but weak* gross, regime-specific, and **net-negative after the borrow
  cost on the very names it tells you to short.** It is NOT a third leg. The constructive payoff is
  methodological: **the multi-regime FINRA test caught an edge that a 1-year sample had rated DSR 0.98**
  — the single cleanest demonstration in this whole document of *why we do not trust short windows*, and
  a direct vindication of holding it as a "lead, sized at zero" rather than shipping it.

## Case 22 — 52-week-high momentum (George-Hwang) ❌ (the anomaly inverts on our universe)

- **Hypothesis (from the literature, not our prior set).** Proximity to the trailing 52-week high
  predicts returns: long stocks near their high (ratio = close / 252-day-high ≈ 1), short those far
  below, dollar-neutral. George & Hwang (2004) found this *subsumes* traditional momentum and is more
  durable — attributed to anchoring/underreaction. Picked as a "more-likely-to-generalize" momentum
  form. (`alpca/backtest/high_52w.py`; overlapping-tranche hold → low turnover; no look-ahead.)
- **Result — it does NOT replicate; it inverts.** On the 195-name large-cap universe, 2021–2026, the
  **near-high (momentum) leg is NEGATIVE at every hold (−0.58 to −0.66)**, while the **reversal leg
  (long far-below-high) is positive (+0.53 to +0.64)**. Here, names near their 52-week high
  *underperform* and beaten-down names mean-revert — the opposite of the published anomaly.
- **And the reversal side is regime-concentrated, not a free edge.** Its positivity is carried by
  **2023 (+2.39, the post-2022-bear bounce)**; per-year it is +0.0/+0.24/**+2.39**/−0.55/+0.96/−0.68
  — negative in 2024 and 2026. Low turnover (0.045/day) and a tidy −8% DD make it *look* attractive,
  but it is the same regime-dependence trap (mean-reversion works in recovery years) that we now know
  to distrust.
- **Verdict.** ❌ **Rejected.** The momentum direction fails outright (negative, regime-unstable, DSR
  0.07); the reversal direction is positive only because one recovery year dominates. A famous
  documented anomaly **did not survive contact with our universe/period** — a useful reminder that
  "it's in the literature" is not a substitute for testing it here. (The reversal lead could be
  re-examined later with a fresh-symbol + out-of-2023 holdout, but regime-concentration makes it a
  low-priority maybe, not an edge.)

## Case 23 — Accruals anomaly (Sloan), on SEC EDGAR fundamentals 🟡→❌ (fails the fresh-symbol holdout)

- **Hypothesis.** The first FUNDAMENTAL edge we've tested — orthogonal to all our price/positioning
  work, so a real one would *diversify* the combiner. Earnings made of accruals (vs cash) are
  lower-quality and mean-revert: ACC = (NetIncome − OperatingCashFlow) / avg(TotalAssets); LONG
  low-ACC (cash-backed) / SHORT high-ACC, dollar-neutral, annual rebalance.
- **Data — SEC EDGAR `companyfacts` (free, NO quota, no auth).** The right foundation: full multi-year
  fundamentals for the whole universe, sidestepping the AV quota wall that limited everything else.
  `scripts/download_fundamentals_edgar.py` (ticker→CIK map → NetIncomeLoss / operating CFO / Assets,
  annual 10-K) cached 164/195 symbols. **No look-ahead:** the accrual is acted on only from the 10-K
  **filing** date (~2 months after fiscal year-end), not the period-end. (`alpca/backtest/accruals.py`)
- **Result — the structural profile the rejects all lacked.** Best decile/quintile **Sharpe ~0.44**
  (control mirror −0.45 → sign-confirmed), **turnover 0.006/day** (annual → essentially cost-free,
  escapes the wall that killed Cases 17/19/20/22), and **regime-robust: +5/6 calendar years**
  (+1.15 / +0.33 / +1.07 / +0.31 / −0.62 / +0.98), positive through the 2022 bear. DSR ~0.75.
- **Generalization — promising but NOT proven (the EAR-PEAD lesson applied).** Disjoint split-halves
  are both positive (+0.50 / +0.31), BUT random-subset resampling is only **76% positive (median
  Sharpe +0.23)** — *weaker* than EAR-PEAD's 91% was, and EAR-PEAD then **failed** its truly-fresh
  holdout. These halves/subsets are all drawn from the same 164 symbols, i.e. the *same evidence
  class* that misled us before. So we explicitly do **not** call this validated.
- **⛔ THE TRULY-FRESH-SYMBOL HOLDOUT FAILS.** Fetched bars + EDGAR fundamentals for **16 genuinely
  disjoint** large/mid-caps (ABNB, ADP, AIG, AON, CAG, CLX, DAL, EL, HLT, HSY, KDP, PYPL, ROP, STZ,
  UBER, WM) and ran the **frozen** rule. Result: **TRAIN-164 +0.30 vs FRESH-16 −0.47 (tf 0.25) /
  −0.28 (tf 0.33)** — negative in 4/6 years (2021 −1.6, 2022 −0.7, 2024 −0.5, 2025 −0.4). Consistent
  across quantiles. The accrual edge **does not generalize to unseen symbols** — exactly EAR-PEAD's
  failure mode (Case 18).
- **Verdict.** 🟡→❌ **DOWNGRADED — fails out-of-universe, same as EAR-PEAD.** The regime-robust,
  cost-free, sign-confirmed *in-universe* profile (the best of any candidate) was, once again, **not
  sufficient** — only the fresh-symbol holdout adjudicated, and it came back negative. This is the
  **third** candidate (after EAR-PEAD-18 and SI-tilt-21) where in-sample/in-universe evidence —
  even regime stability, DSR, and subset resampling — passed on an edge that then died out-of-sample.
  *Legitimate (not post-hoc) caveat:* the academic accruals literature conventionally **excludes
  financials/insurers** (AIG/AON/MMC and banks have non-standard accruals), and the fresh-16 skews to
  insurers + recent IPOs (ABNB/UBER/PYPL) where the ratio is ill-defined; a financials-excluded,
  sector-neutral accrual on a *broad* fresh set is a legitimate future refinement — but we do **not**
  claim it as a rescue (EAR-PEAD's sector-rescue was refuted, Case 18). Held as ❌ until a clean
  broad-universe fresh test says otherwise.

## Case 24 — Value composite (E/P + FCF/P + B/P), on SEC EDGAR ⚠️ (generalizes, but too weak)

- **Hypothesis.** The value premium — long cheap / short expensive — is the most-studied anomaly and
  orthogonal to momentum/positioning/accruals, so a surviving version diversifies the combiner. A
  cross-sectional composite of three yield metrics: **E/P, FCF/P, B/P** (mean of percentile ranks),
  long the cheap quantile / short the expensive, dollar-neutral, monthly-ish rebalance.
  (`alpca/backtest/value.py`; the EDGAR fetcher was extended for shares / CapEx / book equity.)
- **Data + no look-ahead.** Market cap = shares × price, so each yield re-prices daily; the
  fundamental is the most recent 10-K known at the rebalance day (EDGAR `filed` date). 164 symbols
  with shares. Low turnover (~0.009/day).
- **Result — the FIRST fundamental that does NOT fail the fresh-symbol holdout, but it's thin.**
  Main-universe Sharpe **~0.14** (tf 0.2), and crucially the **fresh-symbol holdout is *positive*
  (+0.11 to +0.54 across runs ≈ the in-sample level)** — value *generalizes* to unseen symbols, unlike
  EAR-PEAD (Case 18) and accruals (Case 23) which went negative out-of-universe. But it is **weak and
  regime-dependent**: strongly positive in the **2022 value rotation (+1.85)**, negative in the
  growth-led **2026 (−1.58)** — the value premium's well-known cyclicality. DSR ~0.3.
- **Verdict.** ⚠️ **Real and generalizing, but too weak to clear the bar.** A *different* failure mode
  from the others: it is **not overfit** (fresh-symbol holdout positive — a genuine first for the
  fundamental family), it just doesn't carry enough Sharpe (~0.14 < the 0.2 cost-survival floor, DSR
  far below 0.95) and is heavily regime-timed. Honest read: the value premium is faintly present in
  this large-cap universe but not a standalone edge here. *Possible future lift* (not pursued as a
  rescue): sector-neutralization, a small-cap tilt (where value is stronger), or as a **regime-timed
  overlay** (only on in the conditions where its 2022-type payoff concentrates) — but that risks
  regime-fitting. For now it joins the combiner's bench as a real-but-thin diversifier, not a leg.

## Case 25 — Betting-Against-Beta / low-vol ❌ (needs leverage we don't have)

- **Hypothesis.** Low-beta / low-vol stocks earn higher *risk-adjusted* returns (Frazzini-Pedersen
  BAB; Ang et al. low-vol). Cross-sectional, dollar-neutral: long the low-`signal` quantile / short
  high, monthly rebalance. (`alpca/backtest/low_beta.py`; `signal` ∈ {beta, vol}; price-only, no new data.)
- **Result — rejected, exactly as the a-priori caveat warned.** Unlevered dollar-neutral:
  **beta → Sharpe −0.63, vol → −0.95** on the 195-universe, fresh-symbol holdouts −0.17/−0.16.
  Only 2022 (the high-beta crash) was positive; every other year negative. Low turnover (~0.01/day)
  doesn't save it because there's no edge to protect.
- **Why.** The BAB *factor* levers the low-beta leg to be beta-neutral and harvests the risk-adjusted
  premium; this dollar-neutral, **unlevered** version is dominated by the raw beta differential, so in
  a 2021–26 bull (high-beta outran low-beta) long-low/short-high simply loses. The anomaly is real but
  risk-adjusted, and capturing it needs leverage we can't readily apply on the venue.
- **Verdict.** ❌ **Rejected** — a beta/leverage artifact in disguise on this venue, not a tradeable
  market-neutral edge. (Confirmed across both the beta and vol signals.)

## Cases 26–32 — The factor zoo on large caps ❌ (documented premia, thin/absent here)

- **What.** Seven well-cited cross-sectional factors run through one generic engine
  (`alpca/backtest/factor.py`) with the full bar (main + disjoint fresh universe + per-year + cost +
  DSR + rail): **asset growth** (26), **net share issuance** (27), **ROA** (28), **MAX/lottery** (29),
  **idiosyncratic vol** (30), **residual momentum** (31), **vol-managed momentum** (32). Fundamental
  ones use the cached EDGAR multi-year data; price ones use the daily bars + SPY. Zero new data.
- **Result — none clear the rail.**

  | factor | main | OOS | fresh-holdout | +yrs | DSR |
  |---|---|---|---|---|---|
  | asset growth | −0.24 | −0.24 | −0.06 | 1/6 | 0.09 |
  | net issuance | −0.50 | −1.11 | −0.47 | 1/6 | 0.03 |
  | ROA | −0.23 | −1.49 | −0.18 | 3/6 | 0.09 |
  | MAX / lottery | −0.83 | −1.61 | −0.12 | 1/6 | 0.00 |
  | idio-vol | −0.88 | −1.47 | −0.08 | 2/6 | 0.00 |
  | residual momentum | +0.05 | +0.65 | **−0.15** | 1/6 | 0.25 |
  | vol-managed momentum | +0.11 | +0.77 | **−0.08** | 2/6 | 0.29 |
- **Why.** These premia are documented in **broad universes (incl. small/mid-caps) over long
  histories**, and most have **decayed in large caps post-2015**. Our universe is 195 liquid
  large-caps over 2021–26 — exactly where they're weakest. The fundamental factors are weak/negative
  even in-sample; the two momentum variants show a positive in-sample tail but a **negative
  fresh-symbol holdout** (the overfit signature we now reflexively check).
- **Verdict.** ❌ **All seven rejected on our universe.** Not a coding issue (the engine is unit-tested
  and the synthetic-signal control profits); it's that large-cap factor premia are thin and these
  don't survive the fresh-symbol bar. *Where they might live (not pursued here): a small/mid-cap
  universe — a different venue/data scope.* The reusable factor engine makes any future factor a
  one-liner to test at the same rigor.
- **Meta-finding (Cases 26–34, nine documented factors).** Adding **short-interest change (33)** and
  **gross profitability (34)** — *all nine* well-cited cross-sectional factors are rejected on the
  195-name large-cap 2021–26 universe. This is itself a result: **the classic factor zoo is essentially
  absent in liquid US large-caps over this window** (decayed post-2015; the premia concentrate in
  small/mid-caps and longer histories we don't trade). The only cross-sectional things that have shown
  *any* edge here are the cointegrated-pairs basket (real) and the value composite (generalizes but
  thin). Factor investing, naively ported to our venue, does not work — and now we've *measured* it.

## Case 35 — Financials-excluded accruals & value (the sector refinement) ⚠️/❌

- **What.** The legitimate refinement flagged in Cases 23–24: the accruals literature conventionally
  **excludes financials/insurers** (their accruals are ill-defined). Fetched SIC codes (EDGAR
  submissions API, `sic_codes.json`), dropped the 35 financials (SIC 6000–6799), and re-ran accruals
  and the value composite on the 136 non-financial names + a 22-name fresh holdout.
- **Result.**
  - **Accruals ex-financials:** in-sample **improves markedly — main +0.70, +6/6 years** (vs +0.44
    with financials), *confirming the literature*. **But the fresh-symbol holdout is still −0.51.**
    Excluding financials cleaned the in-sample yet **did not rescue the out-of-universe failure** — the
    overfit is real and deeper than a sector artifact. This **refutes the sector-rescue hypothesis** I
    flagged in Case 23: tested honestly, it does not save accruals.
  - **Value ex-financials:** main +0.09, **fresh +0.22 (still generalizes)**, 3/6 years — no material
    change; still real-but-thin (Case 24 stands).
- **Verdict.** ⚠️/❌ **The refinement was worth testing and is now resolved:** accruals' fresh-holdout
  failure is **not** a financials artifact (it stays −0.51 even excluded) → accruals confirmed dead
  out-of-universe; value is unchanged (generalizes but too thin). A clean closure of the last
  open fundamental hypothesis — by measurement, not assumption.

## Case 36 — Sector-neutral value (the within-sector value premium) ❌ (improves in-sample, kills generalization)

- **What.** Value (Case 24) generalizes — its fresh-symbol holdout stayed *positive* (+0.70), the only
  fundamental that did — but it's too thin (~0.11) to deploy. A raw value composite secretly loads on
  cheap **sectors** (energy/financials cheap, tech expensive), which is a regime-timed sector bet, not
  pure value. The literature says the **within-sector** value premium is the more persistent, robust
  slice. So I **demeaned the composite within sector** (coarse 11-bucket SIC map, drop singletons) and
  re-ranked on the residual — long cheap-vs-sector-peers, short expensive-vs-peers — and ran it through
  the full bar (main 195 + 30-name fresh holdout + per-year + cost + DSR + gate).
- **Result.** Sector-neutralizing **lifted the in-universe Sharpe 0.11 → 0.34** — looks like a win — but
  **the fresh-symbol holdout collapsed from +0.70 to −0.64** (1/6 positive years, DSR 0.49). The sector
  bet was carrying the part that *transferred* to unseen names; stripping it just fit main-universe
  sector idiosyncrasies that don't generalize.
- **Verdict.** ❌ **REJECT** — a textbook in-sample-up / out-of-universe-down overfit, caught only by the
  fresh-symbol holdout. Raw (sector-loaded) value remains the better — but still too-thin — version.
  Neutralization is *not* a free improvement here; it destroyed the one property that made value worth
  keeping.

## Case 37 — Value + Momentum combined ("Value and Momentum Everywhere", AMP 2013) ❌ (out-of-time ≠ out-of-symbol)

- **What.** The single strongest zero-new-data candidate for a second uncorrelated leg. Value and
  cross-sectional momentum are the two most-documented market-neutral premia and are **negatively
  correlated** (cheap stocks have been falling; winners have gotten expensive), so a combined rank
  (long cheap-AND-rising) is historically more regime-stable and higher-Sharpe than either leg. I blended
  a 12-2 momentum rank into the value composite and swept the momentum weight 0.0 → 1.0, each blend
  through main + 30-name fresh holdout + per-year + cost + DSR.
- **Result.** Adding momentum **improved every in-universe metric**: main Sharpe 0.11 → **0.51** (w=0.75),
  OOS *time-split* −0.52 → **+0.76**, DSR 0.28 → 0.62. **But the fresh-symbol holdout went negative for
  every momentum weight > 0** (−0.16 to −0.29); only pure value (w=0) kept a positive fresh holdout, and
  that's too thin and below the DSR bar. The momentum component overfits the specific main-universe
  names — its winner/loser ranking does not transfer to the 26 fresh symbols (consistent with momentum
  being weak/absent on liquid large-caps, Cases 26–34).
- **Verdict.** ❌ **REJECT**, and a clean methodological lesson: **out-of-sample-in-time is not
  out-of-sample-in-symbols.** The chronological time-split (0.76) waved the combo through while the
  disjoint-symbol split (−0.23) killed it. The fresh-symbol holdout is the binding test; a passing
  time-split is necessary, not sufficient. Momentum is not the missing second leg on this universe.

## Case 38 — Value on a MID-CAP universe ⚠️ (the size-tilt thesis, confirmed but still thin)

- **What.** Every factor-zoo rejection (Cases 26–37) carried the same caveat: *these premia live in
  smaller, less-efficient names, not our 195 liquid large-caps.* So I built a **fresh ~137-name S&P
  MidCap-400 universe** (zero overlap with the large-caps or the large-cap fresh holdout — a true
  out-of-universe test), pulled 5yr daily bars (Alpaca) + EDGAR fundamentals, and ran the **same raw
  value composite** three ways: the full mid-cap universe, an internal train half, and a disjoint
  **holdout half** (fresh-symbol generalization *within* mid-caps).
- **Result — the first fundamental to clear BOTH halves of the bar, just not by enough:**
  - **Full mid-cap Sharpe 0.21** — nearly **double the large-cap 0.11**. The value premium *is*
    genuinely stronger in smaller names, exactly as the literature predicts.
  - **Holdout-half +0.14, 4/6 positive years** — it **generalizes** to fresh mid-cap symbols (it does
    not collapse the way sector-neutral value and value+momentum did on their fresh holdouts).
  - But **0.21 is still sub-deployable**: DSR 0.37, only 3/6 positive years on the full universe → it
    **fails the falsification rail** on DSR + regime-robustness.
- **Verdict.** ⚠️ **Promising lead, not a deploy.** This is the *first* time a candidate both **beat its
  large-cap version** (premium strengthens as size falls) **and kept a positive fresh-symbol holdout** —
  the size-tilt direction is validated by measurement, not assumed. The honest read: value is real and
  size-dependent, but a 0.21 sleeve is still too thin to deploy alone and too thin to lift the combiner
  (value at 0.11 already diluted it). The lead it opens: push *further* down the cap spectrum to true
  **small-caps (S&P 600)**, where the premium should be strongest — Case 39.

## Case 39 — Value on a SMALL-CAP universe ❌ (the size-tilt is NOT monotonic — it inverts)

- **What.** If value strengthens from large (0.11) to mid (0.21), does it keep climbing into true
  small-caps? Built a fresh **~110-name S&P SmallCap-600 universe** (zero overlap with large/mid),
  same 5yr bars + EDGAR fundamentals, same raw value composite + internal disjoint holdout.
- **Result.** **It inverts: full small-cap value −0.26, holdout −0.19** (DSR 0.08, dies to the cost
  wall too). The size-tilt is **non-monotonic — large 0.11 → mid 0.21 → small −0.26**, peaking at
  mid-cap and reversing in small-caps.
- **Why (honest).** The cheap-yield screen in small-caps loads onto the **value trap**: distressed,
  unprofitable, levered small names that look cheap and stay cheap (or die). In the 2022–24 rate-shock
  regime those got crushed, and small-caps broadly were in a multi-year bear — so the small-cap value
  *premium* was negative in this specific window. Whatever the mix of structural value-trap and
  regime, the measurement is clear: pushing further down the cap spectrum does **not** extend the lead.
- **Verdict.** ❌ **REJECT** for deployment, and it **bounds Case 38**: mid-cap is a genuine local
  sweet spot, not the first rung of a ladder. Stop descending; the action is *at* mid-cap.

## Case 40 — Mid-cap value + LIGHT momentum (the AMP combo, where the premia are real) ⚠️ (best generalizing fundamental yet, still sub-rail)

- **What.** On large-caps, any momentum blend destroyed value's fresh-symbol generalization (Case 37).
  But that was a universe where *neither* premium is strong. On **mid-caps**, where value is genuinely
  real (Case 38), the AMP thesis — value and momentum are negatively correlated, so a light combination
  is additive and more regime-stable — gets a fair test. Swept the momentum weight on mid-cap value,
  watching the disjoint holdout at every step.
- **Result — opposite of large-caps, and the strongest generalizing fundamental in the program:**
  - **value + mom w=0.25:** full Sharpe **0.39** (vs 0.21 pure value, 0.11 large-cap value), and the
    **holdout RISES to +0.24** (from +0.14) — a light tilt improves generalization, it doesn't break it.
  - **w=0.5+:** the holdout flips negative (−0.19, −0.37) — so there is a real *sweet spot* at a light
    tilt, not a monotonic knob (a tuned-to-death overfit would look monotonic).
  - Through the full gate at honest 70-trial DSR: it **clears the two hardest tests** (beats the
    size-baseline AND generalizes to fresh symbols) but **fails the rail** on regime-robustness (3/6
    positive years) and **DSR 0.51 < 0.9**.
- **Verdict.** ⚠️ **Real but sub-threshold — the best second-leg *candidate* we've found, not yet a
  deployable edge.** Unlike every Case-36/37/39 reject (which failed generalization), this one
  generalizes; it just isn't *robust* enough yet (concentrated in 3 of 6 years) or significant enough
  (DSR 0.51) to deploy or to add to the combiner under our own discipline. The honest next moves: widen
  the mid-cap breadth (more names → tighter deciles, push DSR + regime-coverage), confirm the w=0.25
  sweet spot isn't selection (a priori "light tilt" choice, re-tested on more data), and only *then* a
  **date-aligned** combiner test against the pairs basket. A genuine lead, held at zero size until it
  clears the rail.

## Case 41 — The factor zoo ON MID-CAPS ("find more like mid-cap value") ✅ partial (momentum comes alive too)

- **What.** Value came alive on mid-caps (Cases 38/40) when it was dead on large-caps. The obvious
  question: which *other* premia from the rejected zoo (Cases 26–34) revive on less-efficient mid-cap
  names? Widened the mid-cap universe to **~289 names** (bars + EDGAR fundamentals on My Passport) and
  re-ran the whole zoo through the same bar — full universe + disjoint fresh-symbol holdout + per-year +
  DSR. (The breadth jump also let us re-test Case 40 at full breadth: its **fresh holdout STRENGTHENED
  from +0.24 to +0.44 on 78 disjoint names** — generalization improving with breadth is the *opposite*
  of overfit, raising confidence it's a real premium; still sub-rail on regime+DSR though.)
- **Result — the momentum family revives, the quality/lottery factors stay dead:**

  | factor (mid-cap) | full | holdout | +yrs | DSR | large-cap was |
  |---|---|---|---|---|---|
  | **vol-managed momentum** | **0.42** | **+0.25** | **4/6** | 0.53 | ❌ rejected (Case 32) |
  | **residual momentum** | 0.27 | +0.25 | 3/6 | 0.40 | ❌ rejected (Case 31) |
  | value + light momentum | 0.36 | +0.51 | 3/6 | 0.47 | the Case-40 lead |
  | asset-growth / ROA / net-issuance / gross-profit | ~0 … −0.70 | ≤0 | — | ≈0 | dead → still dead |
  | MAX-lottery / idio-vol | −0.6 | <0 | — | ≈0 | dead → still dead |

- **Verdict.** ✅ **The hunt paid off:** **residual & vol-managed momentum — both rejected on large-caps
  — generalize on mid-caps**, vol-managed momentum being the most regime-robust single factor in the
  whole program (4/6 years, DSR 0.53). The *pattern* is now clean and explainable: on mid-caps the
  **value and momentum premia are real and generalizing; the quality/issuance/lottery factors are not.**
  None clears the rail *alone* — but value and momentum are negatively correlated, which sets up Case 42.

## Case 42 — Multi-factor mid-cap combiner: a genuine SECOND-EDGE candidate ✅ (meets the out-of-universe + out-of-regime bar; pending forward + pairs-corr)

- **What.** Case 41 left three generalizing mid-cap legs (value, residual-mom, vol-managed-mom), each
  real but each failing the rail on regime-robustness (3–4 of 6 years). Value vs momentum is negatively
  correlated, so blending should be *more regime-stable* than any leg alone. The decisive metric isn't a
  bigger Sharpe — it's whether the blend is positive in MORE years (clears the 60% regime bar) while
  keeping a positive fresh-symbol holdout. Measured the correlation matrix + the inverse-vol blend's
  per-year profile (`combine.evaluate_combo`, date-aligned streams).
- **Result.**
  - **Correlation matrix confirms the structure:** value vs momentum **−0.26 / −0.30** (genuine
    diversification); the two momentum legs **+0.91** (redundant — same factor).
  - **Pure value + vol-managed momentum (the −0.26 pair, ~50/50 inverse-vol):** **5/6 positive years**
    (2021 +0.45, 2022 +0.81, 2023 +0.08, 2024 +0.36, 2025 −0.41, 2026 +0.76), **maxDD −7.1%**, Sharpe
    ~0.37 — combining lifted regime-robustness from each leg's 3–4/6 to **5/6, clearing the regime bar**
    that blocked Case 40.
  - **value+light-momentum + vol-managed momentum** trades regime for level: **4/6 years, Sharpe 0.53,
    DSR 0.62** — a clean Sharpe↔regime frontier.
- **Verdict.** ✅ **The strongest second-edge candidate the program has produced — and the first to meet
  the same bar that validated the pairs basket.** It is market-neutral, **generalizes out-of-universe**
  (fresh-symbol holdout +0.44), is **regime-robust out-of-regime** (5/6 years), has a **shallow −7% DD**,
  and sits at Sharpe ~0.4–0.5 — the *same quality tier as the pairs basket* (OOS ~0.5). It does **not**
  clear the strict DSR-0.9 falsification gate (DSR ~0.5–0.6) — **but neither does the pairs basket**;
  our actual "validated" standard has always been out-of-universe + out-of-regime generalization, not
  DSR 0.9. **Honest status: a second-edge *candidate*, not yet deployed.** Two confirmations remain
  before capital: **(1) measure its correlation with the large-cap pairs basket** (different universe +
  mechanism → prior ρ≈0, but must be measured — if low, the master combiner finally gets a true second
  leg), and **(2) a forward paper-track with independent resolution**, exactly as the pairs basket got.
  This is the answer to the binding constraint (edge supply) we've been hunting — pending those two gates.

## Gate #1 (for Case 42) — correlation vs the deployed pairs basket ✅ (uncorrelated, ρ = −0.03)

Before the mid-cap blend can earn a combiner slot it must *diversify* the edge we already trade. Built
both daily-return streams on the same calendar (large-cap pairs basket at the validated top-10/ADF
config; mid-cap value+vol-mom blend) and date-joined 1,255 days. **Correlation ρ = −0.033** — essentially
zero, confirming the mechanistic prior (cointegration mean-reversion on large-caps vs cross-sectional
fundamentals+trend on a disjoint mid-cap set share nothing). The two-sleeve inverse-vol book is positive
in **6/6 years** — the uncorrelated legs cover each other's weak years. *(Caveat: the pairs leg here is an
in-sample full-calendar screen, so the combined Sharpe/DSR is inflated vs the walk-forward ~0.83; the
robust, regime-independent results are the ρ≈0 and the 6/6-year coverage, not the level.)* Gate #1 passes.

## Case 43 — SURVIVORSHIP-BIAS point-in-time re-test ❌→ flips value to momentum (the session's most important result)

- **What.** Every mid-cap result (Cases 38/40/42) was measured on names that **exist today** — omitting
  the value-traps that went bankrupt or delisted (BBBY, RAD, ENDP, AVYA, BIG, CANO, …). Value's LONG leg
  buys cheap names, so excluding the cheapest-that-died inflates it. Verified Alpaca serves **delisted
  history up to the delisting date** (BBBY 1256 bars, SIVB stops at its 2023 collapse, …) — so the fix is
  feasible: pulled ~50 delisted mid/small-caps into My Passport and re-ran on survivor-only vs
  survivor+delisted. (Conservative: after delisting the bars stop, so the backtest books ~0, not the
  final gap-to-zero — the TRUE hit is worse than measured.)
- **Result — the dead names FLIP which leg is real:**

  | factor | survivor-only | + value-traps | mechanism |
  |---|---|---|---|
  | value | +0.04 | **−0.45** | value *buys* the dying cheap names → survivorship hid its worst trades |
  | value + light-mom | +0.35 | **−0.45** | the value drag dominates |
  | **vol-managed momentum** | +0.39 | **+1.35** (5/6 yr) | momentum *shorts* the falling names → survivorship hid its BEST trades |
  | residual momentum | +0.24 | **+0.69** | same — shorting losers that collapse |

- **Verdict.** ❌ **Mid-cap VALUE is survivorship-inflated** — the "edge" (and the value+momentum combo
  built on it, Case 42) is largely an artifact of excluding the bankrupt cheap names. **But mid-cap
  MOMENTUM is survivorship-ROBUST and far stronger than the survivor-only backtest showed** (0.39 →
  1.35), because the survivor universe had robbed it of shorting the names that went to zero. The real
  edge here is **momentum, not value.** *Two non-negotiable caveats before believing the 1.35:* (1) that
  gain comes from **shorting stocks going to zero — exactly the hardest-to-borrow / no-locate names** (the
  same **adverse-selection borrow wall** that killed PEAD, Case 14, and the SI-tilt, Case 21); the
  realizable edge sits between 0.39 and 1.35 and **must be re-run with adverse-selection borrow on the
  dying shorts.** (2) The survivorship-clean, borrow-free slice is the **LONG** leg (winners don't
  delist), which points to a **long-momentum / index-hedged-short** construction (the EAR-PEAD pattern,
  Case 18) as the honest deployable form. **The second-edge candidate is reframed: not value+momentum,
  but mid-cap vol-managed MOMENTUM — pending the borrow model and a long/index-hedged build.** This is the
  survivorship control doing exactly its job: it didn't just dock a Sharpe, it **changed which signal we
  believe.**

## Case 44 — Mid-cap momentum under realistic borrow + the long/index-hedged form ⚠️ (real but short-side-only, borrow-gated, magnitude-bracketed)

- **What.** Case 43 said mid-cap momentum is the real edge (0.39 → 1.35 with delisted names) but its
  strength comes from shorting names going to zero — the borrow wall. This stress-tests it: vol-managed
  momentum on survivor-only vs +delisted (point-in-time) universes, under three short-side regimes —
  borrow-free L/S, **adverse-selection borrow** (per-name fee ramping with the short's trailing decline +
  low price; below a floor = **no-locate**, dropped), and a **long-only + SPY-index-hedge** (borrow-free,
  the candidate deployable form). Daily returns winsorized at ±50%/±30% because near-delisting microcaps
  carry mis-adjusted reverse-split artifacts (WW showed a +152× day, PROK +5×) that corrupt both P&L and
  the momentum ranking.
- **Results (winsorized, point-in-time universe):**
  - **borrow-free L/S 1.7** → **adverse-borrow, dollar-neutral-enforced 1.1** (5/6 yr, DSR ~0.95, ~300
    no-locate drops). The edge **survives** realistic borrow + dropping the un-borrowable names + holding
    the book dollar-neutral (so it's alpha, not net-long beta — enforcing neutrality cost only ~0.05).
    Winsorizing *raised* the Sharpe (artifacts were noise that hurt), killing the "it's a data-artifact"
    hypothesis.
  - **long-only + SPY hedge (borrow-FREE): 0.08 survivor, −0.12 point-in-time.** The **long leg has no
    alpha** — the momentum edge is *entirely* in the short leg. The clean, deployable, borrow-free form
    I'd hoped for **does not exist** here.
- **Verdict.** ⚠️ **Real, but not the clean second leg it looked like.** Three honest constraints:
  (1) the alpha is **short-side only** (long/index-hedge is dead), so it **can't escape the borrow wall**;
  (2) even surviving adverse borrow, the deployable magnitude is **bracketed** — survivor-only 0.3 (too
  low, excludes real failures) to cherry-picked-+delisted 1.1 (too high: my +50 delistings are *famous*
  failures, which **over-feeds the short leg** vs a true ~5–10%/5yr delisting rate); (3) the true number
  needs a **representative** point-in-time universe (all delistings, not the dramatic ones) or, decisively,
  a **forward paper-track** — which has zero survivorship/borrow-estimation bias. **Status: a genuine but
  messy short-side momentum anomaly, magnitude 0.3–1.1, borrow-gated — promising enough to forward-track,
  not clean enough to deploy.** The honest second-leg search continues; the pairs basket remains the only
  deployed edge.

## Case 45 — REPRESENTATIVE point-in-time universe: the 1.35 was a cherry-pick artifact ✅ (bracket collapses to ~0.4)

- **What.** Case 44 left momentum's magnitude bracketed 0.3–1.1 because the +50 delistings were *famous
  failures* (BBBY/ENDP/…) that over-feed the short leg. The fix: pull **all 1,707 US-listed delistings**
  from Alpaca's inactive-assets API (no selection on outcome), and on the **SIP feed** (the free IEX feed
  has near-zero delisted coverage; SIP has it all). Filter to mid-cap-caliber **in 2021** (median price
  $5–500, >$3M/day dollar-volume over the first 120 bars) — selected on *start-of-window* characteristics,
  **outcome-blind**, so bankruptcies and acquisitions both qualify. Re-ran momentum on the survivor SIP
  universe + this representative delisting set (feed-consistent).
- **The representative delisting set is dominated by ACQUISITIONS, not failures.** Of the kept names,
  ~all exited *flat/up* (DNB taken private, TPTX acquired by BMS, ACC/American Campus, MTOR/Meritor by
  Cummins, MANT/ManTech…) — **0 failure-like**. Genuine mid-caps ($300M–$10B) rarely go bankrupt; they
  get bought at a premium. The famous bankruptcies I'd cherry-picked had mostly *already* fallen out of
  mid-cap by 2021, so an honest 2021-mid-cap filter excludes them — correctly.
- **Result — the survivorship "boost" was the artifact:**

  | mode (point-in-time) | cherry-picked failures (Case 44) | **representative (Case 45, full 1,707 set)** |
  |---|---|---|
  | borrow-free L/S | 1.35 | **0.43** |
  | adverse-borrow, dollar-neutral | 0.93 | **0.30** |
  | long + SPY-index-hedge (borrow-FREE) | −0.12 | **+0.23** |

  Adding *representative* delistings doesn't help momentum — it slightly **hurts** it (survivor 0.46 →
  +delisted 0.43), because acquisitions jump to a fixed deal price and flatten, giving the short leg fewer
  good trades. The 0.39 → 1.35 jump in Case 43 was entirely the cherry-picked bankruptcies over-feeding
  the short leg. **The 0.3–1.1 bracket collapses to ~0.3–0.45** (75 representative mid-cap delistings,
  74 acquisitions / 1 failure — definitive on the complete 1,707-name pull).
- **Verdict.** ✅ **Magnitude resolved: mid-cap vol-managed momentum is a MODEST ~0.4 Sharpe edge** (≈0.43
  gross, ≈0.30 after adverse-selection borrow), not the 1.1+ the cherry-picked test implied — and the
  survivorship effect on momentum is **negligible (slightly negative), not a boost**, once delistings are
  representative. The **borrow-free long/index-hedged form is mildly positive (+0.23)** — representative
  delistings include acquired *winners* the long leg holds into a buyout premium — so a deployable,
  borrow-free sleeve may exist at modest Sharpe (the cherry-picked −0.12 was the famous failures starving
  the long leg). **Net: momentum is a real but modest (~0.4) edge, uncorrelated with the pairs basket
  (gate #1), with a plausible borrow-free long/hedged form (~0.2) — a credible weak second leg for the
  combiner, pending a forward track.** *(Numbers definitive on the complete 1,707-name pull: 75
  representative mid-cap delistings; stable from the 48→50→75-name re-runs.)* The lesson compounds Case 43: a
  cherry-picked "fix" for survivorship can itself be a survivorship bias in the opposite direction — only
  the **full, outcome-blind** point-in-time universe gives the honest number.

## Case 46 — SURVIVORSHIP point-in-time re-test of the DEPLOYED edge (the pairs basket) ✅ ROBUST

- **What.** We'd turned the point-in-time lens on the momentum *candidate* (Cases 43–45) but never on the
  *deployed* edge — the cointegrated-pairs basket, also validated on a survivor-only large-cap universe.
  If a pair's leg later delisted (an acquisition freezes its price at the deal value; a failure craters
  it), the spread stops mean-reverting and the trade takes a loss the survivor universe never sees. Built
  a feed-consistent SIP point-in-time large-cap universe: 195 survivors (re-pulled on SIP) + 32
  representative large-cap delistings (Alpaca inactive-assets, filtered to large-cap-caliber in 2021,
  outcome-blind — mostly acquisitions: CERN, XLNX, NUAN, AZPN, DNB, ACC…).
- **Surfaced a tooling gap first:** the production `walkforward_pairs` builds its calendar from the
  **global timestamp intersection**, so adding any short-history name collapses the window to ~zero
  (it returned 0 windows). That intersection design **structurally excludes delisted names — survivorship
  baked into the implementation.** So I wrote a **`delisting_aware_walkforward`** (union calendar,
  per-window screen among names with ≥80% bars, per-pair backtest; a leg that delists mid-window closes
  at its last real price — capturing the acquisition freeze / crash). Promoted it into `pairs.py` with
  tests; on a survivor universe it reproduces the legacy walk-forward exactly.
- **Result.**
  - Legacy survivor WF **+0.831** — **reproduces the validated 0.83 exactly on the SIP feed** (independent
    confirmation the deployed edge is real and feed-robust).
  - Delisting-aware: survivor **+0.831 → +delisted +0.934** (delta **+0.10, slightly UP**), with only
    **8 delisted-leg trades across 3 names** (DM, DNB, EDR) entering the traded top-10 over the whole
    walk-forward.
- **Verdict.** ✅ **The deployed edge is ROBUST to survivorship — it is NOT inflated.** Adding
  representative large-cap delistings barely moves the WF Sharpe (and if anything nudges it up), because
  (a) the strict 5% ADF + top-10-by-stability screen rejects delisted names' truncated/trending series
  (the spurious "WBT cointegrates with everything" from a naive full-sample screen vanished under proper
  no-lookahead walk-forward training), and (b) large-cap delistings are overwhelmingly *acquisitions*
  (price freezes near the deal value → spreads don't blow up). The same brutal test that demoted value
  (Case 43) and right-sized momentum (Case 45) **clears the pairs basket.** New reusable control banked:
  `delisting_aware_walkforward` (the survivorship-honest pairs WF). *(Caveat: 32 delistings, ~all
  acquisitions; a wave of large-cap bankruptcies — historically rare — would test the crash case harder.)*

## Case 47 — The HONEST two-sleeve combined book (pairs + momentum) ❌ DILUTES (deploy pairs alone)

- **What.** The combiner's whole purpose is to turn uncorrelated legs into a better book. Gate #1 (Case 42)
  had hinted a pairs+momentum combo helps (ρ≈0, 6/6 positive years) — but it used **in-sample pairs
  returns** and the **full** momentum window. This does it honestly: pairs leg = **walk-forward OOS**
  daily returns (the new dated `delisting_aware_walkforward`, deployed top-10/5%-ADF config); momentum leg
  = the borrow-free **long/index-hedge** daily returns on the mid-cap SIP universe; date-aligned on their
  945 common OOS days and inverse-vol blended.
- **Result.**
  - Pairs WF leg **+0.83** (matches deployed); **momentum long/hedge over the common window −0.15** —
    NEGATIVE. ρ = **+0.06** (uncorrelated, as gate #1 found).
  - Equal-weight combined **0.14**, inverse-vol combined **0.47** — both **below the pairs leg alone (0.83)**.
- **Verdict.** ❌ **The combiner DILUTES — deploy the pairs basket ALONE.** The momentum sleeve's
  borrow-free form is **negative over the honest overlapping window (2022–2026)**; its earlier ~0.21 was
  carried entirely by **2021**, which the pairs walk-forward (first ~252 days = training) excludes. You
  cannot lift a +0.83 leg by blending a −0.15 leg, however uncorrelated. **This supersedes the optimistic
  gate-#1 reading:** in-sample pairs + a momentum window that included 2021 manufactured a lift that the
  rigorous apples-to-apples (WF pairs + same-date momentum) erases. The combiner remains **edge-supply-
  limited** — we still have one deployable edge (pairs) and no positive-over-the-relevant-window second
  leg. The momentum forward track stays live (sized tiny) to let reality speak, but expectations are now
  honestly **low**. New capability banked: dated WF OOS returns → any future leg gets a clean,
  date-aligned, in-sample-free combiner test.

## Case 48 — Cross-sectional calendar seasonality (Heston-Sadka same-month) ❌ (real + uncorrelated, but unstable; lift was a partial-year artifact)

- **What.** The binding constraint (Case 47): a market-neutral leg that is positive over 2022→ AND
  uncorrelated with pairs. Cross-sectional seasonality fits structurally — rank each name by its OWN
  return in the SAME calendar month in PRIOR years (strict no-lookahead), long the historically-strong-
  this-month names, short the weak; P&L on a calendar clock, orthogonal to trend and pairwise
  mean-reversion. Zero new data, and (needing prior-year history) it only trades from 2022 on — exactly
  the forward-relevant window. Built it as a no-lookahead factor signal (`cross_sectional_seasonality_
  signal`, +2 unit tests) and ran the full bar on large- and mid-caps.
- **First look — promising:** dead on large-caps (−0.33), but on **mid-caps**: full +0.11, holdout
  **+0.40** (generalizes — *better* on fresh symbols), **low turnover 0.056**, and over the pairs-OOS
  window a standalone **+0.35**, **uncorrelated with pairs (ρ=+0.06)**. The first candidate with the
  properties momentum lacked (positive over the window, uncorrelated, generalizing, cost-robust). The
  inverse-vol combined book was 0.81 and a pairs-dominant 75/25 weighting showed a *lift* to **0.88**.
- **Then the robustness test killed it:**
  - **Per-year: 2023 +0.57, 2024 +0.89, but 2025 −0.69, 2026 −0.19** — it worked for two years then
    *reversed*; it has been negative the last ~1.5 years. Not regime-robust.
  - The 0.88 "lift" is **entirely a partial-2026 (Jan–Jun) artifact**. Excluding 2026: combined-no-2026
    **0.767 < pairs-alone-no-2026 0.778** — it **dilutes**, exactly like momentum.
- **Verdict.** ❌ **REJECT as a second leg.** Genuinely uncorrelated (ρ=0.06) and the *cleanest-looking*
  candidate yet, but **5 years is too little history for a calendar premium** — the signal is unstable
  (negative 2025–26) and the only combiner lift came from an unreliable partial year. A naive read
  (combined 0.81–0.88, positive, uncorrelated, generalizing holdout) would have shipped it; the
  **with/without-2026 split caught it.** The signal is kept in the toolkit (correct, no-lookahead,
  reusable) but the edge is not there. Still **one deployable edge (pairs); no positive-and-robust
  second leg.** The combiner remains edge-supply-limited.

## Case 49 — Short-volatility / variance-risk-premium 🟢 (the FIRST leg that LIFTS the book — with an un-sampled-tail asterisk)

- **What.** The binding constraint after Cases 47–48: a leg positive over 2022→ AND uncorrelated with
  pairs AND robust across years. The variance-risk-premium (implied vol > realized on average → selling
  vol pays) is a genuinely different signal source from equity stat-arb and trend, so a priori
  uncorrelated. Tested the **borrow-free long-SVXY** sleeve (post-2018 −0.5× short-VIX ETF; just buy it)
  and a short-VXX sleeve, with the full bar PLUS the two things that decide short-vol — the **2022
  drawdown** and **tail-correlation with pairs** — and the Case-48 partial-year split.
- **Result — it clears every gate the others failed, and lifts the combiner:**
  - **Standalone long-SVXY: Sharpe 0.59, 6/6 positive years** (even 2022 +0.03), **but maxDD −46%**
    (short-VXX −78%) — the tail is real.
  - **ρ vs pairs = +0.04; TAIL ρ = +0.12** (on pairs' worst-10% days short-vol loses only −0.6%/day —
    *mild*, not a severe tail trap vs pairs).
  - **Combined (inverse-vol): Sharpe 1.08 vs pairs-alone 0.83, DSR 0.90** (first candidate to clear the
    strict 0.9 DSR bar), and the combiner sizes short-vol to just **12%** → **combined maxDD −5.5%,
    identical to pairs-alone.** Lift survives the **ex-2026 check (1.18 vs 0.78)** and is positive in 3/4
    leave-one-year-out (drop-2024 +0.46, drop-2025 +0.42; drop-2023 −0.09 — 2023 carries some).
- **Verdict.** 🟢 **The first genuine second leg — materially better than momentum (diluted) and
  seasonality (partial-year artifact).** It is positive 6/6 years, uncorrelated (ρ=0.04), robust across
  years, and *actually lifts* the book (0.83→1.08, DSR 0.90) while inverse-vol sizing keeps the combined
  drawdown unchanged (−5.5%). **The non-negotiable asterisk:** short-vol is **negatively skewed with an
  un-sampled catastrophic tail** — 2021–2026 contained no volmageddon (no Feb-2018/Mar-2020-style spike),
  so Sharpe/DSR/−5.5%-DD all **understate the true risk**; a vol spike can take SVXY −50% in days (at 12%
  weight ≈ −6% to the book *plus* whatever pairs does, with the left tail fatter than measured). **Honest
  position: a real diversifying leg, deployable only at small (≤~12%) size with explicit tail management
  (hard cap, and a vol spike is the known failure mode) — not a free lunch.** Put on a forward
  paper-track alongside pairs+momentum, sized tiny, tail caveat documented; let live reality (including
  the eventual spike) adjudicate. The combiner finally has a leg that lifts it — with eyes open.

## Case 50 — Short-vol TAIL STRESS test ✅ (the 8% cap survives a simulated volmageddon)

- **What.** Case 49's metrics understate short-vol's risk because 2021–2026 had no volmageddon (worst
  SVXY day in-sample: only −21%). The responsible close is not to wait for the spike but to *simulate* it
  and confirm the deployed hard cap protects the book. Injected a catastrophic single-day SVXY shock
  (−40% ≈ Feb-2018-scale on the post-2018 −0.5× fund; −50% ≈ realistic worst; −70% ≈ beyond-historical /
  model risk) at the combined book's **worst** day (joint stress — vol spikes often coincide with pairs
  spreads widening), across 5% / 8% / 12% sizing.
- **Result — at the deployed 8% cap:** a −40% spike costs −4.3% on the day → −7.1% combined drawdown; a
  realistic **−50% spike → −5.1% day / −7.9% drawdown**; a beyond-historical −70% gap → −6.7% / −9.4%.
  At 12% the −70% case reaches −11.0%; at 5% it's −8.0%. The base (un-stressed) drawdown is −5.3%.
- **Verdict.** ✅ **The 8% hard cap is validated — the sizing IS the tail management.** Because short-vol
  is a small slice of a pairs-dominant (92%) book, even a catastrophic −50% volmageddon only deepens the
  combined drawdown from −5.3% to −7.9%, and a beyond-historical −70% gap stays under −10%. The known
  failure mode is survivable at the deployed size; no re-cap needed. This is the honest discipline a
  negatively-skewed leg demands — the disaster is quantified *before* it happens, and the position is
  sized so it can't break the book. Short-vol stays on the forward track at 8%, eyes open.

## Case 51 — Cross-asset TREND / managed-futures (the short-vol tail hedge that wasn't) ❌

- **What.** Short-vol's one weakness is its crash tail; the ideal 3rd leg would be *crisis alpha* —
  cross-asset trend-following (CTAs) is famous for profiting in vol spikes (2022 was a banner year for
  trend). So I built the canonical diversified managed-futures sleeve on a 15-ETF cross-asset basket
  (equities/bonds/gold/commods/dollar/credit): per asset, sign of trailing 12-month return, vol-scaled to
  equal risk, summed (it goes SHORT, so the *timing* must earn its keep) — and tested it against the two
  controls that catch the TSMOM illusion (buy-and-hold, and long-only/no-timing), plus correlation to both
  legs, the short-vol tail-hedge, and the 3-leg combiner.
- **Result — fails on every axis:**
  - **Timing is illusory:** trend Sharpe **0.31** vs buy-and-hold **0.82** and long-only-no-timing
    **1.12** — both controls *beat* it. The shorting/timing actively hurts (whipsaws); only the
    vol-scaling has value. (Reproduces Kim-2016 on cross-asset.)
  - **Crisis-alpha thesis FAILS in-sample:** per-year **2022 −0.72, 2023 −1.02** — trend *lost* in 2022,
    the exact year it should have shone (my basket is equity/credit-heavy; the clean bond-short /
    commodity-long signals got diluted and whipsawed).
  - **No tail hedge:** ρ vs short-vol **+0.17**; on short-vol's worst-10% days trend averages
    **−0.15%/day** — it does *not* offset the crash.
  - **3-leg combiner DILUTES:** 0.93 vs the 2-leg pairs+short-vol book **1.07** (inverse-vol over-weights
    the weak trend leg at 42%).
- **Verdict.** ❌ **REJECT.** A genuinely appealing thesis (trend as a crash hedge for short-vol),
  falsified by measurement on every count. A faster-signal / cleaner-asset-class construction *might* do
  better, but the controls (long-only beats timing by 0.8 Sharpe) put a high burden of proof on that —
  the timing is the problem, not the tuning. **Book stays at two legs (pairs + short-vol).** The hunt's
  honest hit rate this session: 1 win (short-vol) in 5 candidates — edges remain scarce, exactly as the
  program has found all along.

## Case 57 — Mid-cap cointegrated-pairs as a 2nd leg ❌ (the mechanism *generalizes* — it's just too weak to *lift*)

The highest-prior 2nd-leg hunt: take the **one mechanism that works** — mean-reversion of cointegrated price residuals,
which gave WF 0.83 on large-caps and *strengthened* under survivorship (Case 46: +0.83→+0.94) — and ask whether it
generalizes to a **disjoint mid-cap universe** as a diversifying leg. Built on the exact validated cadence
(train=252/test=63/top-10/5%-ADF/2bps) via the delisting-aware walk-forward, two-phase: build + survivorship gate, then
(only on survival) a full gauntlet. The result is the most *instructive* rejection in the book because it cleanly
separates two questions the platform usually conflates.

- **It CLEARS the survivorship bar — the first mid-cap family to do so.** PIT Sharpe **+0.28** vs survivor **−0.43**; the
  disjoint-symbol-half holdout (md5-parity split, each half its own PIT universe) is **positive in BOTH halves**
  (selection +0.16 / holdout +0.34). This is the exact out-of-universe bar that *inverted* mid-cap value (Case 43:
  +0.35→−0.45), mid-cap momentum (Case 45), reversal (Case 53) and EAR-PEAD (Case 55) into negatives. Pairs survives it:
  the mechanism is genuinely **not name-specific** — it really does carry to never-selected mid-cap symbols. ρ to the
  large-cap book = **−0.009** over the shared 945 days: a *perfect* diversifier in character.
- **But it FAILS on strength, and the leg gate (not the survivorship bar) caught it.** Honest read of the same numbers:
  the **survivor universe is a net loser (−0.43)** — the +0.28 is carried by adding 75 delistings (an *inverted*
  survivorship delta: dead names *lifting* a losing book via ~11 idiosyncratic merger/crash freezes — CADE/DM/EDR/MUDS/
  SAFM — not a robust mechanism). It is a **cadence knife-edge**: train sweep 189→−0.55, **252→+0.28**, 315→−0.20,
  378→−0.55 — positive at *exactly one point*, negative on both neighbors; ADF-off inverts it to −0.14 (vs large-cap,
  which stays positive 252→315). **DSR 0.060** at honest n_trials=30 — statistically indistinguishable from best-of-30
  noise (t≈0.86); two *consecutive* losing years (2022 **and** 2023). The leg gate fails **3/5** (lifts=n, robust_loo=n,
  partial_year_safe=n), and the honest **same-window** inverse-vol combiner lift is **−0.04**: a 0.28-Sharpe sleeve at
  ~0.42 inverse-vol weight **dilutes** the 0.83 book to 0.79. Uncorrelated + positive is *necessary but not sufficient* —
  the sleeve must be strong enough that blending nets positive, and this one isn't.
- **Verdict.** ❌ **Reject as a 2nd leg — but a different rejection than all the others.** Not a survivorship artifact
  (PIT-positive), not overfit-to-names (disjoint halves both positive), not redundant beta (ρ≈0) — simply **too weak**:
  cadence-fragile, DSR≈noise, and combiner-diluting. This is the cleanest demonstration that the platform's two upstream
  bars are **complementary**: the survivorship/holdout bar catches *fake-but-strong* edges; the leg gate's
  lift/robustness checks catch *real-but-too-weak* ones — and a candidate must clear **both**. (Methodology banked: an
  *inverted* survivorship delta — PIT > survivor — is a yellow flag, not a green one; it can mean a few delisting events
  are propping up a losing book. And the **cadence map** was decisive again, as with the large-cap 378-inversion.)
  Scoreboard unchanged: **one validated edge (large-cap pairs).** *Possible non-deployment future use: as a tiny
  diversifying sleeve **inside** the pairs book under a non-inverse-vol weighting — but at −0.04 the gap is real, not noise
  to optimize away, so not pursued.*

## Case 56 — Three negatives in one gauntlet: the formulaic-alpha zoo, the pairs OU-refinement, defensive low-beta ❌❌❌

A 4-agent honest-gauntlet sweep of the **uncommitted, unvalidated work** sitting in the repo — built but never run
through the bar. Three independent candidates, one shared lookahead audit. Every one rejected on a *different* honest
gate; the scoreboard holds at one thin edge (pairs). This is the harness doing its job — the value is rejecting cheaply.

- **The published formulaic-alpha zoo — DEBUNKED.** `alpha101.py` implements 21 *de-collinearized family
  representatives* of the Kakushadze **Alpha101** library and the overlapping **Alpha158/Qlib158** and **GTJA191** sets
  (one clean representative per microstructure family — rank-reversal, price-volume corr, decay-momentum, K-bar
  microstructure, returns-vol — so the test covers the *class*, not 450 near-clones). Run through the same bar the rest
  of the zoo faced (main 194-sym universe + **disjoint 30-sym fresh-symbol holdout** + 70/30 OOS + per-year + 2bps cost
  + Deflated Sharpe with n_trials=21 + an explicit Bonferroni PSR≥0.9976): **all 21 rejected.** The best factor
  (`A_a4_tsrank_low`) tops out at **DSR 0.596** — far below the 0.90 gate — and 0/21 clear Bonferroni. The headline is
  the **out-of-universe inversion**: the factors that looked strongest in-sample flip sign on never-seen symbols
  (`gap_reversal` OOS **+1.29 → fresh −0.16**; `intraday` OOS **+1.01 → fresh −0.77**). High in-sample/OOS Sharpe is
  pure overfit to the *selection universe*. The famous "101/158/191 alphas" collapse on our venue exactly like value,
  momentum, and seasonality did. *(Confirmed lookahead-clean — see audit below.)*
- **Lookahead audit — clean, but caught a real correctness bug.** An adversarial audit verified **no look-ahead** in any
  of the 21 builders — confirmed *empirically* by corrupting day T-1's OHLCV and checking every earlier signal row stays
  byte-identical (no builder reads a future index), plus the engine's strict **1-bar lag** (trade on `signal[t-1]`, earn
  `ret[t]`, no same-bar close fill). It did find one genuine bug: `_decay_linear` had its weights **inverted**
  (`arange(d,0,-1)` put the *heaviest* weight on the *oldest* day — an anti-decay filter), mislabeling which momentum
  variant `decay_mom` tested. **Fixed** (recent-heavy `arange(1,d+1)`); the verdict is unchanged (recent-weighted decay
  momentum is still momentum = beta, already debunked, and `decay_mom` did not survive either way). *No survivor was ever
  manufactured by a bug — the gauntlet's no-lookahead guarantee holds.*
- **Pairs OU-refinement — INERT, nothing to commit.** The uncommitted `cost_cal_entry` (per-pair
  `act_entry_z = max(entry_z, 4·cost_frac/ou_std)`) + `ou_sizing` (leg ∝ `min(|z|/act_entry_z, 1)`) refinement to the
  *one real edge* produces **bit-for-bit identical** daily returns to the validated baseline (WF 0.83, −5.5% DD) on both
  the fitted large-cap set AND a disjoint mid-cap holdout — each flag alone and together, to <1e-12. Root cause
  (instrumented): at 2bps, `4·cost_frac = 0.0008` while the log-price spread `ou_std ≈ 0.02–0.15`, so the cost-floor
  `0.0008/ou_std` is always ≪ `entry_z = 2.0` → `act_entry_z = 2.0` always → zero pairs skipped, no reshape. The flags
  would only bind at **>~20bps cost or far tighter spreads**. Lookahead-clean (act_entry_z from a TRAIN-only OU fit;
  ou_sizing from trailing z; hedge fit on TRAIN). **No improvement to the deployed basket.** The co-tested
  `regime_monitor` overlay is worse than inert — it **DESTROYS** the edge (WF 0.83 → **−0.32**, ret +14.9% → −1.1%): the
  rolling-ADF gate flattens pairs exactly when the spread mean-reverts hardest. Both kept default-OFF; the footgun is now
  loudly labeled in `pairs.py`.
- **Defensive low-beta — reject_beta.** Classic Betting-Against-Beta (long-low-β / short-high-β) on 265 mid-caps
  (2021–2026) is a **loser**: Sharpe **−1.06**, −50% return, −53% DD. The *only* positive sign is the reverse
  (long-high-β) at +1.05 — and it is **pure beta** (high SPY correlation; the defensive book's β to SPY is −0.65). A
  low-β long sleeve is dampened market exposure, not alpha; it fails the leg gate. (Run on a survivor universe — even the
  survivorship-favored version is negative on the defensive side.)
- **Verdict.** ❌❌❌ Three independent rejections, three different gates (out-of-universe holdout / cost-inertness /
  beta-decomposition). **Zero new edges; scoreboard unchanged: one thin regime-dependent edge (pairs).** The hard-won
  lesson re-confirmed: a high in-sample or OOS Sharpe means nothing until it clears the **fresh-symbol holdout**, and a
  "refinement" to a real edge must *change the returns out-of-universe* to count. Negative results, banked cheaply, so we
  never re-litigate the formulaic-alpha zoo, OU-entry tuning, or low-β again.

## Case 55 — PEAD 3rd-leg hunt, definitive re-test ❌ (fresh-symbol holdout kills it a 3rd time; data-gated)

The requested 3rd-leg hunt. PEAD was rejected twice before (surprise-PEAD → adverse-selection borrow,
Case 14; EAR-PEAD → fresh-symbol holdout, Case 18). Re-ran the strongest variants through the *full
post-audit gauntlet* (adverse borrow · fresh-symbol holdout · per-year + 2024+ · the second-leg gate vs
the cadence-fixed pairs book) on the available data — earnings are **hard data-gated at 40/195** (AV free
tier now rate-limits immediately; full breadth needs a premium key or Finnhub).

- **EAR-PEAD (beta-hedged, trailing-beta index short — the borrow-free form): REJECT.** It clears **all
  five leg-gate checks** (ρ=−0.01, lift +0.23, robust LOO, partial-year-safe), is positive across
  per-year, and is positive on the **2024+ slice (+0.58)** — yet the **fresh-symbol holdout is NEGATIVE
  (−0.22 at thr1, −0.64 at thr2)** on the 19 disjoint names. Same signature as Case 18 (−0.52) and reversal
  (Case 53): leg-gate-clean, out-of-universe-dead. It's specific to the 40 mega-caps it's fit on.
- **Surprise-PEAD L/S survives adverse borrow (+0.65) — but it's BETA, not alpha.** Flat-borrow +0.98 →
  adverse +0.65 looks like survival, but the **short leg alone is −0.34** (a loser), so the dollar-neutral
  L/S is carried entirely by the long (post-beat) leg = bull-market beta. Not a clean market-neutral
  diversifier; consistent with Case 14's "the apparent edge is the long beta leg."
- **Verdict.** ❌ **PEAD is not the 3rd leg** — three rejections now, the decisive one being the
  fresh-symbol holdout (the out-of-universe bar that has caught value, EAR-PEAD ×3, accruals, the momentum
  cherry-pick, and reversal). The recurring methodology point, paid for again: **the leg gate is necessary
  but not sufficient; the out-of-universe holdout is the decisive bar.** EAR-PEAD passing all 5 gate checks
  + 2024+ and STILL failing the fresh holdout is the cleanest demonstration yet. *(Data caveat: a
  definitive full-breadth verdict needs the 195-name earnings, which is blocked on AV's free tier.)*
  Scoreboard unchanged: one thin regime-dependent edge (pairs).

## Case 54 — Post-audit hardening sweep (cadence map · long-vol hedge · regime overlay) — parallel hunt

Four audit-ranked directions run in parallel, each survivorship-PIT-first. Two delivered real value, one
a conditional candidate, one a clean reject:

- **Pairs cadence map → HARDENING APPLIED (a live-config fix).** Gridded the WF across train/test/top_n on
  the large-cap universe. The edge is **robust to selection breadth** (top_n 6/10/15 → 0.79/0.83/0.86, all
  stable) but **fragile to the train (screen) window**: train 252 → **0.83**, 315 → 0.30, **378 → −1.42
  (inverts)**. Test cadence: test=42 is the sweet spot (**0.91**, H1 0.53 — far less regime-front-loaded
  than the deployed 63's H1 0.27). **Decisive catch: the LIVE deploy was screening on `train=378` — the
  −1.42 cadence.** Fixed `compute_pairs_book` + `deploy_pairs_paper` to **train=252** (validated, robust
  interior; the map says never lengthen, and avoid test≥126). The one deployed edge now runs on the
  evidence-chosen cadence instead of a cliff. *(Data caveat: a flaky external-drive read jittered maxDD /
  the 2024-slice but Sharpe + H1/H2 were identical across reads — the load-bearing signal is solid.)*
- **Long-vol / convexity hedge → CANDIDATE, narrowly scoped.** A long-VXX sleeve pays **+1.2 to +1.9% on
  the book's worst-10% days** (real, correctly-signed convexity) at −24 to −48%/yr carry. **On pure pairs
  it HURTS** at every weight (raises maxDD) — the pairs tail is a slow stat-arb grind, not a vol spike, so
  the convexity never fires and the carry dominates. **On the pairs+short-vol book** (the audit's
  undiversified-tail concern) it clearly **works**: VXX 5% → Calmar 0.806→1.136, maxDD 9.6%→4.6% (−5pp),
  Sharpe +0.06 — cuts the tail harder than the return. A *term-structure-managed* version (long-vol only
  when VIXY/VIXM short-end is rich, ~24% of days) halves the carry. **Verdict: reserve it as the hedge for
  short-vol IF short-vol is ever revived as a leg** (the audit downgraded short-vol to OBSERVE, so it's not
  deployed today); do NOT bolt it onto pure pairs. The leg gate FAILS by design (a negative-carry hedge
  can't pass a Sharpe-lift gate) — a hedge is judged on Calmar/maxDD, not lift.
- **Regime overlay on pairs → REJECT.** Pairs lost in 2022 (−0.9), but that hole is a **concentrated
  ~2-month event, not a trailing-regime-detectable state** — a no-lookahead overlay (trailing vol /
  drawdown / dispersion) can't isolate it, and conditioning the book on its own favorable regime mostly
  cuts exposure in the years pairs does *best* (net harm). Even a lookahead oracle (zero 2022 exposure)
  only lifts full-sample Sharpe to 1.06. Keep the pairs book un-overlaid.
- *(Defensive low-beta hunt + the synthesis agent hit the session limit mid-run; defensive-low-beta is
  queued to re-run.)*

## Case 53 — Cross-sectional short-horizon REVERSAL ❌ (passed the leg gate, killed by the survivorship PIT)

- **What.** A documented equity anomaly (long last-week's losers / short last-week's winners),
  market-neutral, a different mechanism + horizon than the pairs basket — the audit (Case 52) flagged it
  as a possible cadence-orthogonal new leg. Ran the FULL post-audit bar: cost-wall sweep + fresh-symbol
  holdout + per-year + the 2024+ slice + the second-leg gate vs the cached pairs book.
- **It looked like the first real post-audit candidate:** large-cap reversal is dead (efficient), but
  **mid-cap reversal (3-day signal, 10-day rebalance)** scored **0.74 at 5 bps** (positive across *all 6*
  window/rebalance configs — not cherry-picked; +0.52 even at 10 bps), fresh-holdout **+0.47**, **2024+
  +1.17** (positive on recent data — the exact check short-vol failed), ρ=−0.04 vs pairs, and **passed
  ALL FIVE leg-gate checks** (LOO 80%, ex-recent +0.20).
- **Then the survivorship point-in-time test killed it:** add the representative delistings back and the
  Sharpe goes **+0.69 → −0.40 (delta −1.08)**, 5/6 → 3/6 positive years. Reversal *longs losers* (the
  survivor universe omits the dead losers that kept falling) and *shorts winners* (it omits the acquired
  names that popped to a deal premium — reversal shorts them and eats the pop). Both adverse cases were
  missing from survivor-only, inflating it.
- **Verdict.** ❌ **REJECT — a survivorship artifact, like value (Case 43).** The decisive methodology
  point: **the leg gate is necessary but NOT sufficient.** Reversal cleared every leg-gate check (positive,
  uncorrelated, lifts, robust, partial-year-safe) AND the cost wall AND the 2024+ slice — and was still
  wrong. **The survivorship point-in-time test is the deeper bar for any long/short equity factor** (it has
  now caught value, EAR-PEAD, accruals, the momentum cherry-pick, and reversal — five artifacts). It belongs
  *upstream* of the leg gate, not after it. Scoreboard unchanged: one thin regime-dependent edge (pairs).

## Case 52 — The adversarial AUDIT (the harness turns its lens on its own claims) ⚠️ (we were overselling)

A 9-agent adversarial workflow audited the session's own code + tried to *refute* the two "validated"
edges. It found the platform had been **overselling its own results** — the single most important finding
of the program, because it's about *us*, not a rejected candidate. Honest corrections, all reproduced
from code + cache (the *signals* are clean — advocate-confirmed no lookahead; secrets clean):

- **Pairs basket is regime-dependent, not "0.83."** WF 0.831 is real but **front-loaded**: first-half
  Sharpe **0.27** (bootstrap CI straddles zero), second-half **1.39**; per-year **2022 −0.9** → 2026 +1.9;
  strip 10 of 945 days and it falls to **0.16** (excess kurtosis +6.77 — a snap-back payoff). It also
  depends on a **fast re-screen cadence** (lengthen train 252→378 → Sharpe **−1.42**). Honest forward
  expectation: **~0.3–0.5, regime-dependent**, not the headline 0.83. Still the best edge we have.
- **Short-vol is NOT a current diversifier.** Its entire lift is the **2022–23 vol-crush**: pre-2024 book
  lift **+0.92** (short-vol standalone +2.17), but **since 2024 the lift is −0.27** (−0.31 since 2025;
  standalone ~0). The "6/6 positive years" was the *standalone* calendar Sharpe and masked that the
  **book** lift went negative. It is "a 2022–23 vol-crush bet wearing a diversifier costume."
- **The combined 1.08 / +0.09-lift headline is pre-2024-driven.** On the paper-relevant 2024+ window the
  deployed-book lift is **negative**. The canonical 0.92 (union) book is also ~25% pairs-absent padding.
- **The ρ=0.04 correlation is real and stress-stable** (no tail-trap — corr does *not* spike in stress),
  but "uncorrelated ≠ hedged": pairs is **dead weight** in a vol spike, so the combined tail is
  dominated by short-vol's own crash (4/5 worst combined days had both legs red).
- **Momentum's gate verdict flipped NO-GO → GO** on appended bars (recency-driven in-sample lift) — the
  gate window isn't pinned; verdict non-reproducible. Stays on probation, unproven out-of-universe.
- **Bugs fixed (this commit):** (1) `delisting_aware_walkforward` bled train-window bars into OOS for
  short-history/delisted legs (`eq[-(test+1):]` tail-slice) — replaced with date-honest joined-timestamp
  emission (didn't affect shipped survivor-only numbers; corrupted any delisting-honest run);
  (2) `adverse_borrow_neutral` ran 100% net-long when all shorts were no-located (beta leak);
  (3) `combine_tracks` now enforces caps internally (not by caller trust); (4) `return_translation` no
  longer hides negative edges as +inf; (5) the parity "nondeterminism" was benign wall-clock latency
  telemetry (fixed the test, not the engine); (6) `leg_gate` now re-fits inverse-vol weights *out-of-fold*
  in its robustness checks; (7) a silent-degradation overlap guard added to the canonical backtest.
- **Net honest scoreboard:** **exactly one thin, regime-dependent edge (pairs, forward ~0.3–0.5)**;
  short-vol is not a current diversifier; momentum unproven; **zero new edges have cleared the project's
  own out-of-universe + out-of-regime bar.** The right next move is a *genuinely* uncorrelated axis —
  ideally a **long-volatility / convexity** sleeve to actually hedge the short-vol crash the correlation
  analysis proved is undiversified — and every candidate must clear the paid-twice holdout bar.

## Methodology capstone — the SECOND-LEG GATE (`alpca/backtest/leg_gate.py`)

The Cases 47–51 hunt judged every candidate against the same gauntlet by hand. That gauntlet is now one
tested call, `evaluate_leg_candidate(candidate_returns, book_returns)`, running the five checks a real
diversifying leg must pass — **each learned the hard way from a failed candidate:**

1. **forward_positive** — candidate Sharpe > 0 over the overlap window (*momentum was negative over
   2022→, Case 47*).
2. **uncorrelated** — |ρ| < 0.30 vs the deployed book (the point of a diversifier).
3. **lifts** — inverse-vol combined Sharpe > book + 0.03 (*momentum & cross-asset trend DILUTE, Cases
   47, 51*).
4. **robust_loo** — the lift is positive in ≥ 60% of leave-one-year-out folds (a lift carried by one
   year isn't robust).
5. **partial_year_safe** — the lift survives excluding the most-recent (often partial) year
   (*seasonality's "lift" was a partial-2026 artifact, Case 48*).

GO iff all five pass; fail-closed and fully explainable (every sub-result returned with its number).
**Validated on real data — it reproduces the hand verdicts exactly:** short-vol → GO (5/5, combined
1.08, lift +0.25); momentum → NO-GO (Sharpe −0.17, lift −0.38, LOO 0%, ex-recent −0.51). With
`scripts/run_leg_gate.py` any future candidate — or the AI loop — gets the full second-leg rigor in one
call instead of a multi-day manual investigation. This is the session's controls distilled into
permanent, automated, tested infrastructure.

## Methodology upgrade — Deflated Sharpe Ratio

Given how many strategies this project has tried (~34 in the registry + the dozen edge
families here), naive p-values overstate significance. The harness now includes the
**Probabilistic** and **Deflated Sharpe Ratio** (Bailey & López de Prado): the DSR tests a
Sharpe against the *expected maximum* Sharpe over the number of trials, so it accounts for
selection bias. Use DSR > 0.95 — not a raw p<0.05 — as the real significance bar going forward.

## Anti-overfitting discipline — "are we fooling ourselves?"

Every reported edge is one query against the same data, so the standing risk is that our
*survivors* are overfit even though our *rejections* are honest. The controls that keep us honest,
each having actually killed or saved a candidate:

1. **Buy-and-hold benchmark** — a long sleeve must beat B&H on return *and* Sharpe or it is labelled
   beta (Cases 2, 18-long).
2. **Walk-forward, not tail-split** — re-fit on train, trade held-out, roll (pairs basket, lead-lag).
   A 70/30 chronological split of one regime is the *weak* form and is flagged as such.
3. **Shuffle placebo** — for any *learned* structure, re-run with the structure randomized; it must
   beat its own placebo (killed lead-lag, Case 19).
4. **Adversarial friction models** — realistic borrow/slippage applied in the direction that hurts
   (adverse-selection borrow downgraded surprise-PEAD, Case 14; the 2bps cost wall killed Cases 17/20).
5. **Regime stability** — per-calendar-year Sharpe; a real edge is positive in most years incl. the
   bear, not concentrated in one lucky period (EAR-PEAD: 6/6 years, Case 18 audit).
6. **No-lookahead hedge/feature** — trailing, not full-sample, estimates (EAR-PEAD's trailing beta).
7. **Honest trial counting** — deflate DSR by the project's *true* search breadth (hundreds of
   configs, not the ~37 of one sweep); an edge that only clears at low trial counts isn't one.

**Honest scorecard (final, after the fresh-symbol holdouts):** only **one** edge remains validated —
the **cointegrated-pairs basket** (quarterly walk-forward, OOS ~0.5). **EAR-PEAD has been downgraded:**
it passed everything *inside its fitted universe* (audit: 6/6 regime-positive, trailing hedge, DSR 0.82
@400 trials; subset resampling 91% positive) but **failed the gold-standard fresh-symbol holdout
(−0.52 on 19 disjoint names, long-leg drift +0.07 ≈ zero)** — its apparent edge was specific to the
tech-heavy mega-cap universe it was built on, and its combiner lift (+0.16) is retracted. **Two stars
of this session were caught by the two cleanest tests:** the SI tilt (DSR 0.98 on 1 yr of Nasdaq) died
on the **multi-regime FINRA** test (net −0.31 / 6 yr); EAR-PEAD (0.68, audit-passed) died on the
**fresh-symbol holdout**. Both were held at zero size and never shipped. **The lesson, paid for twice
in one session:** in-universe and in-sample tests — even regime stability and DSR — can all pass on an
edge that is still overfit; only *out-of-universe* (new symbols) and *out-of-regime* (new years) holdouts
catch it. That is the discipline working exactly as intended, and the honest answer to "are we overfitting?"
— we were on two of three candidates, and the harness caught both before a dollar was at risk.

## What we learned

1. **The harness is the product.** Its job is to *reject*, and it has correctly rejected every
   overfit/beta/illusory candidate — including ones with gorgeous in-sample numbers (pairs
   1.78, PCA 0.99). Select-on-IS / report-on-OOS is the single most valuable discipline.
2. **Market-neutral is the only place real edge has appeared** — and even there it's modest
   (Sharpe ~0.5) and fragile to universe/frequency choices.
3. **Costs are destiny.** Every high-turnover variant (intraday MN, fast momentum, fast
   reversal) dies to spread + impact. Edge that survives is **low-turnover** by necessity.
4. **Right framework, wrong venue** recurs (A-S, residual stat-arb, funding arb): techniques
   that work on L2/perp/rebate venues do not transfer to a top-of-book, taker, spot account.
5. **Beta is not a sin — mislabeling it is.** rsi-mr is deployed live as honest risk-reduced
   beta (the swing job). We just never call it alpha.

6. **Combining is the biggest lever, but it's edge-supply-limited.** The diversification-of-
   Sharpe math is real and our combiner works (legs uncorrelated at ρ≈0.05, beats the equal-
   weight null) — but you can't manufacture a high combined Sharpe from one good leg plus weak
   diversifiers. The constraint is the *supply of genuinely-good uncorrelated edges*.
7. **Daily-ROI is the wrong target.** At any honest Sharpe (0.5–1.2), the daily expected
   return is ~2–5 bps, buried under ~40–60 bps of daily noise (noise ≈ 10–20× the edge). The
   correct scoreboard is **combined OOS Sharpe (DSR-deflated for trial count) and drawdown**.

**Bottom line:** after equities (all families), crypto (daily + hourly), market-making, two
sizing/factor generalizations, a CTA edge, an alt-data probe, a funding signal, seasonality,
a portfolio combiner, and now a realistic **adverse-selection borrow** model — the **84-symbol
cointegrated-pairs market-neutral basket (OOS Sharpe ~0.5)** remains the **only** fully-validated
edge. **PEAD has been downgraded from "strong second candidate" to 🟡:** on a *flat* borrow
assumption it cleared DSR 0.92, but once you model that the worst-miss names you most want to
short are exactly the crowded shorts that go special/no-locate, the dollar-neutral Sharpe more
than halves and **DSR falls to 0.58**. The short leg is a standalone loser (−0.47); the apparent
edge was the long (beta) leg. PEAD is **not dead** — only 24/195 symbols are cached, and full
breadth (plus a real SUE signal instead of raw `surprise_pct`) could revive the short side — but
the bar it now has to clear is **"survive `adverse_borrow` at full breadth,"** not "flat-borrow
DSR > 0.95." This is the anti-delusion machinery working as designed: a realistic friction model
caught an edge that the optimistic friction model had waved through.

**The EAR-PEAD arc — promising, then retracted (Case 18).** The same discipline that killed
surprise-PEAD's short leg appeared to *find a fix*: a price-only **EAR** signal hedged with a **cheap
index short** posted Sharpe ~0.68 on the 40, passed a regime audit (6/6 years) and 91%-positive subset
resampling, and lifted the combiner +0.16 — we provisionally called it a second leg. **Then the
fresh-symbol holdout failed it (−0.52 on 19 disjoint names).** The edge was specific to the fitted
tech-heavy mega-cap universe and does not generalize. So the validated-edge count is back to **one —
the cointegrated-pairs basket.** The honest path to higher *profit-per-day* still runs through **more
genuinely-uncorrelated legs sized to Kelly** — but a candidate only counts *after* it clears an
out-of-universe holdout, not before. We found zero such legs this session; we correctly rejected five
candidates (Cases 17, 18, 19, 20, 21) and shipped none. That is the right outcome: no false edge reached
capital, and the bar for "validated" is now explicitly out-of-universe + out-of-regime.

## Case 58 — Meta-labeling (López de Prado) on the deployed pairs basket ❌ (the secondary model has no out-of-sample skill)

- **The idea.** Don't hunt a new edge — *amplify* the one we've validated. Take the deployed
  cointegrated-pairs basket's primary signals and train a **secondary classifier** (meta-model) to
  predict, **at entry**, whether each individual pair-trade will be profitable, then **skip the
  low-conviction trades**. This is the canonical meta-labeling setup (AFML ch. 3). It was the
  lowest-risk candidate on the register (RESEARCH_CANDIDATES #8) precisely because it needs no new
  data and can only filter an edge we already trust.
- **Method (no-lookahead, rigorous).** Replayed the **deployed config** (train 252 / test 63 /
  top-10 / ADF ≤ −2.86 / 2 bps) on the 195 large-cap daily bars with a trade-emitting backtest,
  collecting **430 pair-trades across 15 OOS windows** with *entry-only* features (|entry-z|, ADF
  stat, half-life, |hedge|, lookback, train-spread vol, side) and label = trade net return > 0.
  Trained a numpy logistic meta-model under **purged, embargoed, expanding-window CV** (5 folds over
  the back 60%, 5-day embargo) → out-of-sample P(profit) per trade. Reconstructed the basket
  faithfully (dropping a trade = that pair sits flat; same per-window denominator) for a τ sweep,
  plus a **shuffle-label placebo**.
- **Result — no usable signal.** The replay's baseline basket Sharpe reproduced the validated
  **0.831** bit-for-bit (the harness is faithful). But the meta-model's **out-of-sample AUC was
  0.379 — *below* 0.5, i.e. anti-predictive — and *below* its own shuffle-label placebo (0.46).**
  The secondary model has **zero out-of-sample skill** at separating winning from losing pair-trades
  on these features.
- **The τ-sweep trap (why the "Sharpe 1.31" is a mirage).** Filtering at τ=0.60 *appeared* to lift
  the basket to Sharpe 1.31 — but it did so by **dropping 185 of 258 eval trades** to leave just 73,
  and since AUC < 0.5 the model is **anti-predictive**, so it cannot be choosing the right ones. The
  apparent lift is a **trade-count-reduction / luck artifact**, not skill: with a model worse than a
  coin, any Sharpe change is noise from slashing exposure. The verdict gate (AUC > 0.55 **and** beats
  placebo **and** retains ≥40% of trades) correctly rejects it.
- **Verdict.** ❌ **REJECT — honest null confirmed.** Meta-labeling adds no out-of-sample
  information here; the deployed basket is already near the information content its own signals carry.
  This is a *clean* negative: the baseline matched 0.831 exactly, the placebo bracketed the result,
  and the one number that looked good (τ=0.60) was unmasked as a count-reduction artifact by the
  AUC. The pairs basket stays deployed **as-is** — there is no conviction-filter to bolt on.
  (`scripts/test_meta_labeling.py` → `data/meta_labeling_results.json`.)
