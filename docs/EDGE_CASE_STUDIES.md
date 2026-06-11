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
| 1 | **Cointegration-pairs market-neutral basket** | Market-neutral | In-sample Sharpe 1.78 → **OOS 0.54**, −3% DD | ✅ **THE survivor** (modest, real) |
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
| 14 | **PEAD** (post-earnings drift, L/S) | Market-neutral event | Dollar-neutral Sharpe 0.66–0.82; short leg carries it | 🟡 **Encouraging, unvalidated** (~1yr) |
| 15 | Seasonality (turn-of-month, pre-FOMC) | Event-clock overlay | Standalone Sharpe 0.24–0.34, exposure 3–34% | ⚠️ Weak alone; ✅ uncorrelated leg |
| 16 | **Portfolio combination** (inverse-vol blend) | Allocation | 5 legs avg \|corr\| 0.05; combined ~0.87 ≈ null | ⚙️ Method works; edge-supply-limited |

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
- **Result.** Dollar-neutral Sharpe **0.66** (and **0.82** at a stricter ±3% threshold),
  −8% maxDD. The **short leg carries it (0.50) vs the long leg (0.32)** — exactly as theory
  predicts (the long leg is mostly beta; neutral alpha lives in the short).
- **Verdict.** 🟡 **The first new candidate that didn't immediately die — but NOT validated.**
  The honest caveat is decisive: in-sample Sharpe is **0.00** across all configs because the
  free data only covers ~1 year, so positions only populate the back half — the "OOS 1.21" is
  *not* a clean walk-forward, just the active window. Single regime, weak power, paper-shorting
  realism unmodeled. **To judge it properly needs multi-year history (a free Finnhub key).**
- **Next step.** Add `FINNHUB_API_KEY`, pull 5+ years, re-run with a real walk-forward + DSR.

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
  daily/annual return. Tested on 5 *real* legs: pairs basket (MN), rsi-mr (beta), cross-
  sectional (MN), turn-of-month, pre-FOMC.
- **Result.** The legs are genuinely uncorrelated (avg |off-diagonal corr| **0.05**), and the
  inverse-vol blend beats/ties the equal-weight null (~**0.87**). BUT the combined Sharpe sits
  *below* the best single leg, because four of five legs are weak (0.0–0.4): **combining one
  good leg with weak diversifiers dilutes, it doesn't lift.** The diversification formula only
  delivers when you have *several genuinely-good* uncorrelated edges — which we don't.
- **Honest ROI translation.** At the achieved combined Sharpe (~0.87) and an 8% vol target:
  ~5–9% / year ≈ **~2 bps/day expected, under ~40 bps/day of noise (noise ≈ 18× the edge).**
  The edge is *invisible* day-to-day. **"X% per day" targets are noise-mining** — the right
  scoreboard is combined OOS Sharpe (deflated for trial count via DSR) and max drawdown.
- **Verdict.** ⚙️ **The method is real and is the single biggest lever** — but its output is
  capped by the *supply of good uncorrelated edges*. The bottleneck is finding more real
  edges (e.g. validating PEAD), not the allocator.

## Methodology upgrade — Deflated Sharpe Ratio

Given how many strategies this project has tried (~34 in the registry + the dozen edge
families here), naive p-values overstate significance. The harness now includes the
**Probabilistic** and **Deflated Sharpe Ratio** (Bailey & López de Prado): the DSR tests a
Sharpe against the *expected maximum* Sharpe over the number of trials, so it accounts for
selection bias. Use DSR > 0.95 — not a raw p<0.05 — as the real significance bar going forward.

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
and a portfolio combiner — the **84-symbol cointegrated-pairs market-neutral basket (OOS
Sharpe ~0.5)** remains the only fully-validated edge, with **PEAD the one encouraging new
lead** (dollar-neutral Sharpe ~0.7 on ~1yr, short-leg-carried) that's worth multi-year data to
confirm. The combiner is ready to stack edges the moment a second validated one exists.
