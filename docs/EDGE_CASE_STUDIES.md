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

**Bottom line:** after equities (all families), crypto (daily + hourly), market-making, two
sizing/factor generalizations, a CTA edge, an alt-data probe, and a funding signal — the
**84-symbol cointegrated-pairs market-neutral basket (OOS Sharpe ~0.5)** remains the only
edge that has survived honest, out-of-sample testing.
