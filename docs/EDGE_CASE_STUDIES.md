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
| 14 | **PEAD** (post-earnings drift, L/S) | Market-neutral event | Flat-borrow DSR 0.92, but **DSR 0.58 under adverse-selection borrow**; short leg −0.47 standalone | 🟡 **Downgraded** — long leg is beta, short leg fails realistic shorting frictions (24/195 symbols; revisit at full breadth) |
| 15 | Seasonality (turn-of-month, pre-FOMC) | Event-clock overlay | Standalone Sharpe 0.24–0.34, exposure 3–34% | ⚠️ Weak alone; ✅ uncorrelated leg |
| 16 | **Portfolio combination** (inverse-vol blend) | Allocation | 6 legs avg \|corr\| 0.04; **EAR-PEAD leg lifted combined 0.83 → 0.99** | ⚙️ Method works; edge-supply bottleneck easing |
| 17 | **Overnight→intraday reversal** | Market-neutral event | Gross Sharpe 0.93 / DSR 0.90 (control-confirmed) → **−0.41 at 2bps** (~2×/day turnover) | 🔴 **REAL anomaly, untradeable** — canonical cost-wall case |
| 18 | **EAR-PEAD, index-beta-hedged** | Market-neutral event | Hedged Sharpe **0.67, IS 0.70 ≈ OOS 0.66**, −12% DD, DSR 0.89; cheap SPY short (not single-name) | 🟢 **Strongest earnings result** — tradeable short side; rescues Case 14 |
| 19 | **Lead-lag cross-predictability** | Market-neutral (learned) | Walk-forward real −1.02 ≈ shuffle placebo −1.14 (+0.11); gross only 0.27, dies by 1bp | ❌ **Fitted noise** — fails placebo *and* cost wall |
| 20 | **Gap reversion** (multi-day hold) | Market-neutral event | No gross edge (−0.14 @ 0bps); gap-momentum control *beats* it on large caps | ❌ **Signal failure** — large-cap gaps are informational, not reverting |
| 21 | **Short-interest (borrow-fee) tilt** | Market-neutral positioning | Real Nasdaq SI; anomaly Sharpe 2.67 *after* DTC-scaled borrow, control mirrors −3.2, turnover 0.01/day | 🟡 **Strong lead** — right sign, survives borrow, low turnover; but only ~1yr (needs FINRA depth) |

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
- **➡️ The edge-supply bottleneck, eased (the EAR-PEAD update).** The original 5-leg blend was
  edge-starved — one good leg (pairs) plus weak diversifiers, combined Sharpe **0.83**, *below*
  the best single leg. Adding the **EAR-PEAD beta-hedged leg (Case 18)** — the first genuinely
  *new* uncorrelated edge (corr **0.04**) with a real standalone Sharpe — lifted the combined
  inverse-vol Sharpe **0.83 → 0.99 (+0.16)**, measured cleanly with/without the leg. This is the
  combiner doing exactly what the math promises *once you actually feed it a second real edge* —
  the first time a new leg moved the combined number rather than diluting it. (The combined still
  trails the rsi-mr *beta* leg's raw 1.18, but that leg is pure market exposure; the blend is
  near-market-neutral with a far better drawdown profile — a different, more durable risk object.)
- **Honest ROI translation.** At the achieved combined Sharpe (~0.99) and a 6% vol target:
  ~6% / year ≈ **~2.3 bps/day expected, under ~36 bps/day of noise (noise ≈ 16× the edge).**
  The edge is *invisible* day-to-day. **"X% per day" targets are noise-mining** — the right
  scoreboard is combined OOS Sharpe (deflated for trial count via DSR) and max drawdown.
- **Verdict.** ⚙️ **The method is real and is the single biggest lever** — and the bottleneck
  (edge supply) is now *easing*: EAR-PEAD became the second real uncorrelated leg and lifted the
  blend +0.16. The path to higher combined Sharpe (and thus higher honest profit-per-day) is
  **more legs like it** — the scout's statarb borrow-fee/order-flow signals and the lead-lag
  family are the next veins to mine — not faster trading of any single one.

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

## Case 18 — EAR-PEAD, beta-hedged with a cheap index short 🟢 (strongest earnings result)

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
- **Verdict.** 🟢 **A real, tradeable, market-neutral earnings alpha — the rescue of PEAD.**
  Caveats kept honest: only 40 symbols and one (bull-ish) 5-yr regime — but the beta hedge removes
  the bull tailwind and it *still* survives OOS; the hedge ratio is full-sample (mild optimism).
  DSR 0.89 sits just under the 0.95 bar — the `avearnings` job filling the universe to 195 is the
  direct route over it. **Next:** confirm at full breadth, then add as the combiner's 2nd leg.

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

## Case 21 — Short-interest (borrow-fee) tilt 🟡 (strong lead, power-limited to 1 year)

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
- **Result (preliminary, ~40 of 195 symbols cached, ~1-yr / 24 obs each).**

  | variant | Sharpe | within-yr OOS | turn/day |
  |---|---|---|---|
  | anomaly (short high-DTC), no borrow | ~3.2 | +5.8 | 0.010 |
  | **control (chase shorts)** | **−3.2** | −5.9 | 0.010 |
  | anomaly + DTC-scaled borrow (the crux) | **2.67** | +5.3 | 0.010 |

  Right sign (the control is a near-perfect mirror), **survives its own DTC-scaled borrow**
  (Sharpe 2.67), low turnover, and statistically significant *within the year* (PSR 0.99; DSR 0.98
  under a clean same-direction deflation). This is the **only scout-#1 signal that clears the bar
  the others failed** — it is not a cost-wall casualty (Cases 17/20) nor a placebo failure (Case 19).
- **The binding caveat (why 🟡, not 🟢).** It is **one year, one regime.** DSR/PSR account for
  sample length and trial count but **not regime risk**, and the free Nasdaq feed gives only ~1 yr
  (24 settlement points) — there is no true multi-year out-of-sample. A Sharpe of ~2.7 over a single
  year is exactly the kind of number that can be regime-specific. So this is a **strong lead, not a
  validated edge.**
- **Verdict.** 🟡 **The most promising new candidate of the session — but unvalidated on depth.**
  Next step is concrete: pull **multi-year FINRA short-interest history** (the API is reachable) to
  get a real walk-forward across regimes; if it holds, this becomes a genuine third leg — and being
  positioning-driven (not price), it should be uncorrelated to both the pairs basket and EAR-PEAD.
  Until then it stays a lead, sized at zero. (The universe is still filling toward 195; the ~1-yr
  depth, not breadth, is the limit.)

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

**The constructive flip-side (Case 18):** the same discipline that *killed* surprise-PEAD's short
leg also *found the fix*. Switching to a price-only **EAR** signal and hedging beta with a **cheap
index short** (instead of the borrow-fragile single-name short) yields a market-neutral earnings
sleeve that survives honestly — **Sharpe 0.67, IS 0.70 ≈ OOS 0.66, −12% DD, DSR 0.89** — comparable
to the pairs basket, uncorrelated to it, and with a *tradeable* short side. So the validated-edge
count is now **the pairs basket plus a strong, near-validated EAR-PEAD second leg** (just under the
0.95 DSR bar, pending the universe filling to 195). That second uncorrelated leg is exactly what the
combiner (Case 16) was edge-starved for — and the honest path to higher *profit-per-day* runs
through **more such legs sized to Kelly**, not through trading any single one faster.
