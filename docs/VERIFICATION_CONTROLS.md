# Alpca — Verification Controls (the anti-overfitting machinery)

This is the catalog of *controls* the evaluation harness applies to every edge before it is
allowed to be called "validated." It is the most important asset in the repo: the strategies
come and go, but the controls are what make a result trustworthy. The companion graphic is
`docs/edge_funnel.png` — every documented case study classified by the control that killed it,
down to the 2 deployed sleeves.

**The thesis of the whole program:** the *denominator* is the point. Anyone can show you a
backtest with a Sharpe of 2. The question that matters is *how many things did you try, and what
survived an honest battery of out-of-sample, out-of-universe, out-of-regime, and cost-realistic
tests?* On our venue, the answer is: almost nothing.

**The bar for "validated"** (non-negotiable, learned by paying for it twice — see Cases 18 & 21):

> A validated edge clears **out-of-universe** (fresh symbols) AND **out-of-regime** (new calendar
> years), **net of realistic costs**, and is **not a survivorship artifact**. In-sample Sharpe,
> regime-stability, and even a Deflated Sharpe Ratio of 0.98 can ALL pass on an overfit edge —
> only the out-of-universe + out-of-regime + survivorship holdouts catch it.

---

## The controls, ranked by how much work they do

| Control | What it does | What it catches | Cleanest catch |
|---|---|---|---|
| **Fresh-symbol holdout** | Select parameters on one symbol set; report on a **disjoint** set | Edges fit to a specific universe | EAR-PEAD: train-40 **+0.68** → fresh-19 **−0.52** (Case 18) |
| **Survivorship PIT** | Rebuild the universe *as it was*, delisted/bankrupt names included | Edges that exist only by excluding losers | Mid-cap value **flips to momentum** once delistings restored (Case 43) |
| **Cost / borrow wall** | Charge real spread + adverse-selection borrow on every leg | High-turnover & hard-to-short edges | Overnight reversal: gross **0.93** → **−0.41** at 2 bps (Case 17) |
| **Multi-regime / per-year** | Sharpe must be positive in a *majority* of calendar years | Lucky-window artifacts | Short-interest tilt: 1-yr Nasdaq **2.34** → 6-yr FINRA **net −0.42** (Case 21) |
| **Walk-forward / OOS** | Train on in-sample, report only on held-out forward data | Edges that decayed after the fit window | PCA stat-arb: IS **0.99** → OOS **−1.18** (Case 10) |
| **Shuffle placebo** | Re-run on shuffled labels; if real ≈ shuffled, it is noise | Learned-structure mirages | Lead-lag: real **−1.02** ≈ shuffle **−1.14** (Case 19) |
| **DSR / Bonferroni** | Deflate Sharpe for the number of trials searched | Multiple-comparisons winners | Formulaic-alpha zoo: **0/21** pass Bonferroni (Case 36) |
| **Beta decomposition** | Regress returns on the market; separate alpha from beta | "Strategies" that are just dampened exposure | rsi-mr Sharpe 1.18 but β 0.21, never beats B&H return (Case 2) |
| **vs Buy-and-hold** | Directional edges must beat the trivial long baseline | Beta dressed as alpha | Every single-asset directional family (Case 2) |
| **Forward paper-track** | Independent live resolution on Alpaca paper | In-sample self-deception of any kind | The deployed pairs basket is on a live track (Case 1) |
| **Adversarial self-audit** | The harness turns its lens on its own surviving claims | Our *own* overselling | Case 52 — caught us inflating the combiner lift |

---

## Why each control exists (the failure it was built to stop)

### 1. Fresh-symbol holdout — out-of-universe
The single most valuable control we built. Three separate candidates (EAR-PEAD, accruals,
financials-excluded accruals — Cases 18, 23, 35) passed *every in-universe test* — regime
stability, subset resampling, DSR > 0.8 — and then went **negative on a disjoint set of
symbols**. The edge was a property of the *symbols it was fit on*, not the market. In-sample
tests, no matter how many, cannot catch this. Only holding out **new symbols** can.

### 2. Survivorship PIT — point-in-time universe reconstruction
The session's most important methodological result (Case 43). Mid-cap *value* looked like a
generalizing second edge — until we rebuilt the universe with the delisted/bankrupt names that
*were* there at decision time. Value's premium was largely the artifact of silently dropping the
cheap names that went to zero; with them restored, **momentum**, not value, was the real signal.
A backtest on today's surviving tickers is a backtest on the winners.

### 3. Cost / borrow wall
Two distinct walls, both fatal:
- **Spread/impact** kills frequency. Every high-turnover edge (overnight reversal Case 17,
  lead-lag Case 19, gap reversion Case 20, intraday x-sectional Case 3/4) had its gross edge
  fully consumed by ~2 bps/leg at its turnover. *Profit/day = edge × sizing, NOT trade
  frequency.*
- **Adverse-selection borrow** kills short legs. The names you most want to short (worst
  earnings misses, highest short interest) are exactly the ones that go special / no-locate.
  Modeled as a saturating ramp from base → special borrow + a no-locate ceiling, it killed
  PEAD (Case 14), the SI-tilt (Case 21), and mid-cap momentum's short leg (Case 44).

### 4. Multi-regime / per-year stability
A single great year is not an edge. The short-interest tilt posted a Sharpe of **2.34** on
1 year of Nasdaq data (Case 21) — which happened to be the one good stretch. Across 9 years of
multi-regime FINRA data it was **net −0.42, positive in only 3 of 6 years**. The per-year /
multi-regime test is the cleanest proof of *"don't trust short windows."*

### 5. Walk-forward / out-of-sample
The baseline discipline: parameters chosen on in-sample data, performance reported only on
held-out forward data. Catches edges that decayed after their fit window (PCA stat-arb Case 10:
IS 0.99 → OOS −1.18; naive pairs "fixes" Case 5 that each made walk-forward worse).

### 6. Shuffle placebo
For any edge that *learns* structure (lead-lag relationships, cross-predictability), re-run the
exact pipeline on **shuffled labels**. If the "real" result is statistically indistinguishable
from the shuffled one, the structure was fitted noise. Lead-lag (Case 19) failed this *and* the
cost wall — a double kill.

### 7. DSR / Bonferroni — multiple-comparisons deflation
Search 191 formulaic alphas and a few will look great by chance. The Deflated Sharpe Ratio
discounts the observed Sharpe by the number of trials; Bonferroni sets a family-wise threshold.
The famous Alpha101 / Alpha158 / GTJA191 libraries scored **0/21 past Bonferroni** on our venue
(Case 36), with in-sample/OOS winners *inverting* on fresh symbols.

### 8. Beta decomposition & vs-buy-and-hold
Most "winning" directional strategies are risk-reduced **beta**, not alpha. rsi-mr has a Sharpe
of 1.18 (better than B&H's 0.86) but a β of 0.21 and **never beats buy-and-hold's total
return** (Case 2). Regressing on the market separates the two; market-neutral strategies have no
B&H to beat, so the return *is* the alpha.

### 9. Forward paper-track — independent resolution
A live paper-track with independent resolution beats any in-sample number. The deployed
pairs basket runs on a forward Alpaca-paper track precisely so the claim is adjudicated by the
market, not by us.

### 10. Adversarial self-audit
The advocate/skeptic pairing applied to our *own* surviving claims. Case 52 caught the combiner
lift being oversold; every survivor gets a refute pass so we don't manufacture false positives.

---

## What survived all of it

| Sleeve | Class | Honest result | Controls cleared |
|---|---|---|---|
| **Cointegrated-pairs basket** | Market-neutral | Walk-forward Sharpe ~0.83 (top-10 + 5% ADF), −5% DD; **survives survivorship PIT** (Case 46) | OOS, walk-forward, survivorship PIT, per-year, forward paper-track |
| **Short-vol / variance-risk-premium** | Carry | Lifts the book (+0.25), low tail-ρ, **survives a simulated volmageddon** at the 8% cap (Case 50) | OOS, per-year, tail-stress, low cross-leg correlation |

**Deployed book:** pairs ~92% + short-vol ~8% → full-period Sharpe ~0.92, DSR ~0.87, max DD
~−5%, positive in 5 of 6 years. Modest, market-neutral, uncorrelated to beta — and *real*,
because of everything above that it had to survive.

**The honest ceiling:** ~1–2 bps/day at half-Kelly sizing against ~15–25× that in daily noise.
Per-day profit targeting is noise-mining; the edge compounds slowly and only because it is real.

---

*Companion artifacts: `docs/edge_funnel.png` (this catalog, visualized), `docs/strategy_landscape.png`
(the overfit catch), `docs/deployed_results.png` (the live book), `docs/EDGE_CASE_STUDIES.md`
(all 63 cases in full).*
