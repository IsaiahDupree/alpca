# Alpca — The Second-Leg Hunt (Cases 41–48): an honest field report

*A standalone narrative of the most rigorous stretch of the honest-evaluation program: the search for a
second, uncorrelated, deployable edge to pair with the one validated edge (the cointegrated-pairs
basket). Every candidate here looked promising and was rejected by a **different** honest gate. The value
is the gauntlet, not a new edge — we still have exactly one. For the full scoreboard see `FINDINGS.md`;
for per-experiment detail see `EDGE_CASE_STUDIES.md`.*

---

## The one validated edge (the baseline we're trying to beat)

**Cointegrated-pairs market-neutral basket** — walk-forward (re-screen each quarter, trade the next),
concentrated **top-10 + 5% ADF screen**, **WF Sharpe ~0.83**, −4% drawdown. Deployed on a shadow forward
paper-track. This session it was also **survivorship-stamped** (Case 46): re-run on a point-in-time
large-cap universe (195 survivors + 32 representative delistings) via a new delisting-aware walk-forward,
it holds at **+0.83 → +0.93**. It is *not* survivorship-inflated.

The binding constraint on making *more* money is not leverage or frequency — it is **edge supply**: a
second genuinely-uncorrelated, robust, positive edge to combine. The combiner math only pays off with
real uncorrelated legs. This report is the hunt for one.

---

## The candidates, and the gate each one died on

| # | Candidate | Looked like | Died on (the honest gate) |
|---|---|---|---|
| 41 | Factor zoo on mid-caps | momentum family "revives" (holdout +0.25) | — (led to 43–45) |
| 42 | Mid-cap value+momentum, gate #1 | ρ=−0.03 vs pairs, 5/6 yrs, *lift* | used **in-sample pairs** + 2021 — see 47 |
| 43 | Mid-cap **value** | 0.35, generalizes | **survivorship**: +50 bankrupt value-traps → −0.45 |
| 44 | Momentum under realistic borrow | ~1.1 L/S survives borrow | **long/hedge borrow-free form is 0.08** (alpha is short-side-only) |
| 45 | **Representative** point-in-time universe | resolves momentum's magnitude | the 1.35 was a **cherry-pick** of famous failures → true ~0.4 |
| 46 | Survivorship test of the **deployed** pairs edge | (a check, not a candidate) | **PASSED** — pairs robust, 0.83→0.93 |
| 47 | Honest two-sleeve combiner (pairs + momentum) | gate #1 hinted a lift | momentum **negative over 2022→** (carried by 2021) → **dilutes** (0.47<0.83) |
| 48 | Cross-sectional **seasonality** | +0.35 OOS, ρ=+0.06, generalizes | **regime-unstable** (−2025/−2026); "lift" was a **partial-2026 artifact** |
| 49 | **Short-vol / VRP** | uncorrelated, 6/6 yrs, lifts | ✅ **CLEARS** — combined 0.83→1.08, DSR 0.90, DD unchanged at 12% size; asterisk = un-sampled −46% tail |

---

## The five hard-won lessons (each paid for with a dead candidate)

1. **Survivorship bias can flip which signal is real — and a careless fix re-biases the other way.**
   A survivor-only universe inflated *value* (it buys the cheap names; the cheapest-that-went-bankrupt
   were missing) and *suppressed* momentum (it shorts those same names). Adding back ~50 **cherry-picked
   famous bankruptcies** then over-fed momentum's short leg (0.39→1.35) — a bias in the opposite
   direction. Only the **full, outcome-blind point-in-time universe** (all 1,707 Alpaca delistings via the
   inactive-assets API, SIP feed, filtered on 2021 characteristics) gave the honest number (~0.4) — and
   revealed that representative mid-cap delistings are mostly **acquisitions, not bankruptcies**.

2. **Out-of-sample-in-TIME ≠ out-of-sample-in-SYMBOLS.** Value+momentum posted a +0.76 chronological
   split and a −0.23 disjoint-symbol split. The fresh-symbol holdout is the binding test.

3. **A combiner number is only honest if BOTH legs are OOS AND measured over the SAME window.** Gate #1
   showed a pairs+momentum lift only because it used *in-sample* pairs returns and a momentum window that
   included 2021. The rigorous version (walk-forward pairs + same-date momentum) erased it: the momentum
   sleeve is **negative over 2022→**, so the book dilutes.

4. **When a combiner lift appears, split out the most-recent/partial year.** Seasonality's 0.83→0.88
   "lift" vanished — and reversed to a dilution — once the partial 2026 (Jan–Jun) was excluded. A single
   lucky stretch manufactures fake diversification.

5. **Costs and borrow are destiny, and the alpha is often on the side you can't trade.** Momentum's edge
   lives entirely in the *short* leg (shorting losers), which runs straight into the adverse-selection
   borrow wall; the borrow-free long/index-hedged form has no edge.

---

## New controls banked this session (now part of the toolkit)

- **Full outcome-blind point-in-time universe** — via Alpaca's inactive-assets API + SIP feed, filtered
  on entry-date (2021) characteristics, never on outcome. Catches survivorship in *both* directions.
- **`delisting_aware_walkforward`** — pairs walk-forward on a union calendar that lets delisted legs
  trade and close at their last real price (the production `walkforward_pairs` uses a global timestamp
  intersection and structurally excludes delisted names). Now also exposes **dated OOS returns** for
  clean, in-sample-free combiner tests.
- **Reverse-split-artifact winsorization** — near-delisting microcaps carry mis-adjusted reverse-splits
  (WW showed a +152× day) that corrupt both P&L and signal rankings.
- **Partial-year split on any combiner lift** — see lesson 4.
- **`cross_sectional_seasonality_signal`** — a correct, no-lookahead calendar factor (the edge was
  rejected, but the signal is reusable).

---

## Where it stands, and what's next

**One deployable core edge (pairs, ~0.83, survivorship-stamped) — and, after three rejections, a first
real second leg: short-vol / VRP (Case 49).** The binding constraint that finally cracked:

> **A leg that is positive over 2022→ AND uncorrelated with the pairs basket AND robust across years.**

Value, momentum, and seasonality each failed it on a different axis. **Short-volatility clears it** — on
a genuinely different risk axis (selling volatility): ρ=0.04 with pairs, positive 6/6 years, robust
across leave-one-out, and it **lifts the combined book 0.83 → 1.08 (DSR 0.90)** while inverse-vol sizing
holds the combined drawdown at −5.5%. **Its asterisk is non-negotiable and front-and-center:** short-vol
is negatively skewed with an *un-sampled catastrophic tail* (−46% standalone; 2021–2026 had no
volmageddon), so the Sharpe/DSR/DD all understate the true risk — it deploys only at small (~12%) size
with explicit tail management, and a vol spike is the known failure mode. It now joins the forward
paper-track (pairs + momentum + short-vol), each sized to honest conviction.

Veins still untried with this rigor: a **market-neutral event signal** and **index reconstitution**
(data-gated). The job, as always: reject cheaply, suspect any lift, and respect the tail you didn't
sample.

*Bottom line: the platform tells the truth. It rejected three plausible second legs — each of which a
naive backtest would have shipped — and then found a real one whose biggest risk it refuses to hide.*
