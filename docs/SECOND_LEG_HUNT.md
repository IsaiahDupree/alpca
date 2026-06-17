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
| 49 | **Short-vol / VRP** | uncorrelated, 6/6 yrs, lifts | 🟡→❌ — *looked* like it cleared (combined 1.08) but the audit (Case 52) showed the lift is a 2022–23 vol-crush artifact; **−0.27 since 2024** → not a current diversifier |
| 52 | **Adversarial self-audit** | refute our own claims | ⚠️ caught the OVERSELLING: pairs is forward ~0.3–0.5 (not 0.83), short-vol drags since 2024, 5 code bugs fixed; **one thin regime-dependent edge** stands |

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

**One deployable core edge (pairs) — and, after four rejections plus a self-audit, STILL no validated
second leg.** The binding constraint:

> **A leg that is positive over 2022→ AND uncorrelated with the pairs basket AND robust across years.**

Value, momentum, seasonality, and cross-asset trend each failed it on a different axis. **Short-vol
*looked* like it cleared it** (Case 49: ρ=0.04, "6/6 years", combined 1.08) — but the **adversarial audit
(Case 52) refuted the diversification**: the entire lift is the 2022–23 vol-crush, and **since 2024 the
book lift is −0.27** (short-vol drags). The "6/6 years" was the standalone calendar Sharpe, not the book
lift; ρ≈0 is real and stress-stable but "uncorrelated ≠ hedged" (pairs is dead weight in a vol spike).
So short-vol is **not a current diversifier** — it stays on the forward track at a tiny 8% tail-cap only
to observe whether the VRP regime returns. The audit also corrected the *core*: pairs WF 0.83 is
regime-dependent (first-half 0.27, 2022 −0.9); forward-honest **~0.3–0.5**.

The honest direction: a **genuinely uncorrelated axis** — ideally a **long-volatility / convexity** sleeve
that actually *hedges* short-vol's crash (the correlation analysis proved the current book's tail is
undiversified), plus a market-neutral event signal — each cleared through the out-of-universe +
out-of-regime bar, not an in-sample lift.

*Bottom line: the platform tells the truth — including about itself. It rejected four plausible second
legs, then turned the same adversarial lens on its own "validated" claims and caught the overselling
before a real reviewer (or a live drawdown) could.*
