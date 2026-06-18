# Alpca — State of the Program (the package)

*The single-page truth, current as of 2026-06-18. For depth: `FINDINGS.md` (executive synthesis),
`EDGE_CASE_STUDIES.md` (all 55 cases), `SECOND_LEG_HUNT.md` (the 2nd-leg narrative), `SYSTEM_MAP.md`
(platform + deployed portfolio), `AI_RESEARCH_LOOP.md`, `PIPELINE_AND_AUTOMATION.md`. This file is the
honest scoreboard + what we'd hand to a skeptical reviewer.*

## What this project is

An **honest-evaluation platform** for systematic trading on Alpaca. Its deliverable is not a pile of
"edges" — it is a harness whose job is to **reject** strategies that don't generalize, and to catch its
*own* overselling. Across **55 distinct edge experiments** (price, microstructure, event-driven,
positioning, fundamental — run by hand and by an AI research loop, then adversarially self-audited), the
honest result is **exactly one thin, regime-dependent edge**, with everything else rejected on a
*different* honest gate. That is the point, not a disappointment: real edges are scarce, and the machine
that says so cheaply is the asset.

## The one validated, deployed edge

**Cointegrated-pairs market-neutral basket.** Walk-forward (re-screen, trade held-out, roll), concentrated
**top-10 + 5% ADF screen**, **train=252 / test≈42–63** cadence. Honest figures:
- Walk-forward Sharpe **~0.83 in-sample**, but **regime-dependent** — first-half 0.27 (CI straddles zero),
  second-half 1.39; 2022 was −0.9. **Forward-honest expectation ~0.3–0.5.**
- **−4 to −10% drawdown** in the robust cadence basin.
- **Survivorship-stamped** (Case 46): re-run on a point-in-time universe (survivors + delistings via a
  delisting-aware walk-forward) it holds +0.83 → +0.94 — *not* an artifact of dropping dead names.
- **Cadence-hardened** (Case 54): the live deploy had been screening on `train=378`, which the cadence map
  proved *inverts* the edge (WF −1.42); fixed to the validated `train=252`. The edge is robust to selection
  breadth (top_n 6/10/15) but fragile to the screen window — chosen now on evidence, not the default.
- Deployed on a **shadow forward paper-track** (`com.alpca.forwardtrack`, daily) — the live OOS curve is
  what ultimately adjudicates.

Sized to Kelly, a ~0.3–0.5 market-neutral sleeve with a shallow drawdown is a genuinely useful core. The
lever for *more* dollars/day is **additional uncorrelated surviving legs** — which remain unfound.

## What was rejected, and the gate that caught each

| Candidate | Looked like | Killed by |
|---|---|---|
| Single-asset trend/breakout/MR, x-sec & TS momentum | beats the tape | **beta** — never beat buy-and-hold |
| The factor zoo on large-caps (asset-growth, ROA, issuance, gross-profit, MAX, idio-vol…) | documented premia | **absent on liquid large-caps** |
| Mid-cap **value** | 0.21, generalized | **survivorship artifact** (Case 43: +0.35 → −0.45 with delistings) |
| Mid-cap **momentum** | 1.35 with delistings | **cherry-pick** (Case 45: representative PIT → ~0.4) + **short-side-only/borrow-gated** |
| **Short-vol / VRP** | combined 1.08, ρ≈0 | **2022–23 vol-crush artifact** (Case 52: lift −0.27 since 2024) |
| Cross-sectional **seasonality** | +0.35, uncorrelated | **regime-unstable**; lift was a partial-2026 artifact |
| Cross-asset **trend / managed-futures** | crisis-alpha hedge | **timing illusory** (loses to B&H + long-only) |
| Short-horizon **reversal** | 0.74, passed the leg gate | **survivorship artifact** (Case 53: +0.69 → −0.40) |
| **PEAD / EAR-PEAD** (×3) | 0.6–0.7, passed the leg gate | **fresh-symbol holdout** (Case 55: −0.22/−0.64) |
| Market-making / HFT / crypto / news | various | **infeasible on the venue / data** |

## The hard-won lessons (the real IP)

1. **Only out-of-universe (fresh-symbol) + out-of-regime (held-out years) holdouts adjudicate.** In-sample
   fit, a positive time-split, per-year stability, a high Deflated Sharpe, subset resampling, *and* a
   five-check leg gate can **all pass on an edge that is still overfit.** The fresh-symbol holdout has now
   caught ~7 such candidates.
2. **The leg gate is necessary but NOT sufficient.** Reversal and EAR-PEAD cleared every leg-gate check
   (positive, uncorrelated, lifts, robust LOO, partial-year-safe) *and* the 2024+ slice — and still died
   on the survivorship/fresh-symbol bar, which sits **upstream**.
3. **Survivorship bias can flip which signal is real — and a careless "fix" re-biases the other way.** Only
   the *full, outcome-blind* point-in-time universe (all delistings, selected on entry-date traits) is honest.
4. **A combiner number is only honest if BOTH legs are OOS and measured over the SAME window** (the in-sample
   pairs + 2021-included momentum that manufactured a fake 1.08 lift).
5. **Costs and borrow are destiny**; the alpha is often on the side you can't trade (momentum's short leg).
6. **Turn the adversarial lens on your OWN claims.** A 9-agent self-audit (Case 52) caught the platform
   overselling — pairs "0.83" → forward ~0.3–0.5; short-vol "diversifier" → 2022–23 artifact.

## Reusable controls banked (the toolkit)

Buy-and-hold benchmark · walk-forward (re-fit/trade/roll) · shuffle placebo · adversarial friction
(adverse-selection borrow) · regime stability · no-lookahead trailing estimates · honest-trial Deflated
Sharpe · **full outcome-blind point-in-time universe** (Alpaca inactive-assets + SIP) ·
**`delisting_aware_walkforward`** (dated OOS returns) · **reverse-split-artifact winsorization** ·
**the second-leg gate** (`leg_gate.py` — positive/uncorrelated/lifts/robust-LOO/partial-year-safe) ·
**partial-year split on any combiner lift** · the canonical deployed-book backtest (caught a real bug).

## Honest profit-per-day reality

At any honest Sharpe (0.3–1.0), expected daily return is a few bps under ~15–25× that in daily noise.
Per-day ROI targeting is noise-mining. The only levers are **(a) more genuinely-uncorrelated surviving
legs** and **(b) Kelly-scaled sizing** — never trading the one thin edge harder.

## Direction

Keep hunting *generalizing* edges, each cleared through the survivorship-PIT + fresh-symbol + out-of-regime
bar (the leg gate alone is not enough). Open threads: a **Finnhub earnings source** to unblock a
*full-breadth* PEAD verdict (the 40-name result is suggestive, not definitive); a genuine
**long-vol/convexity hedge** *if* short-vol is ever revived; **defensive low-beta** (re-queued). The
binding constraint is unchanged: **a leg positive over 2022→, uncorrelated with pairs, and robust across
years.** None found yet — and the platform's value is that it tells us so.
