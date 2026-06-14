# Alpca — Findings Synthesis

*The high-altitude "what we learned." For the per-experiment detail see `EDGE_CASE_STUDIES.md` (35
cases), the loop see `AI_RESEARCH_LOOP.md`, the platform see `SYSTEM_MAP.md`. This is the executive
summary of the honest-evaluation program: the scoreboard, the hard-won lessons, and the direction.*

---

## Executive summary

Across **44 distinct edge experiments** — price, market-microstructure, event-driven, positioning, and
fundamental, run both by hand and by an AI research loop — **one edge is validated and deployed** (a
cointegrated-pairs market-neutral basket) **and a second-edge candidate has emerged on mid-caps —
reframed by the survivorship test from value to MOMENTUM.** The hunt off large-caps found value and
momentum both *appearing* to generalize on mid-caps and uncorrelated with the pairs basket (ρ = −0.03,
gate #1 passed) — but the **survivorship-bias point-in-time re-test (Case 43) flipped it:** putting the
delisted value-traps back in **collapsed value (0.35 → −0.45) and lifted vol-managed momentum (0.39 →
1.35)**. Value *buys* the dying cheap names; momentum *shorts* them. So the real survivorship-robust
mid-cap edge is **momentum**, pending an adverse-selection-borrow model (its strength is from shorting
names going to zero) and a forward paper-track. Re-measured and *corrected* this
session, its honest walk-forward Sharpe is **~0.83** at the concentrated **top-10 + 5% ADF screen**
(−4% drawdown) — far better than the misleading "0.29" of an over-diversified top-24 basket that
diluted the edge into weak pairs. It is deployed on a shadow forward paper-track. **Until this session,
no other candidate survived** — but moving off large-caps (where the literature says the premia don't
live) and onto a fresh ~289-name mid-cap universe finally surfaced a second one: a **value +
vol-managed-momentum** blend that generalizes out-of-universe and is regime-robust (5/6 years), pending
two confirmations before deployment.

That is not a failure of the project — it *is* the project. The deliverable is a harness whose job is
to **reject** edges that don't generalize, and it has done so relentlessly, including against several
candidates that looked excellent in-sample. Sized to Kelly, a ~0.83 market-neutral sleeve with −4% DD
is a genuinely useful core; the lever for *more* is additional uncorrelated surviving legs — and the
mid-cap blend is the first credible one.

---

## The scoreboard (44 experiments, by outcome)

**✅ Validated & deployed (1):**
- **Cointegrated-pairs basket** — market-neutral, walk-forward (re-screen each quarter, trade the
  next). **WF ~0.83 at the concentrated top-10 + 5% ADF screen** (the "0.29" was an over-diversified
  top-24 — diluting into weak pairs halved the edge), −4% drawdown. Deployed on a
  forward paper-track to let the *live* OOS curve adjudicate.

**🟡 Second-edge candidate — momentum, but short-side-only + borrow-gated (Cases 41–44):**
- **Mid-cap vol-managed momentum.** The hunt found value AND momentum *appearing* to generalize on
  mid-caps (Cases 41–42), uncorrelated with the pairs basket (**ρ = −0.03**, gate #1). The **survivorship
  point-in-time re-test (Case 43) flipped it**: adding back ~50 delisted value-traps **collapsed value
  0.35 → −0.45** (value *buys* the dying names) and **lifted momentum 0.39 → 1.35** (momentum *shorts*
  them). **Case 44 then stress-tested momentum and found the honest limits:** (a) under **adverse-selection
  borrow + no-locate drops + dollar-neutral + artifact-winsorization** it still holds **~1.1** on the
  point-in-time universe (real alpha, not beta) — but (b) the **long/SPY-index-hedged, borrow-FREE form
  has NO edge** (0.08 survivor, −0.12 PIT): the alpha is **entirely short-side**, so it **can't escape the
  borrow wall**; and (c) the magnitude is **bracketed 0.3–1.1** — my +50 delistings are *famous* failures
  that over-feed the short leg, so the true number needs a representative point-in-time universe or a
  forward track. **Net: a genuine but messy short-side momentum anomaly — promising enough to
  forward-track, not clean enough to deploy. Still one validated+deployed edge (pairs).**

**⚠️ Generalizes but too weak to deploy (2):**
- **Value composite (E/P + FCF/P + B/P)** — the *first* fundamental whose fresh-symbol holdout stayed
  *positive* (it generalizes to unseen symbols), but Sharpe ~0.14 and heavily regime-timed (strong in
  the 2022 value rotation, weak in growth-led 2026). Real premium, too thin alone; on the combiner bench.
- **Mid-cap value + light momentum (Cases 38, 40)** — the *strongest generalizing fundamental in the
  program.* Value is **stronger in mid-caps (0.21) than large-caps (0.11)** and still generalizes
  (holdout +0.14→**+0.44 at full ~289-name breadth** — generalization *improving* with breadth, the
  opposite of overfit); a **light** momentum tilt (w=0.25, the AMP combo) lifts it to **0.36–0.39** —
  and unlike large-caps, the light blend *improves* generalization instead of breaking it (w≥0.5 breaks
  → a real sweet spot). Size-tilt is **non-monotonic**: it **inverts in small-caps (−0.26, Case 39)** —
  the value-trap + 2021–26 small-cap bear — so mid-cap is a genuine local sweet spot, not a ladder.

- **Momentum family on mid-caps (the survivor-only view, Case 41)** — *residual* and *vol-managed*
  momentum, both rejected on large-caps (Cases 31–32), *appear* to generalize on mid-caps (holdout
  +0.25, vol-managed the most regime-robust factor at 4/6 years). But the survivorship + borrow stress
  (Cases 43–44) showed the apparent strength is a short-side-only, borrow-gated anomaly — see the 🟡
  candidate above, which is the honest, post-stress version of this line.

**❌ Rejected — overfit (caught out-of-sample) (3):**
- **EAR-PEAD beta-hedged** — 0.68 / 6-of-6 years / DSR 0.89 in-universe, but fresh-symbol holdout −0.52.
- **Short-interest / borrow tilt** — DSR 0.98 on 1 year of data, but −0.42 across 9 years of FINRA.
- **Accruals (Sloan)** — +5/6 years, cost-free in-universe, but fresh-symbol holdout −0.47.

**❌ Rejected — cost wall (real signal, untradeable) (3):**
- **Overnight→intraday reversal** (gross Sharpe 0.93 → −0.41 at 2 bps), **gap reversion**,
  **lead-lag** (also failed its shuffle placebo). High turnover dies to spread/impact.

**❌ Rejected — beta / leverage artifact (3 families):**
- Single-asset directional (trend/breakout/MR), cross-sectional & time-series momentum — capture a
  fraction of bull-market beta with less drawdown; never beat buy-and-hold.
- **Betting-Against-Beta / low-vol** — unlevered dollar-neutral version is short-beta in a bull
  (−0.63 / −0.95); the risk-adjusted premium needs leverage we lack.

**❌ Rejected — the factor zoo on large caps (Cases 26–35):**
- Nine documented cross-sectional factors — asset growth, net issuance, ROA, MAX/lottery, idio-vol,
  residual & vol-managed momentum, short-interest *change*, **gross profitability** — all rejected on
  the 195-name large-cap 2021–26 universe (thin/absent; these premia live in small/mid-caps + longer
  histories). Financials-excluded accruals improved in-sample (+0.70/6-of-6) but its fresh-symbol
  holdout stayed −0.51 — the sector-rescue refuted by measurement.

**❌ Rejected — value refinements that improve in-sample but break generalization (Cases 36–37):**
- **Sector-neutral value** — demeaning the value composite within sector *raised* the in-universe
  Sharpe (0.11 → 0.34) but **collapsed the fresh-symbol holdout from +0.70 to −0.64**. The sector bet
  was carrying the part that generalized; neutralizing fit main-universe idiosyncrasies. Reject.
- **Value + momentum combo (AMP)** — blending 12-2 momentum into value improved *every* in-universe
  metric (main 0.11 → 0.51, OOS time-split −0.52 → +0.76, DSR → 0.62) but the fresh-symbol holdout went
  **negative for any momentum weight** (−0.16 to −0.29). Momentum overfits the specific names and doesn't
  transfer — a clean proof that **out-of-sample-in-time ≠ out-of-sample-in-symbols.** Reject.

**❌ Rejected — infeasible on the venue / data (several):**
- Market-making / HFT / microstructure (Alpaca ~1.2 s fills, no L2, no rebates), crypto (daily +
  hourly), news/alt-data (free APIs too thin), 52-week-high momentum (inverts on our universe).

**Combiner of survivors:** pairs (0.83) + value (0.11) are uncorrelated (|corr| 0.02), but an
inverse-vol blend *dilutes* to 0.53 — value is too thin to add and equal-risk weighting over-weights
it. **The deployable portfolio is the pairs basket alone.** The binding constraint stays *edge
supply* — a second genuinely-good uncorrelated leg.

---

## The hard-won lessons (paid for repeatedly)

1. **Only out-of-universe + out-of-regime holdouts adjudicate.** In-sample fit, a positive 70/30
   split, per-year regime stability, a high Deflated Sharpe, *and* subset resampling can **all pass on
   an edge that is still overfit.** Five different candidates (EAR-PEAD, the SI tilt, accruals,
   sector-neutral value, value+momentum) proved this — each died only on a *fresh-symbol* (new names)
   or *multi-regime* (new years) holdout. **Out-of-sample-in-TIME ≠ out-of-sample-in-SYMBOLS:** the
   value+momentum combo (Case 37) posted a +0.76 chronological OOS split while its disjoint-symbol
   holdout was −0.23. The fresh-symbol holdout is the binding bar for "validated"; a passing time-split
   is necessary, not sufficient.

2. **Costs are destiny.** Every high-turnover variant died to spread + impact + borrow. A *real* gross
   signal (overnight reversal, Sharpe 0.93) is worthless if it turns the book over ~2×/day. Low
   turnover is a prerequisite, not a detail.

3. **Beta is not alpha — and a bull market hides the difference.** Long-biased single-asset strategies
   look great in a 2021–2026 backtest and beat nothing once benchmarked against buy-and-hold. Always
   benchmark vs B&H; judge market-neutral sleeves on their own Sharpe.

4. **Short windows lie.** A Sharpe of 2+ over one year (the SI tilt on 1 yr of Nasdaq data) was a
   lucky-regime artifact that nine years of FINRA data demolished. Multi-year, multi-regime data is
   non-negotiable — which is why **free, no-quota, multi-year sources (SEC EDGAR, FINRA)** are the
   preferred foundations over quota-limited feeds.

5. **The realistic friction model is the one that matters.** Modeling *adverse-selection* borrow (the
   crowded shorts you most want are exactly the ones that go special/no-locate) turned surprise-PEAD
   from a "DSR 0.92 candidate" into a reject. The optimistic friction model had waved it through.

5b. **Survivorship bias doesn't just dock a Sharpe — it can flip which signal is real.** Testing the
   mid-cap edge on a today-exists universe inflated VALUE (it buys the cheap names; the cheapest-that-
   went-bankrupt were missing) and *suppressed* MOMENTUM (it shorts those same falling names; its best
   trades were missing). Putting the delisted value-traps back (Alpaca serves delisted history) sent
   value 0.35 → −0.45 and momentum 0.39 → 1.35 (Case 43). Point-in-time universes including delistings
   are mandatory before believing any cross-sectional equity edge — and the side that *benefits* from the
   dead names (a short leg) then runs into the borrow wall (lesson 5).

6. **Real edges are scarce, and that's the truth — not a tuning problem.** After 35 experiments, one
   survivor. The constraint on profit is **edge supply**, and more search (now AI-accelerated)
   does not lower the bar — it just lets us reject faster.

---

## The anti-overfitting toolkit (each killed or saved a real candidate)

Buy-and-hold benchmark · walk-forward (re-fit, trade held-out, roll) · **shuffle placebo** (randomize
any *learned* structure; it must beat its own placebo) · **adversarial friction models** (borrow /
slippage applied in the direction that hurts) · **regime stability** (per-calendar-year Sharpe) ·
no-lookahead trailing estimates · honest trial-count Deflated Sharpe · **multi-regime data** · and the
decisive **fresh-symbol holdout** (run frozen params on symbols never used in development).

---

## The AI research loop (what it adds)

A self-driving hunt — regime detector → an LLM proposes a strategy *config* from a **constrained space
of known, tested templates** (never arbitrary code) → the harness runs it on the main + a disjoint
fresh universe → a **deterministic falsification gate (veto) + live model verdict** decides. Both model
tiers run on existing credentials (Haiku via the Claude Code OAuth subscription; OpenAI via a reused
key in the gitignored `.env`); it's launchd-scheduled. **It has shipped zero edges — correctly** —
independently reproducing the same rejections (e.g., the accruals fresh-holdout failure) we found by
hand. **The model proposes; only the data validates.** Its value is multiplying search throughput
without ever lowering the bar.

---

## Profit-per-day reality

At any honest Sharpe (0.3–1.0), expected daily return is a few bps, buried under ~15–25× that in daily
noise. The lever for more dollars/day is **(a) more genuinely-uncorrelated surviving edges and
(b) Kelly-scaled sizing — not trading frequency.** Today: one real edge (pairs, WF ~0.83, −4% DD),
sized to Kelly, is a useful core; more dollars/day needs *more uncorrelated surviving legs*. The
combiner math works but is **edge-supply-limited**.

---

## Direction

Keep hunting *generalizing* edges (the only lever), with the loop running the brutal tests
automatically on each. The most promising untested veins, all on free EDGAR/FINRA data and gated by
the fresh-symbol + out-of-regime bar:
- **Sector-neutral value** and a **small-cap value tilt** (where the premium is historically stronger).
- **Financials-excluded, sector-neutral accruals** on a broad fresh universe.
- ~~A real ADF cointegration screen to lift the pairs basket~~ **— DONE: top-10 + 5% ADF screen
  lifted the validated edge to WF ~0.83 (from a diluted 0.29); deployed.**
- Letting the AI loop run weekly and **reviewing only the survivors** — expecting few, by design.

**Bottom line:** the platform tells the truth. One real, modest, deployed edge; a disciplined,
now-automated machine for finding more; and an honest refusal to mistake a backtest for money.
