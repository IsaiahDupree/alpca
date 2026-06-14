# Alpca — Findings Synthesis

*The high-altitude "what we learned." For the per-experiment detail see `EDGE_CASE_STUDIES.md` (35
cases), the loop see `AI_RESEARCH_LOOP.md`, the platform see `SYSTEM_MAP.md`. This is the executive
summary of the honest-evaluation program: the scoreboard, the hard-won lessons, and the direction.*

---

## Executive summary

Across **45 distinct edge experiments** — price, market-microstructure, event-driven, positioning, and
fundamental, run both by hand and by an AI research loop — **one edge is validated and deployed** (a
cointegrated-pairs market-neutral basket) **and a modest second-edge candidate has emerged on mid-caps:
vol-managed momentum (~0.4 Sharpe), uncorrelated with the pairs basket.** The arc that got there is the
methodology in miniature: value and momentum both *appeared* to generalize on mid-caps (ρ = −0.03 vs the
pairs basket, gate #1 passed); a survivorship re-test with ~50 *cherry-picked bankrupt* value-traps
*looked* like it flipped value→momentum and lifted momentum to 1.35 — but rebuilding on the **full,
outcome-blind point-in-time universe** (all 1,707 Alpaca delistings, SIP feed; representative mid-cap
delistings are mostly **acquisitions, not bankruptcies**) **collapsed that 1.1+ to ~0.4** (≈0.43 gross, ≈0.30 after adverse-selection borrow, ≈0.23 as a borrow-free long/index-hedged sleeve). A real
but modest momentum edge — the cherry-picked "fix" had itself been a bias in the opposite direction.
Pending a forward paper-track. Separately, the pairs basket was re-measured and *corrected* this
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

## The scoreboard (45 experiments, by outcome)

**✅ Validated & deployed (1):**
- **Cointegrated-pairs basket** — market-neutral, walk-forward (re-screen each quarter, trade the
  next). **WF ~0.83 at the concentrated top-10 + 5% ADF screen** (the "0.29" was an over-diversified
  top-24 — diluting into weak pairs halved the edge), −4% drawdown. Deployed on a
  forward paper-track to let the *live* OOS curve adjudicate.

**🚀 Now on a forward paper-track (as of this session):** the borrow-free **long/index-hedged momentum
sleeve** (`deploy_momentum_paper.py` + `alpca/live/momentum_portfolio.py`) is wired into the daily
`com.alpca.forwardtrack` job alongside the pairs basket — mark prior book → log sized target →
accumulate a live OOS curve, sized tiny on the honest ~0.23 Sharpe. The two sleeves are uncorrelated
(ρ=−0.03), so this is a genuine two-sleeve forward experiment. The live curve, not the backtest, now
adjudicates — exactly the right next step for a modest, survivorship/borrow-honest candidate.

**🟡 Second-edge candidate — mid-cap momentum, modest (~0.4) and borrow-aware (Cases 41–45):**
- **Mid-cap vol-managed momentum.** Found on mid-caps (Cases 41–42), uncorrelated with the pairs basket
  (**ρ = −0.03**, gate #1). The survivorship re-test (Case 43) *appeared* to flip value→momentum and lift
  it to 1.35 — but **Case 45 resolved that as a cherry-pick artifact:** rebuilt on the **full, outcome-blind
  point-in-time universe** (all 1,707 Alpaca delistings, SIP feed, filtered to mid-cap-in-2021), where
  representative delistings turn out to be **mostly ACQUISITIONS, not bankruptcies** (genuine mid-caps get
  bought, not wiped out). On that honest universe the magnitude **collapses from 1.1 to ~0.4**: borrow-free
  ≈0.43, after adverse-selection borrow ≈0.30, and a mildly positive **borrow-free long/SPY-hedged form
  ≈0.23** (the long leg holds acquired winners into their buyout premium; adding representative delistings
  slightly *hurts* the L/S, confirming no survivorship boost). **Net: a real but MODEST (~0.4)
  momentum edge, uncorrelated with the pairs basket, with a plausible borrow-free deployable form (~0.2) —
  a credible weak second leg for the combiner, pending a forward track.** The arc is itself the lesson: a
  cherry-picked survivorship "fix" was a bias in the opposite direction; only the full outcome-blind PIT
  universe gave the honest number. Still one validated+deployed edge (pairs).

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

5b. **Survivorship bias doesn't just dock a Sharpe — it can flip which signal is real, and a careless
   "fix" re-biases the other way.** A today-exists universe inflated VALUE and suppressed MOMENTUM; adding
   the dead names back sent value 0.35 → −0.45 and momentum 0.39 → 1.35 (Case 43). But that "+50
   delistings" set was *cherry-picked famous bankruptcies*, which over-fed the short leg — itself a
   survivorship bias in the opposite direction. Only the **full, outcome-blind point-in-time universe**
   (all 1,707 Alpaca delistings, SIP feed, filtered on 2021 characteristics) gave the honest number, and
   it **collapsed momentum's 1.1+ back to ~0.4** (Case 45) — because representative mid-cap delistings are
   mostly **acquisitions, not bankruptcies**. Two rules: (i) point-in-time universes are mandatory; (ii)
   the delisting set must be *selected on entry-date characteristics, never on outcome* — or you just
   trade one survivorship bias for its mirror image.

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
