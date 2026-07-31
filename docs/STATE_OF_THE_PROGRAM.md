# Alpca — State of the Program (the package)

*The single-page truth, current as of 2026-07-30. For depth: `FINDINGS.md` (executive synthesis),
`EDGE_CASE_STUDIES.md` (all 68 cases), `SECOND_LEG_HUNT.md` (the 2nd-leg narrative, Cases 41-48),
`SYSTEM_MAP.md` (platform + deployed portfolio), `AI_RESEARCH_LOOP.md`, `PIPELINE_AND_AUTOMATION.md`,
`PAPER_EDGE.md` (the live forward-track scoreboard — auto-generated, updated daily). This file is the
honest scoreboard + what we'd hand to a skeptical reviewer.*

## What this project is

An **honest-evaluation platform** for systematic trading on Alpaca. Its deliverable is not a pile of
"edges" — it is a harness whose job is to **reject** strategies that don't generalize, and to catch its
*own* overselling. Across **68 distinct edge experiments** (price, microstructure, event-driven,
positioning, fundamental — run by hand, by an AI research loop, and by two independent literature/corpus
harvests at scale (176 + 217 candidates, Cases 62/64), then adversarially self-audited), the honest result
is **exactly one thin, regime-dependent core edge plus one small diversifying overlay**, with everything
else rejected on a *different* honest gate. That is the point, not a disappointment: real edges are
scarce, and the machine that says so cheaply is the asset.

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
- **Out-of-regime confirmed** (Case 60): the SIP feed unlocked 10.5yr history (2016-2026, vs the original
  2021-2026 tuning window). Split OOS: **2016-2020 (never seen) Sharpe 0.554**, positive in all four years,
  survives both the 2018 vol-spike and the 2020 COVID crash — *slightly above* the in-regime 2021-2026
  number (0.505). Full 10.5yr Sharpe **0.514**. The honest expectation is revised **down on magnitude
  (~0.5, not the 0.83 headline), up on confidence** (regime-robust, not a 2021-2026 artifact).
- **Deployed book vs the real tail** (Case 67): marked through the actual 2018-02-06 SVXY −83% day and the
  March-2020 crash — the combined book (pairs 92% + short-vol 8% cap) lost only −6.7% on the worst SVXY
  day in history and −8.2% through COVID. Full 2016-2026 combined-book read: **Sharpe 0.63, maxDD −13.3%,
  +9/11 years** — short-vol lifts pairs-alone (0.51 → 0.63).
- Deployed on a **shadow forward paper-track** (`com.alpca.forwardtrack`, daily) — the live OOS curve is
  what ultimately adjudicates. **Current live read** (`docs/PAPER_EDGE.md`, 2026-07-30): pairs 18 realized
  marks, cum −1.68%, Sharpe −3.50 — noisy-but-not-alarming this early (n=18 is far too small to estimate a
  Sharpe; the drawdown is well inside the −4/−10% expected band). Investigated 2026-07-30: the loss is
  **not** a regime break — it's dominated by one idiosyncratic single-name gap (MO −9.3% on 2026-07-29 vs
  SPY +1.65% the same day, breaking one pair leg), exactly the tail risk a market-neutral book is supposed
  to be exposed to. Combined funded book (pairs 92% + short-vol 8%): 31 marks, cum −1.53%, Sharpe −2.64 —
  watch it, don't act on it yet.

Sized to Kelly, a ~0.3–0.5 market-neutral sleeve with a shallow drawdown is a genuinely useful core. The
lever for *more* dollars/day is **additional uncorrelated surviving legs** — still thin (see Direction).

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
| The **formulaic-alpha zoo** (Alpha101/158/GTJA191, 21 family reps) | the famous "101/158/191 alphas" | **fresh-symbol holdout** (Case 56: best DSR 0.596, 0/21 Bonferroni; OOS winners *invert* out-of-universe) |
| **Pairs OU-refinement** (cost-cal entry-z + OU sizing) | hardens the one edge | **inert** (Case 56: bit-identical at ≤20bps; `regime_monitor` variant *destroys* it 0.83→−0.32) |
| **Defensive low-beta / BAB** | low-risk premium | **beta** (Case 56: defensive −1.06; only long-high-β positive, β−0.65 = dampened market) |
| **Mid-cap cointegrated-pairs** (the one mechanism, disjoint universe) | a diversifying 2nd pairs sleeve | **too weak** (Case 57: *clears* survivorship + disjoint-holdout — first mid-cap family to — but DSR 0.06, cadence knife-edge, ρ≈0 yet combiner lift −0.04 = dilutes) |
| **Conference-driven return drift** (WWDC/I-O/Build, mined by the research pipeline) | Sharpe 0.97 published | **beta** (Case 61: long-only underperforms just buy-and-holding the 3 names; beta-hedged alpha is a 29-event recency artifact) |
| **Option-expiration-week SPY**, **bond-ETF duration rotation** (harvest-at-scale, 176 strategies) | documented anomalies | **decisive reject on our window** (Case 62A/B: opex week inverted, −6bps/day; rotation is pure BND beta, alpha −2%/yr) |
| Higher-moment tail hedge (skew/kurt SPY cut) | survives its own control | **real but not ours** (Case 62C: genuine long-equity DD insurance; we run market-neutral, no beta sleeve to apply it to) |
| **Insider buying** (long-only, no borrow wall — the best-shot mined candidate) | 1.06 Sharpe, +1.4%/yr alpha, ρ+0.07 to pairs | **survivorship artifact** (Case 63: on the 1,702-symbol delisted-inclusive universe the spread inverts to −0.61; indistinguishable from a random subset) |
| Market-making / HFT / crypto / news | various | **infeasible on the venue / data** |

**On probation (forward-tracking, not rejected, not funded):**

| Candidate | Backtest read | Status |
|---|---|---|
| **Hardened cross-sectional momentum** (Case 65 — z-weighted + trend-gated, the one edge confirmed independently in two separate programs) | 0.50 in-regime / out-of-regime-stable (0.47/0.54), but t-stat 1.62 < 2 and weakens to 0.23 survivorship-corrected | Real improvement over the old construction, **not yet fundable** — kept on the live forward track |
| **Analyst-revision drift** (Case 66 — EPS-revision momentum, long-only) | No backtest possible (AlphaVantage estimates endpoint is snapshot-only, no PIT history) | **Forward-only by construction** — the clock started 2026-06-24, zero capital until the live curve earns it |
| **Multi-signal composite** (Case 68 — momentum + value + short-interest, z-scored cross-section) | Sharpe 0.64, t 2.05 (first significant x-sec result), out-of-regime 0.60, fresh-symbol holdout positive (0.90→0.28) | Clears every control **we can run** on survivor-only data — but survivorship itself is structurally un-runnable (EDGAR/FINRA have zero delisted coverage), so it forward-tracks instead. *Bug found + fixed 2026-07-30: the forward track's price cache was never being refreshed by the cron job, so every realized mark read exactly 0.0 for 26 days — the clock has now actually started.* |

## The hard-won lessons (the real IP)

1. **Only out-of-universe (fresh-symbol) + out-of-regime (held-out years) holdouts adjudicate.** In-sample
   fit, a positive time-split, per-year stability, a high Deflated Sharpe, subset resampling, *and* a
   five-check leg gate can **all pass on an edge that is still overfit.** The fresh-symbol holdout has now
   caught ~7 such candidates.
2. **The two upstream bars are COMPLEMENTARY — clear BOTH, neither alone suffices.** Reversal and EAR-PEAD
   cleared every leg-gate check (positive, uncorrelated, lifts, robust LOO, partial-year-safe) *and* the 2024+
   slice — and still died on the survivorship/fresh-symbol bar (it catches *fake-but-strong* edges). The
   *converse* (Case 57): mid-cap pairs **cleared** the survivorship + disjoint-half holdout — the first mid-cap
   family to — yet the leg gate killed it on **strength** (DSR 0.06, cadence knife-edge, combiner lift −0.04 =
   dilutes). The survivorship bar catches fake-but-strong; the leg gate catches *real-but-too-weak*. *(And an
   **inverted survivorship delta** — PIT > survivor — is a yellow flag, not a green one: it can mean a few
   delisting events are propping up a losing book.)*
3. **Survivorship bias can flip which signal is real — and a careless "fix" re-biases the other way.** Only
   the *full, outcome-blind* point-in-time universe (all delistings, selected on entry-date traits) is honest.
4. **A combiner number is only honest if BOTH legs are OOS and measured over the SAME window** (the in-sample
   pairs + 2021-included momentum that manufactured a fake 1.08 lift).
5. **Costs and borrow are destiny**; the alpha is often on the side you can't trade (momentum's short leg).
6. **Turn the adversarial lens on your OWN claims.** A 9-agent self-audit (Case 52) caught the platform
   overselling — pairs "0.83" → forward ~0.3–0.5; short-vol "diversifier" → 2022–23 artifact.
7. **Idea supply exhausts before venue constraints do.** Two independent, at-scale sweeps — the public
   literature (Case 62: 176 harvested strategies → 65 novel/feasible → 2 decisive rejects + 1
   not-applicable) and the user's private 217-strategy trading-knowledge corpus, forward-tested across
   *five other codebases* (Case 64) — converge on the same result: **zero new Alpaca-deployable alpha**,
   and the one directional edge that *does* replicate across both programs (cross-sectional momentum,
   ~0.68) still fails Alpca's own survivorship/significance bar (Case 65). The bottleneck is no longer
   "we haven't looked" — it's the venue (price-taker, no shorting on crypto, thin borrow) and honest
   effect sizes on liquid large-caps.
8. **A live forward-track can silently go stale — check that the INPUTS moved, not just that the job
   exited 0.** Case 68's composite sleeve ran clean (`exit=0`) for 26 straight days while its price cache
   (`cache_sip_10y`) never got refreshed by the cron script — every "realized" mark was mechanically 0.0
   because `last_close` was identical to the entry price every single day. The job's own success signal
   (a nonzero exit code, a `[composite] ... rebalanced=False` log line) gave no hint of the bug. Found
   2026-07-30 by asking *why is this number always exactly 0.00* instead of trusting the pipeline because
   it wasn't erroring. Fixed by adding the missing `download_data.py --feed sip` refresh step to
   `forward_track.sh`. **Lesson: for any forward-track sleeve, periodically confirm the underlying price
   (or signal) data actually has a new `mtime`/value each day — a frozen-but-not-crashing pipeline is the
   most dangerous kind, because it looks exactly like a real (flat) result.**

## Reusable controls banked (the toolkit)

Buy-and-hold benchmark · walk-forward (re-fit/trade/roll) · shuffle placebo · adversarial friction
(adverse-selection borrow) · regime stability · no-lookahead trailing estimates · honest-trial Deflated
Sharpe · **full outcome-blind point-in-time universe** (Alpaca inactive-assets + SIP) ·
**`delisting_aware_walkforward`** (dated OOS returns) · **reverse-split-artifact winsorization** ·
**the second-leg gate** (`leg_gate.py` — positive/uncorrelated/lifts/robust-LOO/partial-year-safe) ·
**partial-year split on any combiner lift** · the canonical deployed-book backtest (caught a real bug) ·
**out-of-regime split on deep (10.5yr) SIP history** (Case 60 — the pairs edge's first confirmation) ·
**at-scale literature/corpus harvesting + LLM extraction** (`alpca/research/` — Cases 61/62/64, 393
candidates screened) · **random-same-size-subset placebo** (Case 63 — the insider-buying spread collapsed
to the 48th percentile once survivorship-corrected) · **forward-track input-liveness check** (Case 68 —
confirm the underlying price/signal cache actually has a fresh value each run, not just a clean exit code).

## Honest profit-per-day reality

At any honest Sharpe (0.3–1.0), expected daily return is a few bps under ~15–25× that in daily noise.
Per-day ROI targeting is noise-mining. The only levers are **(a) more genuinely-uncorrelated surviving
legs** and **(b) Kelly-scaled sizing** — never trading the one thin edge harder.

## Direction

**Idea supply is exhausted, not the venue's patience.** Case 64 confirmed the public literature (176
strategies, Case 62) and the user's private 217-strategy trading-knowledge corpus both converge on zero
new Alpaca-deployable alpha. The program is no longer hunting *new candidates* — it's **waiting on live
clocks**: three sleeves are on probation, forward-tracking because a backtest can't (or shouldn't)
adjudicate them:
- **Composite** (Case 68, momentum+value+SI, backtested 0.64/t=2.05 but survivorship-un-runnable) — its
  forward clock was silently frozen for 26 days (input-cache bug, found + fixed 2026-07-30); now live.
- **Analyst-revision drift** (Case 66, long-only EPS-revision momentum) — snapshot-only data, no backtest
  possible by construction; forward-only since 2026-06-24.
- **Hardened cross-sectional momentum** (Case 65, z-weighted + trend-gated, 0.50 in-regime / 0.47-0.54
  out-of-regime but t=1.62 and weakens to 0.23 survivorship-corrected) — real improvement, kept on the
  live track to see if the combiner-lift case can be made with more overlap data.

Open threads that remain genuinely blocked on data, not idea supply: a **Finnhub earnings source** to
unblock a *full-breadth* PEAD verdict (the 40-name result is suggestive, not definitive — no Finnhub key);
a genuine **long-vol/convexity hedge** *if* short-vol is ever revived; deep pre-2011 daily history (Stooq/
Tiingo) to finish the seasonality-decay test (AV free tier caps at ~100 bars). The binding constraint is
unchanged: **a leg positive over 2022→, uncorrelated with pairs, and robust across years.** None funded
yet beyond short-vol — and the platform's value is that it tells us so, cheaply, instead of overselling
a probation sleeve's backtest number.
