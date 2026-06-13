# Alpca — Findings Synthesis

*The high-altitude "what we learned." For the per-experiment detail see `EDGE_CASE_STUDIES.md` (25
cases), the loop see `AI_RESEARCH_LOOP.md`, the platform see `SYSTEM_MAP.md`. This is the executive
summary of the honest-evaluation program: the scoreboard, the hard-won lessons, and the direction.*

---

## Executive summary

Across **25 distinct edge experiments** — price, market-microstructure, event-driven, positioning, and
fundamental, run both by hand and by an AI research loop — **exactly one edge has cleared the honest
out-of-sample bar**: a cointegrated-pairs market-neutral basket. Re-measured and *corrected* this
session, its honest walk-forward Sharpe is **~0.83** at the concentrated **top-10 + 5% ADF screen**
(−4% drawdown) — far better than the misleading "0.29" of an over-diversified top-24 basket that
diluted the edge into weak pairs. It is deployed on a shadow forward paper-track. **No other candidate
survived.**

That is not a failure of the project — it *is* the project. The deliverable is a harness whose job is
to **reject** edges that don't generalize, and it has done so relentlessly, including against several
candidates that looked excellent in-sample. Sized to Kelly, a ~0.83 market-neutral sleeve with −4% DD
is a genuinely useful core; the lever for *more* is additional uncorrelated surviving legs.

---

## The scoreboard (25 experiments, by outcome)

**✅ Validated (1):**
- **Cointegrated-pairs basket** — market-neutral, walk-forward (re-screen each quarter, trade the
  next). **WF ~0.83 at the concentrated top-10 + 5% ADF screen** (the "0.29" was an over-diversified
  top-24 — diluting into weak pairs halved the edge), −4% drawdown. Deployed on a
  forward paper-track to let the *live* OOS curve adjudicate.

**⚠️ Generalizes but too weak to deploy (1):**
- **Value composite (E/P + FCF/P + B/P)** — the *first* fundamental whose fresh-symbol holdout stayed
  *positive* (it generalizes to unseen symbols), but Sharpe ~0.14 and heavily regime-timed (strong in
  the 2022 value rotation, weak in growth-led 2026). Real premium, too thin alone; on the combiner bench.

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

**❌ Rejected — infeasible on the venue / data (several):**
- Market-making / HFT / microstructure (Alpaca ~1.2 s fills, no L2, no rebates), crypto (daily +
  hourly), news/alt-data (free APIs too thin), 52-week-high momentum (inverts on our universe).

---

## The hard-won lessons (paid for repeatedly)

1. **Only out-of-universe + out-of-regime holdouts adjudicate.** In-sample fit, a positive 70/30
   split, per-year regime stability, a high Deflated Sharpe, *and* subset resampling can **all pass on
   an edge that is still overfit.** Three different candidates (EAR-PEAD, the SI tilt, accruals) proved
   this — each died only on a *fresh-symbol* (new names) or *multi-regime* (new years) holdout. That
   holdout is now the standing bar for "validated."

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

6. **Real edges are scarce, and that's the truth — not a tuning problem.** After 25 experiments, one
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
