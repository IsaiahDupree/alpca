# Alpca — System Map: what we have, what we don't, and where we're going

*Companion to `ARCHITECTURE.md` (the latency model) and `EDGE_CASE_STUDIES.md` (the 23 edge
experiments). This doc is the bird's-eye view: the codebase, the data, the edges, and the
AI-strategy-loop direction.*

---

## 1. What Alpca is now

Two things fused:
1. **A latency-aware Alpaca *paper* trading bot** — every order carries its full lifecycle as
   wall-clock timestamps; fills are modeled realistically (spread + sqrt-impact + fees + borrow).
2. **A rigorous honest-evaluation platform** — the part that has grown the most. Its job is to
   *reject* edges that don't survive realistic frictions and out-of-sample tests. 23 edge case
   studies; **1 marginal survivor deployed on a forward paper-track, the rest rejected.**

The platform is the product. The discipline (below) is the moat.

---

## 2. Codebase map (`alpca/`, ~10k LOC, 110 test files)

```
data/        FETCH + SHAPE market data (the inputs)
  bars.py            Alpaca daily/minute bars + NBBO quotes (+ split/div adjustment)
  earnings.py        Alpha Vantage / Nasdaq earnings surprise (key in .env)
  feed.py            live bar/quote feed (REST poll + websocket)
  calendar.py        NYSE sessions/holidays; corporate_actions.py; funding.py; news.py

backtest/    THE EDGES + THE HONEST HARNESS (where research lives)
  evaluation.py      ★ the judge: Sharpe/Sortino/DD, vs buy-and-hold, IS/OOS split,
                     PSR + Deflated Sharpe (DSR), beta/alpha, segment/regime Sharpes
  pairs.py           cointegration screen + walk-forward (THE validated edge)
  combine.py         portfolio combiner (inverse-vol + half-Kelly; correlation matrix)
  pead.py / ear_pead.py   earnings drift (surprise / announcement-return) — rejected
  short_interest.py  borrow-fee/SI tilt — rejected (multi-regime)
  accruals.py        Sloan accruals (EDGAR) — rejected (fresh-symbol holdout)
  overnight.py gap_reversion.py high_52w.py lead_lag.py cross_sectional.py
                     tsmom.py stat_arb_pca.py inventory_skew.py seasonality.py  — all tested
  engine.py runner_backtest.py parity.py panel.py   backtest plumbing

execution/   REALISM (how a fill actually happens)
  fills.py (spread+impact+partials+tick) · fees.py (SEC/FINRA) · open_orders.py (resting)
  queue_prob.py (FIFO) · order.py + order_event_log.py (hash-chained) · router.py · adapters/{sim,alpaca}

runtime/     LIVE LOOP        risk/  GATES        metrics/ latency      calibration/ fit fills to real paper
  runner.py, account.py (T+1/PDT/borrow), position_math.py (signed P&L)

live/        DEPLOY            pairs_portfolio.py = current target book + half-Kelly sizing
strategies/  24 registered single-asset strategies (breakout/MR/trend/event/microstructure/council)
cli.py config.py             config auto-loads .env via dotenv (KEY GOTCHA: bare scripts must too)
```

**How it interconnects (data → decision → reality → judgment):**
```
data/ ──▶ strategies/ or backtest/<edge>.py ──▶ execution/ (realistic fill) ──▶ runtime/ (P&L, risk)
                                   │                                              │
                                   └──────────▶ backtest/evaluation.py ◀──────────┘
                                        (Sharpe vs B&H, OOS, DSR, regime, fresh-symbol holdout)
                                                          │
                                   live/pairs_portfolio.py ──▶ shadow forward paper-track (launchd)
```

---

## 3. Data inventory — the Passport drive

All bulk data lives on **`/Volumes/My Passport/AlpcaData/`** (gitignored; the repo holds only code +
2 small deadband JSONs). Mount with `diskutil mount /dev/disk6s1` if absent.

| Dir | What | Source | Cost |
|---|---|---|---|
| `cache/` | 195-symbol 5-yr **daily** bars + 3-yr 1-min SPY/QQQ/AAPL + NBBO | Alpaca | key, ~free |
| `cache_fresh/` | 30 disjoint symbols' daily bars (fresh-symbol holdouts) | Alpaca | key |
| `earnings_av/` | 40 symbols × ~30-yr quarterly earnings surprise | Alpha Vantage | key, **25 req/day** |
| `earnings_av_holdout/` | 19 disjoint symbols' earnings (EAR-PEAD holdout) | Alpha Vantage | quota |
| `short_interest_finra/` | 188 symbols × ~9-yr bi-monthly short interest | **SEC/FINRA** | **free, no quota** |
| `short_interest/` | 56 symbols × ~1-yr SI (Nasdaq; the misleading 1-yr feed) | Nasdaq | free |
| `fundamentals_edgar/` | 164 symbols × annual 10-K (NI / CFO / Assets) | **SEC EDGAR** | **free, no quota** |
| `fundamentals_edgar_fresh/` | 26 disjoint symbols' fundamentals | SEC EDGAR | free |
| `crypto/`, `crypto_hourly/` | rejected crypto experiments | Alpaca | — |

**Key data lesson:** the *free, no-quota, multi-year* sources (SEC EDGAR `companyfacts`, FINRA
`consolidatedShortInterest`) are far better foundations than the quota-limited AV feed — they enabled
the out-of-regime tests that caught 1-year artifacts. **Prefer EDGAR/FINRA for any new fundamental or
positioning signal.**

---

## 4. Where the edges are (and why almost all died)

Full detail in `EDGE_CASE_STUDIES.md` (23 cases) + `strategy_landscape.png`. Honest scoreboard:

- ✅ **Cointegrated-pairs basket — the ONE validated edge.** Walk-forward Sharpe **~0.29 today**
  (decayed from 0.43–0.54 on record), market-neutral, tiny DD. **Deployed** as a small shadow forward
  paper-track (`live/pairs_portfolio.py`, `scripts/deploy_pairs_paper.py`, launchd `com.alpca.forwardtrack`).
- ❌ **Everything else** — momentum/reversal/PCA/TSMOM (beta or no edge), crypto, market-making
  (infeasible on Alpaca), and the session's 5 new candidates: overnight reversal & gap reversion
  (cost wall), lead-lag (fitted noise / failed shuffle placebo), short-interest tilt (1-yr lucky
  window, died on multi-regime FINRA), EAR-PEAD & accruals (passed every in-universe test but **died
  on the fresh-symbol holdout**).

**The meta-lesson, paid for three times (EAR-PEAD, SI tilt, accruals):** in-sample, in-universe,
regime-stability, DSR, and subset-resampling can *all* pass on an edge that is still overfit. **Only
out-of-universe (fresh symbols) + out-of-regime (new years) holdouts adjudicate.** That is now the
standing bar for "validated."

**Anti-overfit toolkit** (each killed/saved a real candidate): buy-and-hold benchmark · walk-forward ·
**shuffle placebo** (learned structure) · adversarial frictions (borrow/slippage) · **regime
stability** (per-year) · no-lookahead trailing estimates · honest trial-count DSR · **multi-regime
data** (FINRA) · **fresh-symbol holdout** (the decisive one).

---

## 5. What we have vs. what we don't

**Have:** honest-eval harness (the moat) · realistic fill/fee/borrow models · multi-source data
pipeline (Alpaca + AV + FINRA + EDGAR) · 24 strategies + ~12 edge families tested · paper deploy +
automated forward track · 5 launchd jobs (calibration, livesession, swing, discovery, forwardtrack,
avearnings) · 3398-test suite.

**Don't have (and why it matters):**
- **A second generalizing edge.** The combiner is edge-supply-starved; 1 marginal leg. *This is the
  binding constraint on profit-per-day.*
- **A live AI research loop** (§6) — currently a human drives each case study.
- **OpenAI / Anthropic API credentials** wired in (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY` are absent
  from `.env` and `~/.env`). Needed to run the AI loop programmatically.
- **Sector-neutral fundamental infra** (financials-excluded accruals, value composite) — EDGAR is
  fetched; the sector map + neutralization is not built.
- **Intraday/HFT alpha** — structurally infeasible on Alpaca (~1.2s fills, no L2, no rebates).

---

## 6. Direction — AI-driven strategy loops for different regimes

The goal: a self-driving research loop where **AI proposes/critiques strategies, the honest harness
adjudicates, per market regime** — humans review the conclusions, not every step.

**Model tiering (cost-aware):**
- **Haiku (Anthropic, cheap/fast)** — small high-volume tasks: parse/label case-study JSON, summarize
  a backtest result, draft a hypothesis stub, classify the current regime from features, write the
  one-line verdict. Pennies per call.
- **OpenAI medium / o-series (larger reasoning)** — the harder thinking: propose *novel* strategy
  hypotheses grounded in the literature, design the falsification test, critique a result for overfit,
  decide the next vein. Fewer calls, more tokens.

**The regime loop (design):**
```
1. classify regime   (bull / bear / chop / high-vol; from SPY trend + realized vol + breadth)
2. propose           (OpenAI: "what edge should work in THIS regime, on data we have?")
3. implement+test    (code the backtest -> run through evaluation.py with the FULL bar:
                      vs B&H, OOS, DSR, regime stability, shuffle placebo, FRESH-SYMBOL holdout)
4. judge             (Haiku: structured verdict; gate = must clear out-of-universe + out-of-regime)
5. if survives       -> add to combiner + forward paper-track; else -> log rejection + why
6. loop              (per regime, on a schedule; humans review the survivors)
```
The harness is the safety rail: **AI can propose anything, but nothing counts until it clears the
fresh-symbol + out-of-regime holdout.** This makes an AI loop *safe* — it can't talk its way past the
data.

**Status — the Haiku per-regime gate is WIRED and LIVE (2026-06-13):**
1. ✅ **Auth via the local Claude Code OAuth token** (same mechanism as the ACD project) — no
   separate API key needed for Haiku. `alpca/ai/oauth.py` reads the macOS Keychain
   (`Claude Code-credentials`), auto-refreshes the ~8h token via `api.anthropic.com/v1/oauth/token`
   and writes the rotated creds back (read-mostly; only writes after a successful refresh). The
   Messages API is called with `Authorization: Bearer …` + `anthropic-beta: oauth-2025-04-20` + the
   Claude Code system prefix. (OpenAI medium still needs `OPENAI_API_KEY` in `.env` when we want it.)
2. ✅ `alpca/ai/router.py` — multi-model router. `small()`→Haiku (OAuth), `think()`→OpenAI medium,
   `route(heavy=)` picks the tier. Keys/tokens read from env/keychain only, never logged.
3. ✅ `alpca/ai/strategy_gate.py` — **`classify_regime()`** (deterministic bull/bear/chop/high-vol) +
   **`falsification_gate()`** (the deterministic hard rail: fresh-symbol holdout, regime-robustness,
   cost-survival, DSR — VETO power) + **`haiku_verdict()`** (live per-regime GO/NO-GO + rationale) +
   **`gate()`** (GO only if rail-pass AND Haiku-GO). *Proven live:* on the real accruals result it
   returned NO-GO (rail veto + Haiku conf 0.98: "fresh-symbol holdout catastrophically negative…
   severe overfitting"). The model cannot talk its way past the data.
4. ✅ **The loop is built and runs live** — `alpca/ai/regime.py` (detector), `alpca/ai/strategy_generator.py`
   (SAFE generator: the model only *configures* known tested templates, never codes), and
   `scripts/ai_research_loop.py` (detect regime → OpenAI proposes → harness runs it on MAIN + a DISJOINT
   fresh universe → `gate()` → log + flag GO survivors). *Proven live:* 195+30-symbol universe, both
   models live (Haiku OAuth + OpenAI), regime=bull, 4 OpenAI proposals, all rail-FAIL + Haiku NO-GO →
   0 survivors (correct — the known-weak cross-sectional space rejects cleanly; the discipline holds with
   AI driving). Both tiers' creds are live (Haiku via Claude Code OAuth keychain; OpenAI via a reused
   service-account key in the gitignored `.env`).
5. ▶ **Next:** (a) wrap `ai_research_loop.py` as a launchd job for unattended runs; (b) widen
   `STRATEGY_SPACE` beyond cross-sectional templates (the fundamental/EDGAR families) so the loop has
   richer, more-likely-to-generalize candidates to propose — the infra now applies the full bar to anything
   added.

**Near-term edge veins (human or AI-proposed) that fit the data + bar:** financials-excluded
sector-neutral accruals (broad fresh universe), value composite (E/P, FCF/P via EDGAR), post-earnings
drift conditioned on regime, and a real ADF/Johansen cointegration screen to lift the pairs basket
off its decayed 0.29.

---

## 7. The deployed portfolio (`alpca/live/portfolio.py`)

The multi-sleeve book is codified in one place — weights are HONEST CONVICTION, not an optimizer output
(see `deployed_weights()` + `combine_tracks()`, and the live view `scripts/report_forward_track.py`,
wired into the daily `com.alpca.forwardtrack` job):

| Sleeve | Role | Weight | Evidence (post-audit, Case 52) |
|---|---|---|---|
| **pairs** | CORE | 92% | WF 0.83 but **regime-dependent** (H1 0.27 / H2 1.39; 2022 −0.9); forward ~0.3–0.5; survivorship-robust (Case 46). The only standalone edge. |
| **short_vol** | OBSERVE (not a validated leg) | 8% (hard cap) | *looked* like a diversifier (combined 1.08, Case 49) but the audit showed the **lift is a 2022–23 vol-crush artifact — since 2024 the book lift is −0.27**. Tail-cap stress-validated (Case 50); kept tiny only to observe if the VRP regime returns. |
| **momentum** | PROBATION | 0% | dilutes over the OOS window (Case 47); gate flips on recency (Case 52) — tracked only, zero capital |

Honest combined-book numbers (Case 52): over the period both legs were live, pairs+short-vol = **1.07
(lift +0.24)** — but that lift is **entirely pre-2024**; on 2024+ data the lift is **−0.27** (short-vol
drags). The canonical union book (incl. pairs-warmup days) is 0.92. The ρ≈0.04 correlation is real and
stress-stable, but "uncorrelated ≠ hedged" — the combined tail is short-vol's. All three sleeves log live
forward tracks; the reporter blends them into the single combined OOS curve that adjudicates the program.

## 8. The honest bottom line

After **52 experiments and an adversarial self-audit (Case 52)**: **exactly one thin, regime-dependent
edge** — the cointegrated-pairs basket, WF 0.83 but forward-honest ~0.3–0.5 (first-half 0.27, 2022 −0.9),
survivorship-robust, deployed on a forward track. The short-vol "second leg" was **downgraded by the
audit**: its diversification was a 2022–23 vol-crush artifact and it has *dragged* the book since 2024 —
kept tiny only to observe. **Zero new edges have cleared the project's own out-of-universe + out-of-regime
bar.** The profit-per-day is a few bps — invisible day-to-day, real over hundreds of days. The lever for
more is a **genuinely uncorrelated additional axis** (ideally a long-vol / convexity sleeve that actually
hedges short-vol's crash, which the correlation analysis proved is undiversified). The platform's value
is exactly this: it caught its *own* overselling — the harness tells the truth, including about itself.
