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

**Concrete next steps to enable it:**
1. Add `ANTHROPIC_API_KEY` + `OPENAI_API_KEY` to `.env` (gitignored). *(Claude Code's own OAuth login
   is for the CLI; programmatic loop calls need API keys.)*
2. `alpca/ai/router.py` — a thin multi-model router (scaffolded; reads keys from env, picks Haiku vs
   OpenAI-medium by task size). No keys in code.
3. A regime classifier (`alpca/backtest/regime.py`) — cheap, deterministic, feeds the loop.
4. Wire the loop as a launchd research job that proposes → tests → logs, gated by the harness.

**Near-term edge veins (human or AI-proposed) that fit the data + bar:** financials-excluded
sector-neutral accruals (broad fresh universe), value composite (E/P, FCF/P via EDGAR), post-earnings
drift conditioned on regime, and a real ADF/Johansen cointegration screen to lift the pairs basket
off its decayed 0.29.

---

## 7. The honest bottom line

After 23 experiments: **one marginal, decaying, market-neutral edge (pairs, WF ~0.29), deployed
small on a forward track.** The realistic profit-per-day ceiling today is ~1–2 bps — invisible
day-to-day, real over hundreds of days. The lever for more is **more *generalizing* edges sized to
Kelly**, found through a disciplined (now possibly AI-accelerated) loop — *not* trading the one thin
edge harder. The platform's value is that it tells us this truth instead of a comfortable fiction.
