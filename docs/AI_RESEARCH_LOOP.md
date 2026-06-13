# Alpca — the AI Research Loop: implementation, findings, and how it compares

*Companion to `EDGE_CASE_STUDIES.md` (the 23 manual experiments) and `SYSTEM_MAP.md` (the platform).
This documents the AI-driven research loop built 2026-06-13 — what it is, how it's kept safe, what it
found when run live, and how those findings line up with the strategies we tested by hand.*

---

## 1. What it is

A self-driving loop that hunts for market-neutral edges, **gated by the same honest harness** that
judged the 23 manual case studies. One iteration:

```
detect regime  →  PROPOSE a strategy config  →  RUN it through the harness  →  GATE  →  log / flag survivor
 (deterministic)   (OpenAI medium, or            (main + a DISJOINT fresh        (rail veto +
                    heuristic fallback)            universe: sharpe, OOS,          live Haiku
                                                   per-year, cost, fresh-          verdict)
                                                   symbol holdout, DSR)
```

**Components (`alpca/ai/`):**
| Module | Role |
|---|---|
| `regime.py` | deterministic regime detector → bull / bear / chop / high-vol (SPY trend + realized vol) |
| `strategy_generator.py` | proposes a config from a **constrained space of known templates**; runs it through the harness |
| `strategy_gate.py` | the falsification gate: deterministic hard rail (veto) + live Haiku per-regime verdict |
| `router.py` + `oauth.py` | multi-model router — Haiku (cheap) via Claude Code **OAuth subscription** (keychain, auto-refresh); OpenAI medium (heavy reasoning) via a key in gitignored `.env` |

**Strategy space (what the model may configure — never code):**
- `xsec_momentum`, `xsec_reversal` — price-only cross-sectional (lookback / hold / top_k)
- `accruals` — **fundamental** (Sloan earnings-quality, EDGAR; top_frac); diversifies price edges

**Automation:** `scripts/ai_research_loop.py`, scheduled as launchd `com.alpca.airesearch` (weekly
Sat 11:00 ET). Both model tiers verified live under launchd.

---

## 2. The safety architecture — *the model proposes, only the data validates*

This is the design that makes an AI-driven loop trustworthy rather than a hallucination engine:

1. **No arbitrary code.** The model can only *select and configure* known, already-tested backtest
   templates with params clamped to bounds. The worst it can do is propose a config the harness rejects.
2. **The deterministic falsification rail has VETO power.** A candidate is GO only if it clears, in
   code: the **fresh-symbol holdout** (generalizes to unseen symbols), **regime robustness** (positive
   in ≥60% of years), **cost survival** (2 bps), and **DSR ≥ 0.9**. Haiku cannot override a rail FAIL.
3. **Both must agree.** Final GO = `rail_pass AND haiku_GO`. Disagreement → NO-GO.
4. **Fail-closed.** An unparseable model reply defaults to NO-GO.

```
              proposal ─▶ harness (real OOS + fresh-symbol holdout)
                              │
        ┌──────── falsification_gate() ────────┐      ┌── haiku_verdict() ──┐
        │ fresh-symbol · regime · cost · DSR   │  AND │ GO / NO-GO + reason │  ─▶  DECISION
        │   (deterministic, VETO)              │      │ (live, per-regime)  │
        └──────────────────────────────────────┘      └─────────────────────┘
```

The rail encodes the lessons the project paid for across 23 case studies — three different "edges"
(EAR-PEAD, the short-interest tilt, accruals) passed every in-sample/in-universe test yet died on the
fresh-symbol or multi-regime holdout. The loop makes those holdouts mandatory, automatically.

---

## 3. Live findings

Run live on the 195-symbol universe (+30 disjoint fresh), both model tiers active (Haiku via OAuth,
OpenAI via the reused key). Representative iterations (regime = bull):

| # | source | template (config) | sharpe | OOS | **fresh-holdout** | +yrs | rail | Haiku | decision |
|---|---|---|---|---|---|---|---|---|---|
| 1 | OpenAI | accruals (top 10%) | +0.35 | +0.28 | **−0.76** | 4/6 | FAIL | NO-GO | NO-GO |
| 2 | OpenAI | accruals (top 33%) | +0.46 | +0.22 | **−0.45** | 5/6 | FAIL | NO-GO | NO-GO |
| 3 | OpenAI | xsec_momentum lb250 | +0.12 | +0.91 | **−0.12** | 2/6 | FAIL | NO-GO | NO-GO |
| 4 | OpenAI | accruals (top 20%) | +0.30 | −0.29 | **−0.58** | 3/6 | FAIL | NO-GO | NO-GO |
| 5 | OpenAI | xsec_momentum lb20 | +0.48 | +0.80 | **+0.00** | 5/6 | FAIL | NO-GO | NO-GO |

**What we can say:**
- The loop **works end-to-end and unattended** — it detects the regime, OpenAI proposes from both the
  price and fundamental families (it chose `accruals` in 3/5 iterations, reasoning that a fundamental,
  low-turnover edge diversifies a price-heavy book), the harness runs each with a *real fresh-symbol
  holdout*, and Haiku judges per regime.
- **It independently reproduces the manual findings.** The accruals proposals look fine in-sample
  (+0.3–0.5 Sharpe, positive in 4–5 of 6 years) but the **fresh-symbol holdout is negative every time**
  (−0.45 to −0.76) — exactly the Case 23 result we found by hand. The loop caught the same overfit
  without being told.
- **0 survivors across every run.** That is the correct, honest outcome: the templates currently in the
  space are known-weak, and the bar (fresh-symbol + out-of-regime + cost + DSR) is brutal by design.
  **An AI driving the proposals did not lower the bar.**

---

## 4. How it compares to the manually-tested strategies

The loop applies the **identical bar** as the 23 hand-run case studies, so the comparison is apples-to-apples.

| Strategy (how tested) | In-sample / in-universe | Out-of-sample verdict | Status |
|---|---|---|---|
| **Cointegrated-pairs basket** (manual) | IS 1.78 | walk-forward **0.29** (decayed) | ✅ the **only** validated edge — deployed on a forward paper-track |
| EAR-PEAD beta-hedged (manual) | 0.68, 6/6 yrs, DSR 0.89 | fresh-symbol holdout **−0.52** | ❌ overfit |
| Short-interest / borrow tilt (manual) | 1-yr 2.34, DSR 0.98 | 6-yr FINRA **−0.42** | ❌ regime artifact |
| Accruals (manual **and** AI-loop) | +0.44, +5/6 yrs, cost-free | fresh-symbol holdout **−0.45 to −0.76** | ❌ overfit (confirmed twice) |
| Overnight / gap / lead-lag (manual) | various | cost wall / placebo fail | ❌ |
| `xsec_momentum/reversal` (AI-loop) | mixed | fresh-holdout ≤ 0 | ❌ |

**The throughline:** across **24 distinct experiments — manual and AI-proposed — exactly one edge has
cleared the out-of-sample bar** (the pairs basket, and even that has decayed to a marginal WF 0.29).
The AI loop changes *how fast and cheaply we can search*, not *what survives*. The discipline — fresh
symbols + new regimes + realistic costs — remains the binding filter, and it holds whether a human or a
model is proposing.

---

## 5. What this means / direction

- **The loop is a force-multiplier on the search, not a shortcut past the data.** Its value is running
  the brutal honest tests automatically on anything proposed, so we can explore a far wider space
  cheaply while never shipping an overfit edge (it has shipped zero, correctly).
- **Edge supply is still the binding constraint.** One marginal validated edge. More candidates need
  *richer, more-likely-to-generalize* templates in the space — the next ones: a **value composite**
  (E/P, FCF/P via EDGAR), **financials-excluded sector-neutral accruals** on a broad fresh universe,
  and a **real ADF/Johansen cointegration screen** to try to lift the pairs basket off 0.29.
- **Profit-per-day reality is unchanged:** ~1–2 bps/day from the one real edge, sized to Kelly — the
  lever is *more surviving legs*, which the loop now hunts for around the clock.

---

## 6. Reproducibility & security

- Run manually: `.venv/bin/python scripts/ai_research_loop.py --iterations 6` (add `--no-ai` for the
  heuristic proposer; needs no OpenAI key). Scheduled weekly via launchd `com.alpca.airesearch`.
- **Credentials, never exposed:** Haiku uses the local Claude Code **OAuth** token from the macOS
  Keychain (auto-refreshed; nothing written to the repo); OpenAI uses a key in `Alpca/.env`, which is
  **gitignored**. No token or key appears in any source file, log committed to git, or this document.
  Verified: `git check-ignore .env` and a secret scan of every commit.
- 22 offline AI tests (routing, OAuth header shape, regime labels, constrained-proposal validation,
  gate veto, fail-closed, fundamental-template run). Full suite 3420 green.
