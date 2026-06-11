# Alpca — Combining Edges, AI Loops, and the Honest ROI Answer

This report answers two questions directly: **can combinations of strategies + adaptive "AI
loops" lift our risk-adjusted return**, and **what daily rate-of-return is actually achievable**
on this venue? It pairs a survey of open-source frameworks with our own measured combiner
results. Short version up front: **portfolio construction is the single biggest lever and is
real; "AI loop / RL re-allocation" is mostly hype here; and daily-ROI is the wrong target.**

---

## 1. The load-bearing math: diversification of Sharpe

For **k** strategies, each Sharpe **S**, average pairwise correlation **ρ**, equal-risk-weighted:

```
S_combined = S × √k / √(1 + (k−1)ρ)
```

| Scenario | Combined Sharpe |
|----------|-----------------|
| 4 legs, ρ = 0 (uncorrelated) | S × 2 → **four 0.5s make a 1.0** |
| 4 legs, ρ = 0.3 | S × 1.39 → only **0.69** |
| k legs, ρ = 1 (same edge in disguise) | **S** (no gain at all) |

**Implication:** diversification of Sharpe is the *only* mechanism that turns a 0.5 edge into
something deployable, and it works *only* to the extent the legs are genuinely uncorrelated.
Our rejected stacks (momentum/reversal/TSMOM/PCA) failed because they were secretly the same
beta (ρ→1). The rare uncorrelated set is: **market-neutral basket + risk-reduced beta +
small crypto-long + event-clock seasonality.**

## 2. Our measured combiner result (`backtest/combine.py`)

Five **real** legs blended by inverse-vol + half-Kelly:

| Leg | Annual Sharpe | Role |
|-----|---------------|------|
| rsi-mr | ~1.18 (in-sample) | risk-reduced beta |
| pairs-basket | ~0 (in-sample subset) | market-neutral |
| cross-sectional | ~0.40 | market-neutral |
| turn-of-month | ~0.25 | event-clock |
| pre-FOMC | ~0.25 | event-clock |

- **Cross-leg correlation: avg |off-diagonal| = 0.05** — genuinely uncorrelated. ✅ (This is the
  metric the whole edifice lives or dies on; monitor it OOS as a first-class number.)
- **Inverse-vol blend Sharpe ≈ 0.87**, which **beats/ties the equal-weight null (~0.88)** — the
  allocator works mechanically.
- **But the combined Sharpe sits *below* the best single leg**, because four of five legs are
  weak. **Combining one good leg with weak diversifiers dilutes, it doesn't lift.** The
  formula's "5 legs → 2.4" only materializes if *all five* are genuinely good (~equal Sharpe).
- **Conclusion: the bottleneck is the supply of good uncorrelated edges, not the optimizer.**
  Build inverse-vol + half-Kelly first (de Prado: at low N it beats HRP/NCO/mean-variance OOS);
  the marginal value of a fancier allocator is small.

## 3. Adaptive / "AI loop" layers — what's real vs hype

| Approach | Repos | Verdict on our venue |
|----------|-------|----------------------|
| **Inverse-vol + half-Kelly** | PyPortfolioOpt, Riskfolio-Lib | ✅ **Build this.** Robust default; the 80% of the benefit. |
| HRP / NCO clustering | Riskfolio-Lib, mlfinlab | ⚙️ Only graduates over inverse-vol at ≥6 streams; we have ~5. |
| Regime-switching (HMM) gate | MarketRegimeTrader, hmmlearn | ⚠️ Risk overlay only — buys drawdown reduction, **not** Sharpe; overfits regimes. |
| OLPS (OLMAR/PAMR) | universal-portfolios | ❌ High-turnover mean-reversion that **dies to our costs**; conservative members lift little. |
| Bandit / Thompson rotation | stitchfix/mab | ❌ **Don't build.** ~250 days/yr can't distinguish two 0.5-Sharpe arms; chases noise. |
| **Meta-labeling** (triple-barrier) | hudson-and-thames/mlfinlab | ✅ Genuinely useful — *filters* an existing positive signal (e.g. rsi-mr); cuts bad trades. |
| RL allocation | FinRL / FinRL-Meta | ❌ **Hype here.** Its own best honest result (contest, overfit-rejected) is Sharpe ~0.28 ≈ buy-and-hold. |
| Continual / online learning | — | ❌ Overfits to the latest regime. Use periodic walk-forward re-fit with embargo instead. |

**The only adaptive layers worth building** are a realized-vol / 2-state-HMM **exposure dial**
(risk reduction, not alpha) and **meta-labeling** on an existing signal — both framed honestly
as *variance reduction*, plus **Deflated Sharpe Ratio** in the harness (now added) to protect
against our own ~34-strategy trial count.

## 4. The honest answer: what daily ROI is achievable?

`Expected annual excess return = Sharpe × annual_vol`. Take a *realistic* combined Sharpe of
**~0.9** (≈ three genuinely-uncorrelated 0.5 legs) at a sane **8% vol target**:

- Expected excess ≈ 0.9 × 8% = **~7%/yr** (~11% total at ~4% rates).
- **Per day:** 7% / 252 ≈ **~3 basis points/day**, with daily vol ≈ 8%/√252 ≈ **0.50%/day**.
- **The daily noise (±0.5%) is ~15–20× the daily edge (~0.03%).** You physically *cannot* see
  the edge on any given day; it only emerges over hundreds of days.

**Why "X% per day" is unrealistic, plainly:** "1%/day" compounds to ~1,170%/yr — that implies
Sharpe in the tens or insane leverage; it is a noise-mining artifact, not an edge. At any honest
Sharpe (0.5–1.5) the daily expected return is 3–6 bps and is completely buried under daily
noise. **Targeting a daily-ROI number guarantees you react to noise and tinker your edge to
death.**

**The correct targets:** combined **OOS Sharpe 0.8–1.2**, **max drawdown < 8–10%**, and **DSR
significance** after honest trial-count deflation. That corresponds to **~8–15%/yr at single-
digit vol** — genuinely strong for a paper book, and an *invisible* ~3–5 bps/day. Anyone quoting
percent-per-day is selling noise; the same harness that rejected our other 33 strategies should
reject it too.

## 5. Top 3 combination/loop moves (ranked)

1. **Inverse-vol + half-Kelly blender over genuinely-uncorrelated legs** (built — `combine.py`).
   The only mechanism that converts 0.5 edges into a deployable ~0.8–1.0. Monitor the rolling
   cross-leg correlation as a first-class metric; drop any pair that drifts above ρ≈0.3.
2. **Add validated uncorrelated legs** — the combiner is edge-supply-limited, so the highest-EV
   work is *finding more real edges* (validate PEAD with multi-year data; the event-clock
   seasonality sleeves are already ρ≈0 diversifiers).
3. **Risk-overlay loop, framed honestly:** realized-vol / 2-state-HMM exposure dial + meta-
   labeling on rsi-mr + DSR in the harness (DSR done). Buys drawdown/variance reduction, not
   alpha. **Do NOT build:** RL allocation, bandit rotation, high-turnover OLPS — OOS losers here.

## Sources

Riskfolio-Lib · PyPortfolioOpt · mlfinlab (triple-barrier, DSR) · universal-portfolios ·
FinRL (skeptical) · MarketRegimeTrader · Bailey & López de Prado, *Deflated Sharpe Ratio*
(SSRN 2460551). Full URLs in the research transcript / `RESEARCH_CANDIDATES.md`.
