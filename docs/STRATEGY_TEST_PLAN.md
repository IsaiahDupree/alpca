# Alpca — Strategy Testing Plan & Progress Tracker

Ordered plan for the untested-factor backlog, executed by data-readiness (zero-fetch first) then
expected value. Every candidate goes through the **same bar**: vs buy-and-hold where relevant,
out-of-sample split, **per-year regime stability**, **fresh-symbol holdout** (frozen params on a
disjoint universe), realistic cost, and the Deflated Sharpe — then the falsification **gate**.
A candidate is only "validated" if it clears the fresh-symbol + out-of-regime holdout.

Status: ⬜ pending · 🔄 in progress · ✅ done (kept) · ❌ done (rejected) · ⚠️ done (weak/conditional)

| # | Strategy | Data | Status | Result |
|---|---|---|---|---|
| 0 | Generic cross-sectional **factor engine** (`factor.py`) | — | ✅ | built + tested; reused by all factors |
| 1 | **Asset-growth anomaly** (ΔAssets/Assets) | EDGAR (cached) | ❌ | main −0.24, fresh −0.06 (Case 26) |
| 2 | **Net share issuance** (Δshares) | EDGAR (cached) | ❌ | −0.50 / fresh −0.47 (Case 27) |
| 3 | **ROA / earnings quality** (NI/Assets) | EDGAR (cached) | ❌ | −0.23 / fresh −0.18 (Case 28) |
| 4 | **MAX / lottery effect** | daily bars | ❌ | −0.83 / fresh −0.12 (Case 29) |
| 5 | **Idiosyncratic volatility** | daily bars + SPY | ❌ | −0.88 / fresh −0.08 (Case 30) |
| 6 | **Residual momentum** | daily bars + SPY | ❌ | OOS +0.65 but fresh −0.15 (overfit, Case 31) |
| 7 | **Vol-managed momentum** | daily bars | ❌ | OOS +0.77 but fresh −0.08 (overfit, Case 32) |
| 8 | **Short-interest CHANGE** (ΔSI) | FINRA (cached) | ❌ | main −0.27, 2/6 yrs (Case 33) |
| 9 | **Gross profitability** (Novy-Marx) | EDGAR + fetch (Rev, COGS) | ❌ | main −0.31, regime-flipping (Case 34) |
| 10 | **Sector-neutral value + financials-excl accruals** | EDGAR + SIC fetch | ⚠️/❌ | accruals +0.70 in-sample/6-of-6 but **fresh −0.51** (sector-rescue refuted); value unchanged (Case 35) |
| 11 | **VIX term-structure overlay** | fetch ^VIX/^VIX3M | ⏸️ | deferred — data-gated + marginal a priori |
| 12 | **Final combiner** of all surviving sleeves | — | ✅ | pairs 0.83 + value 0.11, corr 0.02; inverse-vol blend **0.53 (DILUTES** — value too thin); deploy pairs alone |
| 13 | **Sector-neutral value** (lift the one generalizing leg) | EDGAR + SIC (cached) | ❌ | in-sample 0.11→0.34 but **fresh +0.70→−0.64** — neutralizing kills generalization (Case 36) |
| 14 | **Value + momentum combo** (AMP, 2nd-leg hunt) | EDGAR + bars | ❌ | every in-universe metric up (main→0.51, OOS-time→+0.76) but **fresh<0 for any mom weight** (Case 37) |
| 15 | **Value on MID-CAPS** (where the premium lives) | new 137-name S&P400 + EDGAR | ⚠️ | **0.21 (2× large-cap 0.11) AND holdout +0.14** — size-tilt confirmed, still sub-rail (Case 38) |
| 16 | **Value on SMALL-CAPS** (push the size-tilt) | new 110-name S&P600 + EDGAR | ❌ | **inverts to −0.26** — value-trap + small-cap bear; size-tilt non-monotonic, peaks mid (Case 39) |
| 17 | **Mid-cap value + LIGHT momentum** (AMP where premia real) | midcap bars + EDGAR | ⚠️ | **0.39 / holdout +0.24** (best generalizing fundamental); light tilt *helps*, but fails regime+DSR rail (Case 40) |
| 18 | **Factor zoo on MID-CAPS** (find more like value) | 289-name midcap + EDGAR | ✅ | **momentum family revives** — residual & vol-managed momentum generalize (holdout +0.25, volmom 4/6 yrs); quality/lottery stay dead (Case 41) |
| 19 | **Mid-cap value + vol-managed-momentum combiner** | midcap bars + EDGAR | 🟢 | **−0.26 corr → 5/6 positive years, −7% DD, Sh ~0.4–0.5**; first to meet the validated bar; pending pairs-corr + forward track (Case 42) |
| 20 | **Gate #1: correlation vs deployed pairs basket** | both caches | ✅ | **ρ = −0.03** (uncorrelated), 2-sleeve book 6/6 positive years — diversifies (Case 42 gate) |
| 21 | **Survivorship point-in-time re-test** (delisted names) | +50 delisted via Alpaca | ❌→🟡 | **FLIPS value→momentum**: value 0.35→−0.45 (inflated, buys traps), vol-mgd momentum 0.39→**1.35** (robust, shorts traps) — real edge is momentum, borrow-gated (Case 43) |
| 22 | **Momentum under adverse borrow + long/index-hedge** | midcap+delisted bars | ⚠️ | L/S survives borrow at ~1.1 (PIT, dollar-neutral, winsorized) BUT **long/hedged borrow-free form DEAD** (0.08); alpha is short-side-only, magnitude bracketed 0.3–1.1 (Case 44) |
| 23 | **Representative point-in-time universe** (all 1707 delistings, SIP) | Alpaca inactive-assets API | ✅ | **1.35 was a cherry-pick artifact** — representative delistings are mostly ACQUISITIONS; magnitude collapses to **~0.4** (0.43 gross / 0.30 borrow / **0.23 long-hedge borrow-free**) (Case 45) |
| 24 | **Forward paper-track the momentum sleeve** | live launchd | ✅ | deployed: long/index-hedge sleeve on `com.alpca.forwardtrack` daily, sized tiny on honest 0.23, uncorrelated w/ pairs |
| 25 | **Survivorship test of the DEPLOYED pairs edge** | SIP PIT large-cap (195+32) | ✅ | **ROBUST**: delisting-aware WF +0.83→+0.93; delisted legs rarely traded (3 names); new `delisting_aware_walkforward` banked (Case 46) |
| 26 | **Honest two-sleeve combiner** (WF pairs + momentum, dated) | both SIP caches | ❌ | **DILUTES**: momentum −0.15 over the OOS overlap (carried by 2021); inverse-vol combined 0.47 < pairs-alone 0.83 → deploy pairs alone (Case 47) |
| 27 | **Cross-sectional calendar seasonality** (2nd-leg hunt) | midcap bars (zero new data) | ❌ | uncorrelated (ρ=0.06) + looked +0.35 OOS, but per-year unstable (−2025/−2026); 0.83→0.88 lift was a partial-2026 artifact, dilutes without it (Case 48) |
| 28 | **Short-volatility / VRP** (2nd-leg hunt, new risk axis) | VXX/SVXY ETPs (SIP) | 🟢 | **FIRST leg that LIFTS**: combined 0.83→1.08, DSR 0.90, ρ=0.04, 6/6 yrs, combined DD unchanged −5.5% at 12% size; asterisk = un-sampled −46% tail (Case 49) |
| 29 | **Short-vol TAIL STRESS** (simulate the volmageddon) | injected shock | ✅ | 8% cap VALIDATED: −50% SVXY spike → −7.9% combined DD (−9.4% at −70%); sizing is the tail mgmt (Case 50) |

## Post-plan: hunting the second leg (Cases 36–40)
With the backlog closed, the next-highest-EV move was to *strengthen the one fundamental that
generalizes* (value) rather than hunt blind. **On large-caps both refinements failed** — sector-neutral
value and value+momentum *raised every in-universe number but failed the fresh-symbol holdout* (Case 37
is the cleanest proof that **out-of-sample-in-time ≠ out-of-sample-in-symbols**: a +0.76 chronological
split, a −0.23 disjoint-symbol split). **Then we moved off large-caps**, where the literature says the
premia don't live: value on a fresh **mid-cap** universe scored **0.21 (2× large-cap) and generalized
(+0.14)**, and a **light** momentum tilt lifted it to **0.39 / holdout +0.24** — the strongest
generalizing fundamental in the program, *but still short of the regime+DSR rail*. The size-tilt is
**non-monotonic**: it **inverts in small-caps (−0.26)** — value-trap + the 2021–26 small-cap bear — so
mid-cap is a genuine local sweet spot, not a ladder. Net: still no *deployable* second leg, but the first
real lead (mid-cap value+light-momentum), held at zero size until it clears the rail.

## Outcome
**Plan complete (11 tested, 1 deferred) + post-plan size-tilt hunt (Cases 36–40).** Of the whole
untested-factor backlog, **zero new deployable edges** — the classic factor zoo (Cases 26–34) is absent
on our large-caps, BAB needs leverage, and the sector-rescue for accruals was refuted (Case 35). The
post-plan value-refinement hunt found **one genuine lead but no deployable second leg**: mid-cap
value+light-momentum (0.39, holdout +0.24) clears generalization but fails the regime+DSR rail. The one
validated edge remains the **cointegrated-pairs basket, corrected upward to WF ~0.83** (top-10 + 5% ADF),
−4% DD. Net: the deployable portfolio is the pairs basket alone; the search continues for a *second*
genuinely-good uncorrelated leg (the binding constraint), with mid-cap value+momentum the leading
candidate. The reusable factor engine + AI loop make the next candidate a quick, gated test.

## Reference points
- The bar that rejects: 3 candidates (EAR-PEAD, SI-level, accruals) passed every in-universe test and
  died only on the fresh-symbol holdout. Value generalized but was too weak. BAB needs leverage.
- The one validated edge: cointegrated-pairs basket, **WF ~0.83** (top-10 + 5% ADF screen), −4% DD.
- Expectation: most of these will reject — the job is to reject cheaply and find the rare survivor.

*Detailed write-ups land in `EDGE_CASE_STUDIES.md` (Cases 26+); this file tracks order + status.*
