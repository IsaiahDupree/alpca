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
| 12 | **Final combiner** of all surviving sleeves | — | 🔄 | pairs (0.83) + value (thin) |

## Reference points
- The bar that rejects: 3 candidates (EAR-PEAD, SI-level, accruals) passed every in-universe test and
  died only on the fresh-symbol holdout. Value generalized but was too weak. BAB needs leverage.
- The one validated edge: cointegrated-pairs basket, **WF ~0.83** (top-10 + 5% ADF screen), −4% DD.
- Expectation: most of these will reject — the job is to reject cheaply and find the rare survivor.

*Detailed write-ups land in `EDGE_CASE_STUDIES.md` (Cases 26+); this file tracks order + status.*
