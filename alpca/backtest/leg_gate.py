"""
The SECOND-LEG GATE — the distilled, automated version of the Cases 47–51 hunt.

Across the second-leg search, every candidate was judged by hand against the same gauntlet; this codifies
it into one tested call. Given a candidate sleeve's DATED daily returns and the deployed book's DATED
returns, it date-aligns them and runs the five checks a real diversifying leg must pass — each one was
*learned the hard way* from a failed candidate:

  1. forward_positive  — candidate Sharpe > 0 over the overlap window. (momentum was NEGATIVE over
                         2022→; Case 47.)
  2. uncorrelated      — |corr(candidate, book)| < max_rho. (the whole point of a diversifier.)
  3. lifts             — inverse-vol combined Sharpe > book-alone + min_lift. (momentum & trend DILUTE;
                         Cases 47, 51.)
  4. robust_loo        — the lift is positive in ≥ min_loo_frac of leave-one-year-out folds. (a lift
                         carried by one year isn't robust.)
  5. partial_year_safe — the lift survives EXCLUDING the most-recent (often partial) year. (seasonality's
                         lift was a partial-2026 artifact; Case 48.)

GO iff all five pass. Fail-closed and explainable: every sub-result is returned with its number.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from alpca.backtest.combine import evaluate_combo, correlation
from alpca.backtest.evaluation import sharpe_of

PPY = 252.0


def _eq(daily, s=1.0):
    e = [s]
    for x in daily:
        e.append(e[-1] * (1 + x))
    return e


def _year(ts):
    return time.gmtime(int(ts)).tm_year


@dataclass
class LegVerdict:
    passed: bool
    n_common: int
    candidate_sharpe: float
    book_sharpe: float
    rho: float
    combined_sharpe: float
    lift: float
    loo_positive_frac: float
    ex_recent_lift: float
    checks: Dict[str, bool] = field(default_factory=dict)
    reasons: List[str] = field(default_factory=list)


def evaluate_leg_candidate(
    candidate: Dict[int, float], book: Dict[int, float], *,
    max_rho: float = 0.30, min_lift: float = 0.03, min_loo_frac: float = 0.6,
    book_label: str = "book",
) -> LegVerdict:
    """Run the five-check second-leg gauntlet. `candidate`/`book` are {epoch: daily_return}."""
    common = sorted(set(candidate) & set(book))
    if len(common) < 60:
        return LegVerdict(False, len(common), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                          {"enough_overlap": False}, ["insufficient overlapping history (<60 days)"])
    cv = [candidate[t] for t in common]
    bv = [book[t] for t in common]
    cand_sh = sharpe_of(_eq(cv), PPY)
    book_sh = sharpe_of(_eq(bv), PPY)
    rho = correlation(cv, bv)
    rep = evaluate_combo({book_label: bv, "candidate": cv}, ppy=PPY)
    w = rep.invvol_weights
    blended = {t: w[book_label] * book[t] + w["candidate"] * candidate[t] for t in common}
    combined_sh = rep.invvol_sharpe
    lift = combined_sh - book_sh

    # leave-one-year-out: is the lift positive in most folds?
    years = sorted({_year(t) for t in common})
    loo_pos = 0; loo_n = 0
    for drop in years:
        idx = [t for t in common if _year(t) != drop]
        if len(idx) < 40:
            continue
        cb = sharpe_of(_eq([blended[t] for t in idx]), PPY)
        pa = sharpe_of(_eq([book[t] for t in idx]), PPY)
        loo_n += 1
        if cb - pa > 0:
            loo_pos += 1
    loo_frac = (loo_pos / loo_n) if loo_n else 0.0

    # partial-year safety: lift survives excluding the most-recent year
    recent = years[-1]
    idx = [t for t in common if _year(t) != recent]
    ex_lift = 0.0
    if len(idx) >= 40:
        ex_lift = sharpe_of(_eq([blended[t] for t in idx]), PPY) - sharpe_of(_eq([book[t] for t in idx]), PPY)

    checks = {
        "forward_positive": cand_sh > 0,
        "uncorrelated": abs(rho) < max_rho,
        "lifts": lift > min_lift,
        "robust_loo": loo_frac >= min_loo_frac,
        "partial_year_safe": ex_lift > 0,
    }
    reasons = []
    if not checks["forward_positive"]:
        reasons.append(f"candidate Sharpe {cand_sh:.2f} ≤ 0 over the overlap window")
    if not checks["uncorrelated"]:
        reasons.append(f"|ρ| {abs(rho):.2f} ≥ {max_rho} — not a diversifier")
    if not checks["lifts"]:
        reasons.append(f"combined {combined_sh:.2f} does not lift book {book_sh:.2f} by {min_lift} (lift {lift:+.2f})")
    if not checks["robust_loo"]:
        reasons.append(f"lift positive in only {loo_frac:.0%} of leave-one-year-out folds (< {min_loo_frac:.0%})")
    if not checks["partial_year_safe"]:
        reasons.append(f"lift does NOT survive excluding the most-recent year ({recent}): ex-recent lift {ex_lift:+.2f}")
    return LegVerdict(
        passed=all(checks.values()), n_common=len(common), candidate_sharpe=round(cand_sh, 3),
        book_sharpe=round(book_sh, 3), rho=round(rho, 4), combined_sharpe=round(combined_sh, 3),
        lift=round(lift, 3), loo_positive_frac=round(loo_frac, 3), ex_recent_lift=round(ex_lift, 3),
        checks=checks, reasons=reasons or ["all five checks passed — a real diversifying leg"])
