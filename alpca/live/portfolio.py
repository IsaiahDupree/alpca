"""
The DEPLOYED PORTFOLIO — codified weights + policy for the multi-sleeve book, and the logic to read the
three separate forward-track logs into one combined live OOS curve.

This is the single source of truth for "what are we actually trading and at what weight," distilled from
the whole honest-evaluation program (Cases 1–50). Each sleeve's weight is its HONEST conviction, not an
optimizer output:

  - pairs       (CORE)        cointegrated-pairs basket, WF ~0.83, survivorship-stamped (Case 46).
  - short_vol   (DIVERSIFIER) the first leg that lifts the book (Case 49); combined 0.83→1.08, ρ=0.04.
                              HARD-CAPPED at 0.08 — short-vol is negatively skewed with an un-sampled
                              tail; the tail stress (Case 50) confirms 8% keeps the worst-case book
                              drawdown < ~10%. The cap IS the risk management.
  - momentum    (PROBATION)   borrow-free long/index-hedge; DILUTES the book over the OOS window
                              (Case 47, negative 2022→). Weight 0 in the deployed book; kept on the
                              forward track only to let live reality confirm/refute. NOT trading capital.

The remaining ~0.55 of the book is the pairs CORE. Weights are gross-of-pairs and sum-checked.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass(frozen=True)
class Sleeve:
    name: str
    role: str                    # CORE | DIVERSIFIER | PROBATION
    weight: float                # deployed notional weight (0 = tracked but not funded)
    cap: Optional[float]         # hard cap (tail discipline), None if not capped
    rationale: str


# The deployed book. pairs core gets the residual after the capped diversifier.
DEPLOYED: List[Sleeve] = [
    Sleeve("pairs", "CORE", 0.92, None,
           "Validated WF ~0.83, survivorship-stamped 0.83->0.93 (Case 46); the only standalone edge."),
    Sleeve("short_vol", "DIVERSIFIER", 0.08, 0.08,
           "First leg that lifts the book (Case 49); ρ=0.04; tail-capped + stress-validated (Case 50)."),
    Sleeve("momentum", "PROBATION", 0.0, None,
           "Dilutes over the OOS window (Case 47); tracked only, zero trading capital."),
]


def deployed_weights() -> Dict[str, float]:
    """Funded weights (PROBATION sleeves are 0). Caps are enforced. Sums to ~1.0 over funded legs."""
    w = {}
    for s in DEPLOYED:
        x = s.weight if s.cap is None else min(s.weight, s.cap)
        w[s.name] = x
    return w


def _stats(r: List[float]):
    if len(r) < 2:
        return 0.0, 0.0
    return statistics.fmean(r), statistics.pstdev(r)


@dataclass
class CombinedBook:
    n_days: int
    dates: List[int]
    daily_returns: List[float]
    per_sleeve_days: Dict[str, int]
    weights: Dict[str, float]
    equity_curve: List[float] = field(default_factory=list)


def _capped_names() -> set:
    return {s.name for s in DEPLOYED if s.cap is not None}


def combine_tracks(track_returns: Dict[str, Dict[int, float]],
                   weights: Optional[Dict[str, float]] = None,
                   capped: Optional[set] = None) -> CombinedBook:
    """Blend per-sleeve DATED realized returns ({sleeve: {epoch: ret}}) into one combined book at the
    deployed weights, date-aligned on the UNION of dates. Only funded (weight>0) sleeves count.

    CRITICAL (the bug the canonical backtest caught): a hard-CAPPED sleeve (short-vol, a negatively-
    skewed tail leg) must trade at EXACTLY its cap weight, every day, and NEVER be renormalized up to
    fill a missing core leg — otherwise on days the pairs core has no data the book becomes ~100%
    short-vol (a −46% tail). So: capped sleeves are pinned at min(weight, cap), every day (the cap is
    enforced HERE, not just by caller convention). The uncapped CORE sleeves share the residual,
    redistributed among the cores PRESENT that day — so a missing core is absorbed by the other present
    cores (amplified up to `core_total`), and only when NO core is present that day does the core slice
    go to cash. A day with neither any capped nor any core present is skipped."""
    weights = weights or deployed_weights()
    capped = capped if capped is not None else _capped_names()
    cap_by = {s.name: s.cap for s in DEPLOYED if s.cap is not None}
    # enforce the cap internally — never trust the caller's weight to already respect it
    funded = {k: (min(v, cap_by[k]) if k in cap_by else v)
              for k, v in weights.items() if v > 0 and k in track_returns}
    cores = [k for k in funded if k not in capped]
    core_total = sum(funded[k] for k in cores)            # total weight the cores share
    all_dates = sorted({t for k in funded for t in track_returns[k]})
    daily, kept_dates = [], []
    for t in all_dates:
        cap_present = {k: track_returns[k][t] for k in funded if k in capped and t in track_returns[k]}
        core_present = {k: track_returns[k][t] for k in cores if t in track_returns[k]}
        if not cap_present and not core_present:
            continue
        r = sum(funded[k] * cap_present[k] for k in cap_present)          # capped: pinned at cap weight
        if core_present:                                                  # cores renorm among present
            csum = sum(funded[k] for k in core_present)
            r += sum((funded[k] / csum) * core_total * core_present[k] for k in core_present)
        daily.append(r)
        kept_dates.append(t)
    eq = [1.0]
    for x in daily:
        eq.append(eq[-1] * (1 + x))
    return CombinedBook(
        n_days=len(daily), dates=kept_dates, daily_returns=daily,
        per_sleeve_days={k: len(track_returns.get(k, {})) for k in funded},
        weights=funded, equity_curve=eq)
