"""
Deploy layer for the ONE validated edge — the cointegrated-pairs basket. Turns the research
(screen → backtest → walk-forward) into a LIVE target book: given bars up to today, which pairs
are active, what each leg should hold, and how large the whole basket should be sized.

No look-ahead: screening uses only the trailing `train` window ending at the as-of bar, and the
z-score uses the trailing `lookback` of the hedged spread. Position convention matches
`backtest_pairs` exactly — EQUAL-DOLLAR legs (long A / short B for a long-spread), the hedge ratio
defines the spread/z signal, not the leg sizing. State carries hysteresis (hold until |z| exits)
so the live book doesn't churn — the prior book is passed in.
"""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from alpca.backtest.pairs import align, screen_pairs


@dataclass
class PairTarget:
    a: str
    b: str
    hedge: float
    z: float
    state: int            # +1 long-spread (long A / short B), -1 short-spread, 0 flat
    half_life: float


@dataclass
class PairsBook:
    targets: List[PairTarget]
    weights: Dict[str, float]      # per-symbol NET target weight, gross-normalized to 1.0
    n_active: int
    gross: float                   # gross exposure before sizing (sum |leg|)
    state: Dict[str, int]          # {"A|B": state} to feed back next call (hysteresis)
    asof: float = 0.0


def compute_pairs_book(
    bars_by_sym: Dict[str, List[dict]], *,
    train: int = 378,
    top_n: int = 10,                  # CONCENTRATED top-10 (WF 0.83); diluting to 20+ halved it
    lookback: int = 60,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
    max_half_life: float = 30.0,
    min_half_life: float = 3.0,
    max_adf: float = -2.86,           # 5% ADF significance screen on the spread (free lift + tighter DD)
    prior_state: Optional[Dict[str, int]] = None,
) -> PairsBook:
    syms = [s for s in sorted(bars_by_sym) if len(bars_by_sym[s]) >= lookback + 5]
    # trailing window only (no look-ahead): screen + signal off the last `train` bars
    trimmed = {s: sorted(bars_by_sym[s], key=lambda x: int(x["timestamp"]))[-train:] for s in syms}
    pairs = screen_pairs(list(trimmed), trimmed, min_overlap=min(train // 2, 200),
                         max_half_life=max_half_life, min_half_life=min_half_life, max_adf=max_adf)[:top_n]
    prior_state = dict(prior_state or {})
    targets: List[PairTarget] = []
    raw: Dict[str, float] = defaultdict(float)
    new_state: Dict[str, int] = {}
    asof = 0.0
    for p in pairs:
        a, b, h = p["a"], p["b"], float(p["hedge"])
        rows = align(trimmed[a], trimmed[b])
        if len(rows) < lookback + 1:
            continue
        asof = max(asof, rows[-1][0])
        la = [math.log(pa) for _, pa, _ in rows]
        lb = [math.log(pb) for _, _, pb in rows]
        spread = [la[i] - h * lb[i] for i in range(len(rows))]
        win = spread[-lookback:]
        mu, sd = statistics.fmean(win), statistics.pstdev(win)
        z = (spread[-1] - mu) / sd if sd > 0 else 0.0
        key = f"{a}|{b}"
        prev = prior_state.get(key, 0)
        state = prev
        if prev == 0:
            if z > entry_z:
                state = -1
            elif z < -entry_z:
                state = 1
        elif prev == 1 and z >= -exit_z:
            state = 0
        elif prev == -1 and z <= exit_z:
            state = 0
        new_state[key] = state
        targets.append(PairTarget(a, b, round(h, 3), round(z, 2), state, p["half_life"]))
        if state != 0:
            raw[a] += state * 0.5          # equal-dollar legs (matches backtest_pairs)
            raw[b] += -state * 0.5
    gross = sum(abs(w) for w in raw.values())
    weights = {s: w / gross for s, w in raw.items()} if gross > 0 else {}
    return PairsBook(targets=targets, weights=weights,
                     n_active=sum(1 for t in targets if t.state != 0),
                     gross=gross, state=new_state, asof=asof)


def half_kelly_leverage(basket_sharpe: float, ann_vol: float, *, fraction: float = 0.5,
                        cap: float = 2.0) -> float:
    """Growth-optimal leverage scaled by a Kelly fraction: f = clip(fraction · Sharpe / σ, 0, cap).
    Mirrors alpca.backtest.combine.half_kelly_leverage so research and deploy agree."""
    if ann_vol <= 0 or basket_sharpe <= 0:
        return 0.0
    return max(0.0, min(cap, fraction * basket_sharpe / ann_vol))


def size_book(book: PairsBook, *, basket_sharpe: float, ann_vol: float, target_vol: float = 0.08,
              kelly_fraction: float = 0.5, cap: float = 2.0) -> Dict[str, float]:
    """Scale the unit-gross book to a deployable leverage. Uses half-Kelly on the basket's
    walk-forward Sharpe, then clips so the realized vol does not exceed `target_vol`."""
    lev_kelly = half_kelly_leverage(basket_sharpe, ann_vol, fraction=kelly_fraction, cap=cap)
    lev_vol = (target_vol / ann_vol) if ann_vol > 0 else 0.0      # vol-target cap
    lev = max(0.0, min(lev_kelly, lev_vol, cap))
    return {s: w * lev for s, w in book.weights.items()}
