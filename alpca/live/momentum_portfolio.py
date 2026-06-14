"""
Deploy layer for the modest SECOND-EDGE candidate — the mid-cap vol-managed-momentum sleeve, in its
honest BORROW-FREE form: LONG the top-quintile momentum winners (equal-weight) + SHORT SPY to neutralize
market beta (Case 45, ~0.23 Sharpe). No single-name shorts -> no borrow wall. The long leg is
survivorship-clean (winners don't delist; acquired winners are held into a buyout premium).

No look-ahead: the momentum signal uses only the trailing window ending at the as-of bar. The book is
MONTHLY-rebalanced (momentum is slow) — the deploy script carries the prior winners forward between
rebalances; this module just computes the target winners + the hedge for a rebalance day.

Position convention matches `vol_managed_momentum_signal` + the long_index_hedge mode of
`scripts/test_momentum_borrow.py`: equal-weight long the top `top_frac` winners summing to +1.0,
SPY weight −1.0 (dollar-for-dollar beta hedge). `size_book` then scales the whole sleeve to a
conservative half-Kelly / vol-target leverage on the honest Sharpe.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from alpca.backtest.factor import _price_ret, vol_managed_momentum_signal


@dataclass
class MomentumBook:
    longs: Dict[str, float]        # {winner symbol: +weight}, equal-weight summing to +1.0
    spy_weight: float              # index hedge, −1.0 when invested (0.0 if insufficient data)
    n_winners: int
    weights: Dict[str, float]      # combined unit book (longs + {SPY: spy_weight}); gross ~2.0
    asof: float = 0.0


def compute_momentum_book(
    bars_by_sym: Dict[str, List[dict]],
    spy_bars: List[dict], *,
    top_frac: float = 0.2,
    lookback: int = 120,
    skip: int = 21,
    vol_window: int = 60,
    spy_symbol: str = "SPY",
) -> MomentumBook:
    syms = sorted(bars_by_sym)
    if len(syms) < max(10, int(round(2 / max(top_frac, 1e-9)))):
        return MomentumBook({}, 0.0, 0, {}, 0.0)
    master = sorted({int(b["timestamp"]) for s in syms for b in bars_by_sym[s]})
    if len(master) < lookback + skip + vol_window + 5:
        return MomentumBook({}, 0.0, 0, {}, 0.0)
    price, _ = _price_ret(bars_by_sym, syms, master)
    signal = vol_managed_momentum_signal(lookback, skip, vol_window)(master, syms, price)
    s = signal[-1]                                   # as-of the most recent bar (no look-ahead)
    ok = np.isfinite(s)
    k = max(1, int(round(len(syms) * top_frac)))
    if ok.sum() < 2 * k:
        return MomentumBook({}, 0.0, 0, {}, float(master[-1]))
    order = np.argsort(np.where(ok, s, -np.inf))
    order = order[np.isin(order, np.where(ok)[0])]
    winners = order[-k:]                             # highest vol-managed momentum
    longs = {syms[j]: 1.0 / k for j in winners}
    spy_w = -1.0 if spy_bars else 0.0
    weights = dict(longs)
    if spy_w:
        weights[spy_symbol] = spy_w
    return MomentumBook(longs=longs, spy_weight=spy_w, n_winners=len(winners),
                        weights=weights, asof=float(master[-1]))


def half_kelly_leverage(sharpe: float, ann_vol: float, *, fraction: float = 0.5, cap: float = 1.0) -> float:
    if ann_vol <= 0 or sharpe <= 0:
        return 0.0
    return max(0.0, min(cap, fraction * sharpe / ann_vol))


def size_book(book: MomentumBook, *, sleeve_sharpe: float, ann_vol: float, target_vol: float = 0.05,
              kelly_fraction: float = 0.5, cap: float = 1.0) -> Dict[str, float]:
    """Scale the unit book (gross ~2.0: +1 long winners, −1 SPY) to a deployable leverage via
    half-Kelly on the honest Sharpe, clipped so realized vol ≤ target_vol. Modest edge -> small."""
    lev_kelly = half_kelly_leverage(sleeve_sharpe, ann_vol, fraction=kelly_fraction, cap=cap)
    lev_vol = (target_vol / ann_vol) if ann_vol > 0 else 0.0
    lev = max(0.0, min(lev_kelly, lev_vol, cap))
    return {s: w * lev for s, w in book.weights.items()}
