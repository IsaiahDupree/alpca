"""
Probabilistic queue-position model for resting LIMIT fills.

Today fill_limit() uses a coarse "volume-cap" proxy: a resting limit can take up to
participation_cap of the bar's volume immediately. Reality is FIFO — you sit BEHIND
the size already displayed at your price and only fill once that queue ahead is
consumed. This module models that.

Reimplemented from the nkaz001/hftbacktest queue models (MIT). Two pieces:

  - probability functions f(front, back) -> [0,1] : when displayed size shrinks from
    a CANCELLATION (not a trade), this is the probability the cancel happened AHEAD
    of our order (advancing us). Edge cases: front=0 -> 0 (nothing ahead),
    back=0 -> 1 (everything ahead). Monotone ↑ in front, ↓ in back.

  - QueuePosition : tracks `front` (shares ahead). Trades are FIFO so they consume
    the queue ahead first and only then fill us; cancellations advance us by
    prob_ahead * reduction but never fill us.
"""

from __future__ import annotations

import math


def _combine(ffront: float, fback: float) -> float:
    tot = ffront + fback
    return ffront / tot if tot > 0 else 0.5     # both empty -> symmetric


class PowerProbQueueFunc:
    """prob_ahead = front^n / (front^n + back^n). n>1 makes the advance more
    sensitive to a large queue ahead. (n=1,2,3 are the common variants.)"""
    def __init__(self, n: float = 2.0) -> None:
        self.n = n

    def __call__(self, front: float, back: float) -> float:
        return _combine(max(0.0, front) ** self.n, max(0.0, back) ** self.n)


class LogProbQueueFunc:
    """prob_ahead = log1p(front) / (log1p(front) + log1p(back)) — gentler than power."""
    def __call__(self, front: float, back: float) -> float:
        return _combine(math.log1p(max(0.0, front)), math.log1p(max(0.0, back)))


class SqrtProbQueueFunc:
    """prob_ahead = sqrt(front) / (sqrt(front) + sqrt(back)) — between log and power."""
    def __call__(self, front: float, back: float) -> float:
        return _combine(math.sqrt(max(0.0, front)), math.sqrt(max(0.0, back)))


# the five named variants the plan calls for
def power_prob(n: float = 2.0) -> PowerProbQueueFunc:
    return PowerProbQueueFunc(n)


PROB_FUNCS = {
    "power1": PowerProbQueueFunc(1.0),
    "power2": PowerProbQueueFunc(2.0),
    "power3": PowerProbQueueFunc(3.0),
    "log": LogProbQueueFunc(),
    "sqrt": SqrtProbQueueFunc(),
}


class QueuePosition:
    """
    FIFO queue position for one resting limit order.

    `front` = shares displayed AHEAD of us at our price when we join. Each bar:
      - advance(traded_qty, depth_reduction, back):
          * cancellations (depth_reduction) advance us by prob_ahead * reduction
            (they shrink the queue ahead but DON'T fill us)
          * trades (traded_qty) consume the remaining queue ahead FIFO, then any
            leftover trade volume FILLS our order
        returns the shares that filled this bar.
    """

    def __init__(self, front: float, prob_func=None) -> None:
        self.front0 = max(0.0, front)
        self.front = self.front0
        self.prob = prob_func or PowerProbQueueFunc(2.0)
        self.filled = 0.0

    @property
    def at_front(self) -> bool:
        return self.front <= 1e-9

    def advance(self, traded_qty: float = 0.0, depth_reduction: float = 0.0,
                back: float = 0.0) -> float:
        # cancellations ahead reduce the queue (no fill)
        if depth_reduction > 0 and self.front > 0:
            self.front = max(0.0, self.front - self.prob(self.front, back) * depth_reduction)
        # trades: eat remaining front (FIFO), then fill us
        traded = max(0.0, traded_qty)
        eat = min(self.front, traded)
        self.front -= eat
        fillable = traded - eat
        self.filled += fillable
        return fillable
