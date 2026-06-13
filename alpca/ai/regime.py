"""
Market-regime detector — cheap, deterministic, no AI. Feeds the AI research loop so the strategy
generator can propose edges suited to the current regime, and so per-year regime stability has a
label to attach to.

Labels: bull / bear / chop / high_vol (+ unknown when there isn't enough history). Derived from the
benchmark's trailing trend and realized volatility — the two coordinates that most change which
market-neutral templates tend to work (trend-following in bull, reversion in chop, de-risk in high_vol).
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import List


@dataclass
class RegimeState:
    label: str
    trend: float            # trailing return over `lookback`
    vol: float              # annualized realized vol over `lookback`
    drawdown: float         # current drawdown from the trailing high
    n: int                  # bars used

    def as_prompt(self) -> str:
        return (f"regime={self.label} trend={self.trend:+.1%} ann_vol={self.vol:.1%} "
                f"drawdown={self.drawdown:+.1%}")


def detect_regime(bench_bars: List[dict], *, lookback: int = 60, ppy: float = 252.0,
                  trend_thr: float = 0.04, vol_thr: float = 0.25) -> RegimeState:
    cl = [float(b["close"]) for b in bench_bars if b.get("close")]
    if len(cl) < lookback + 1:
        return RegimeState("unknown", 0.0, 0.0, 0.0, len(cl))
    w = cl[-lookback - 1:]
    trend = w[-1] / w[0] - 1.0
    rets = [w[i] / w[i - 1] - 1.0 for i in range(1, len(w)) if w[i - 1] > 0]
    vol = statistics.pstdev(rets) * math.sqrt(ppy) if len(rets) > 1 else 0.0
    peak = max(w)
    drawdown = w[-1] / peak - 1.0 if peak > 0 else 0.0
    if vol > vol_thr:
        label = "high_vol"
    elif trend > trend_thr:
        label = "bull"
    elif trend < -trend_thr:
        label = "bear"
    else:
        label = "chop"
    return RegimeState(label, trend, vol, drawdown, len(w))
