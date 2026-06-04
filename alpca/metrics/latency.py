"""
Latency + slippage metrics.

Aggregates the per-order lifecycle latencies (signal->submit->ack->fill) and
slippage-vs-intended into percentile summaries, and renders a readable report.

Pure-python (no numpy needed) so it can run anywhere, including the hot path.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List, Optional

from alpca.execution.order import Order


def percentile(sorted_vals: List[float], q: float) -> float:
    """Linear-interpolation percentile. q in [0,1]. Assumes sorted input."""
    if not sorted_vals:
        return float("nan")
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    idx = q * (len(sorted_vals) - 1)
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return sorted_vals[int(idx)]
    frac = idx - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


@dataclass
class StageStats:
    name: str
    count: int
    mean: Optional[float] = None
    p50: Optional[float] = None
    p95: Optional[float] = None
    p99: Optional[float] = None
    min: Optional[float] = None
    max: Optional[float] = None

    @classmethod
    def from_values(cls, name: str, values: Iterable[Optional[float]]) -> "StageStats":
        vals = sorted(v for v in values if v is not None and not math.isnan(v))
        if not vals:
            return cls(name=name, count=0)
        return cls(
            name=name,
            count=len(vals),
            mean=sum(vals) / len(vals),
            p50=percentile(vals, 0.50),
            p95=percentile(vals, 0.95),
            p99=percentile(vals, 0.99),
            min=vals[0],
            max=vals[-1],
        )


# The lifecycle stages we track, in order, with the Order property that yields each.
LATENCY_STAGES = [
    ("signal->submit", "signal_to_submit_ms"),
    ("submit->ack", "submit_to_ack_ms"),
    ("ack->fill", "ack_to_fill_ms"),
    ("submit->fill", "submit_to_fill_ms"),
    ("signal->fill", "signal_to_fill_ms"),
]


@dataclass
class LatencyReport:
    n_orders: int
    n_filled: int
    stages: List[StageStats]
    slippage_bps: StageStats

    def to_dict(self) -> Dict:
        return {
            "n_orders": self.n_orders,
            "n_filled": self.n_filled,
            "stages": [asdict(s) for s in self.stages],
            "slippage_bps": asdict(self.slippage_bps),
        }

    def render(self) -> str:
        def fmt(x: Optional[float]) -> str:
            return f"{x:8.1f}" if x is not None and not math.isnan(x) else "     n/a"

        lines = []
        lines.append(f"Latency report - {self.n_orders} orders, {self.n_filled} filled")
        lines.append(f"{'stage':<16}{'n':>5}{'mean':>9}{'p50':>9}{'p95':>9}{'p99':>9}{'max':>9}   (ms)")
        lines.append("-" * 75)
        for s in self.stages:
            lines.append(
                f"{s.name:<16}{s.count:>5}{fmt(s.mean)}{fmt(s.p50)}{fmt(s.p95)}{fmt(s.p99)}{fmt(s.max)}"
            )
        sl = self.slippage_bps
        lines.append("-" * 75)
        lines.append(
            f"{'slippage(bps)':<16}{sl.count:>5}{fmt(sl.mean)}{fmt(sl.p50)}{fmt(sl.p95)}{fmt(sl.p99)}{fmt(sl.max)}"
        )
        return "\n".join(lines)


def build_latency_report(orders: List[Order]) -> LatencyReport:
    filled = [o for o in orders if o.fill_ts is not None]
    stages = [StageStats.from_values(name, (getattr(o, prop) for o in orders))
              for (name, prop) in LATENCY_STAGES]
    slippage = StageStats.from_values("slippage_bps", (o.slippage_bps for o in orders))
    return LatencyReport(
        n_orders=len(orders),
        n_filled=len(filled),
        stages=stages,
        slippage_bps=slippage,
    )
