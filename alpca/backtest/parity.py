"""
Backtest-vs-live parity report.

Runs the SAME strategy over the SAME bars two ways:
  A) the event-driven backtester (modeled fills: a fixed slippage assumption)
  B) the live execution path (LiveRunner -> ExecutionRouter -> adapter), which
     applies real (or sim-injected) latency + slippage and passes every order
     through the risk gate.

Because the strategy is deterministic, its DECISIONS are identical in both paths;
any divergence in PnL is attributable to EXECUTION — slippage, latency, and any
risk-blocked orders. This quantifies "how realistic is my backtest?".

Offline (default) uses SimAdapter so it runs with no credentials. Point it at an
AlpacaAdapter-backed router to measure parity against real paper fills.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from alpca.backtest.engine import run_backtest
from alpca.config import RiskConfig
from alpca.execution.adapters.sim import SimAdapter
from alpca.execution.order import Side
from alpca.execution.router import ExecutionRouter
from alpca.risk.risk_engine import RiskEngine
from alpca.runtime.runner import LiveRunner
from alpca.strategies.registry import make


def _r2(x: Optional[float]) -> Optional[float]:
    return None if x is None else round(x, 2)


@dataclass
class ParityReport:
    strategy: str
    symbol: str
    n_bars: int

    bt_total_return: float
    bt_n_trades: int
    bt_slippage_bps: float

    live_total_return: float
    live_entries: int
    live_fills: int
    live_rejects: int
    live_realized_slippage_mean_bps: Optional[float]
    live_realized_slippage_p95_bps: Optional[float]

    signal_to_fill_p50_ms: Optional[float]
    signal_to_fill_p95_ms: Optional[float]

    extras: Dict = field(default_factory=dict)
    # --- Phase 2: TCA decomposition of the live fills (tcapy-style). All optional;
    #     populated from the live trades + bars. None when not computable. ---
    arrival_slippage_bps: Optional[float] = None    # fill vs the touch you crossed
    spread_cost_bps: Optional[float] = None         # quoted half→full spread paid
    best_slippage_bps: Optional[float] = None       # had you filled at the bar's best extreme
    worst_slippage_bps: Optional[float] = None      # ...at the worst extreme (brackets realized for
    #                                                 in-bar fills; a market fill slipped past
    #                                                 high/low can exceed it)
    transient_impact_bps: Optional[float] = None    # fill vs mid +1 bar (price reverts?)
    permanent_impact_bps: Optional[float] = None    # fill vs mid +30 bars (did we move it?)
    n_fills_analyzed: int = 0

    @property
    def return_gap(self) -> float:
        return self.live_total_return - self.bt_total_return

    @property
    def slippage_gap_bps(self) -> Optional[float]:
        if self.live_realized_slippage_mean_bps is None:
            return None
        return self.live_realized_slippage_mean_bps - self.bt_slippage_bps

    def to_dict(self) -> Dict:
        return {
            "strategy": self.strategy, "symbol": self.symbol, "n_bars": self.n_bars,
            "backtest": {
                "total_return": round(self.bt_total_return, 4),
                "n_trades": self.bt_n_trades,
                "assumed_slippage_bps": self.bt_slippage_bps,
            },
            "live_path": {
                "total_return": round(self.live_total_return, 4),
                "entries": self.live_entries,
                "fills": self.live_fills,
                "rejects": self.live_rejects,
                "realized_slippage_mean_bps":
                    None if self.live_realized_slippage_mean_bps is None
                    else round(self.live_realized_slippage_mean_bps, 2),
                "realized_slippage_p95_bps":
                    None if self.live_realized_slippage_p95_bps is None
                    else round(self.live_realized_slippage_p95_bps, 2),
                "signal_to_fill_p50_ms":
                    None if self.signal_to_fill_p50_ms is None
                    else round(self.signal_to_fill_p50_ms, 1),
                "signal_to_fill_p95_ms":
                    None if self.signal_to_fill_p95_ms is None
                    else round(self.signal_to_fill_p95_ms, 1),
            },
            "gap": {
                "return_gap": round(self.return_gap, 4),
                "slippage_gap_bps":
                    None if self.slippage_gap_bps is None else round(self.slippage_gap_bps, 2),
            },
            "tca": {
                "n_fills_analyzed": self.n_fills_analyzed,
                "arrival_slippage_bps": _r2(self.arrival_slippage_bps),
                "spread_cost_bps": _r2(self.spread_cost_bps),
                "best_slippage_bps": _r2(self.best_slippage_bps),
                "worst_slippage_bps": _r2(self.worst_slippage_bps),
                "transient_impact_bps": _r2(self.transient_impact_bps),
                "permanent_impact_bps": _r2(self.permanent_impact_bps),
            },
        }

    def render(self) -> str:
        L = []
        L.append(f"Parity: {self.strategy} on {self.symbol}  ({self.n_bars} bars)")
        L.append("-" * 64)
        L.append(f"{'':<26}{'backtest':>16}{'live-path':>16}")
        L.append(f"{'total_return':<26}{self.bt_total_return:>16.4f}{self.live_total_return:>16.4f}")
        L.append(f"{'trades / entries':<26}{self.bt_n_trades:>16}{self.live_entries:>16}")
        sl_live = ("n/a" if self.live_realized_slippage_mean_bps is None
                   else f"{self.live_realized_slippage_mean_bps:.2f}")
        L.append(f"{'slippage bps (assumed/real)':<26}{self.bt_slippage_bps:>16.2f}{sl_live:>16}")
        L.append("-" * 64)
        L.append(f"return gap (live - backtest): {self.return_gap:+.4f}")
        if self.slippage_gap_bps is not None:
            L.append(f"slippage gap (real - assumed): {self.slippage_gap_bps:+.2f} bps")
        if self.signal_to_fill_p50_ms is not None:
            L.append(f"signal->fill latency: p50 {self.signal_to_fill_p50_ms:.1f}ms  "
                     f"p95 {self.signal_to_fill_p95_ms:.1f}ms")
        if self.live_rejects:
            L.append(f"[!] {self.live_rejects} live orders were risk-blocked "
                     f"(a divergence source vs. the backtest)")
        if self.n_fills_analyzed:
            L.append("-" * 64)
            L.append(f"TCA decomposition ({self.n_fills_analyzed} fills):")
            if self.arrival_slippage_bps is not None:
                L.append(f"  arrival slippage : {self.arrival_slippage_bps:+.2f} bps"
                         + (f"   (best {self.best_slippage_bps:+.2f} / "
                            f"worst {self.worst_slippage_bps:+.2f})"
                            if self.best_slippage_bps is not None else ""))
            if self.spread_cost_bps is not None:
                L.append(f"  spread cost      : {self.spread_cost_bps:.2f} bps")
            if self.transient_impact_bps is not None:
                L.append(f"  impact +1 bar    : {self.transient_impact_bps:+.2f} bps (transient)")
            if self.permanent_impact_bps is not None:
                L.append(f"  impact +30 bars  : {self.permanent_impact_bps:+.2f} bps (permanent)")
        return "\n".join(L)


def decompose_execution(trades, bars: List[Dict[str, float]],
                        *, transient_offset: int = 1,
                        permanent_offset: int = 30) -> Dict[str, Optional[float]]:
    """
    tcapy-style transaction-cost decomposition of executed entries.

    For each entry fill, using the bar at its timestamp:
      - arrival_slippage : fill vs the touch you crossed (ask for a buy / bid for a
                           sell; falls back to the bar open when no quote is present)
      - spread_cost      : the quoted spread in bps (when bid/ask present)
      - best/worst       : slippage had you filled at the bar's two extremes — these
                           bracket the realized arrival slippage
      - transient/permanent impact : fill vs the mid `+offset` bars later (does the
                           price revert = transient, or stay moved = permanent)
    Offsets are in BARS (on 1-min bars: +1 = 1min, +30 = 30min). Pure + offline.
    """
    by_ts: Dict[float, Dict] = {float(b.get("timestamp", 0) or 0): b for b in bars}
    ts_sorted = sorted(by_ts)
    idx_of = {t: i for i, t in enumerate(ts_sorted)}
    arr: List[float] = []
    spr: List[float] = []
    best: List[float] = []
    worst: List[float] = []
    trans: List[float] = []
    perm: List[float] = []

    for t in trades:
        bar = by_ts.get(float(t.entry_ts))
        if bar is None or not t.entry_price or t.entry_price <= 0 or t.qty == 0:
            continue
        side = 1.0 if t.qty > 0 else -1.0
        fill = t.entry_price
        bid, ask = bar.get("bid"), bar.get("ask")
        if bid and ask and ask > bid:
            arrival = ask if side > 0 else bid
            mid = (bid + ask) / 2.0
            spr.append((ask - bid) / mid * 1e4)
        else:
            arrival = bar.get("open") or fill
        if arrival and arrival > 0:
            arr.append((fill - arrival) / arrival * side * 1e4)
            lo, hi = bar.get("low"), bar.get("high")
            if lo and hi:
                s_lo = (lo - arrival) / arrival * side * 1e4
                s_hi = (hi - arrival) / arrival * side * 1e4
                best.append(min(s_lo, s_hi))
                worst.append(max(s_lo, s_hi))
        i = idx_of.get(float(t.entry_ts))
        if i is not None:
            for off, sink in ((transient_offset, trans), (permanent_offset, perm)):
                j = i + off
                if 0 <= j < len(ts_sorted):
                    fb = by_ts[ts_sorted[j]]
                    fb_mid = ((fb["bid"] + fb["ask"]) / 2.0
                              if (fb.get("bid") and fb.get("ask")) else fb.get("close"))
                    if fb_mid and fb_mid > 0:
                        sink.append((fill - fb_mid) / fill * side * 1e4)

    def _agg(xs):
        return (sum(xs) / len(xs)) if xs else None

    return {
        "arrival_slippage_bps": _agg(arr),
        "spread_cost_bps": _agg(spr),
        "best_slippage_bps": _agg(best),
        "worst_slippage_bps": _agg(worst),
        "transient_impact_bps": _agg(trans),
        "permanent_impact_bps": _agg(perm),
        "n_fills_analyzed": len(arr),
    }


def run_parity(
    strategy_name: str,
    bars: List[Dict[str, float]],
    *,
    symbol: str = "DEMO",
    starting_equity: float = 100_000.0,
    commission_bps: float = 1.0,
    bt_slippage_bps: float = 2.0,
    live_slippage_bps: float = 3.5,
    live_slippage_std_bps: float = 1.5,
    sim_seed: int = 0,
) -> ParityReport:
    # A) backtest
    bt = run_backtest(make(strategy_name), bars,
                      starting_equity=starting_equity,
                      commission_bps=commission_bps, slippage_bps=bt_slippage_bps)

    # B) live path through the router+sim adapter
    async def live() -> LiveRunner:
        risk = RiskEngine(RiskConfig(), day_start_equity=starting_equity)
        adapter = SimAdapter(seed=sim_seed, sleep=False,
                             slippage_bps_mean=live_slippage_bps,
                             slippage_bps_std=live_slippage_std_bps)
        router = ExecutionRouter(adapter, risk, None, fill_timeout_s=1.0)
        runner = LiveRunner(make(strategy_name), symbol, router,
                            starting_equity=starting_equity)
        await runner.run(_AsyncList(bars))
        return runner

    runner = asyncio.run(live())
    rep = runner.latency_report()
    entries = sum(1 for o in runner.router.orders
                  if o.side == Side.BUY and o.status.value == "FILLED")

    sig_fill = next((s for s in rep.stages if s.name == "signal->fill"), None)
    tca = decompose_execution(runner.to_result().trades, bars)

    return ParityReport(
        strategy=strategy_name, symbol=symbol, n_bars=len(bars),
        bt_total_return=bt.total_return, bt_n_trades=bt.n_trades,
        bt_slippage_bps=bt_slippage_bps,
        live_total_return=(runner.equity - starting_equity) / starting_equity,
        live_entries=entries, live_fills=runner.stats.fills,
        live_rejects=runner.stats.rejects,
        live_realized_slippage_mean_bps=rep.slippage_bps.mean,
        live_realized_slippage_p95_bps=rep.slippage_bps.p95,
        signal_to_fill_p50_ms=sig_fill.p50 if sig_fill else None,
        signal_to_fill_p95_ms=sig_fill.p95 if sig_fill else None,
        arrival_slippage_bps=tca["arrival_slippage_bps"],
        spread_cost_bps=tca["spread_cost_bps"],
        best_slippage_bps=tca["best_slippage_bps"],
        worst_slippage_bps=tca["worst_slippage_bps"],
        transient_impact_bps=tca["transient_impact_bps"],
        permanent_impact_bps=tca["permanent_impact_bps"],
        n_fills_analyzed=tca["n_fills_analyzed"],
    )


class _AsyncList:
    """Wrap a list as an async iterable for LiveRunner (no feed-latency tracking)."""
    def __init__(self, items): self._items = items
    async def __aiter__(self):
        for it in self._items:
            yield it
