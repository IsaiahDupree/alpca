"""
Offline end-to-end demo — NO credentials required.

Runs the full Alpca pipeline against the SimAdapter:
  synthetic bars -> strategy -> ExecutionRouter (risk + latency timing)
  -> SimAdapter (injected latency + slippage) -> hash-chained ledger
  -> latency + slippage report.

This is the credential-free proof that the measurement pipeline works. Swap
SimAdapter for AlpacaAdapter (see scripts/smoke_alpaca_paper.py) to get the same
report from REAL Alpaca paper fills.

Usage:
    python scripts/demo_offline.py --strategy donchian --bars 500 --seed 1
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from alpca.config import RiskConfig  # noqa: E402
from alpca.backtest.engine import run_backtest  # noqa: E402
from alpca.data.bars import synthetic_bars  # noqa: E402
from alpca.execution.adapters.sim import SimAdapter  # noqa: E402
from alpca.execution.order import Order, OrderType, Side  # noqa: E402
from alpca.execution.order_event_log import OrderEventLog  # noqa: E402
from alpca.execution.router import ExecutionRouter  # noqa: E402
from alpca.risk.risk_engine import RiskEngine  # noqa: E402
from alpca.strategies.base import BUY, EXIT  # noqa: E402
from alpca.strategies.registry import make  # noqa: E402


async def run(args) -> int:
    bars = synthetic_bars(args.symbol, n=args.bars, seed=args.seed, drift=0.0003, vol=0.012)

    # 1) Backtest for the modeled-fill baseline (the slippage the strategy assumes)
    bt = run_backtest(make(args.strategy), bars,
                      commission_bps=args.commission_bps, slippage_bps=args.bt_slippage_bps)
    print("=== BACKTEST (modeled fills) ===")
    print(json.dumps(bt.summary(), indent=2))

    # 2) Replay the SAME strategy through the live execution path (SimAdapter)
    strat = make(args.strategy)
    strat.reset()

    ledger = OrderEventLog(args.ledger)
    risk = RiskEngine(RiskConfig(), day_start_equity=100_000)
    # SimAdapter models real-world slippage worse than the backtest assumed,
    # to demonstrate the backtest-vs-live gap.
    adapter = SimAdapter(seed=args.seed, sleep=args.sleep,
                         submit_latency_ms=6, ack_latency_ms=9, fill_latency_ms=22,
                         slippage_bps_mean=args.live_slippage_bps, slippage_bps_std=1.5)
    router = ExecutionRouter(adapter, risk, ledger, poll_interval_s=0.01, fill_timeout_s=2.0)

    equity = 100_000.0
    in_pos = False
    qty_held = 0.0
    # Size each position at 20% of equity — comfortably under both the 25%
    # concentration cap and the $50k per-order notional cap, so orders pass the
    # RiskEngine and actually fill.
    target_notional_pct = 0.20
    for bar in bars:
        sig = strat.on_bar(bar)
        close = bar["close"]
        ref = sig.price if sig.price is not None else close
        if sig.side == BUY and not in_pos:
            qty = max(1, int((equity * target_notional_pct) / ref))
            o = Order(symbol=args.symbol, side=Side.BUY, qty=qty,
                      order_type=OrderType.MARKET, strategy=strat.name)
            o.mark_signal(intended_price=ref)
            res = await router.submit(o, equity=equity, positions={}, ref_price=ref)
            if res.status.value == "FILLED":
                in_pos, qty_held = True, qty
                equity -= res.avg_fill_price * qty
        elif sig.side == EXIT and in_pos:
            o = Order(symbol=args.symbol, side=Side.SELL, qty=qty_held,
                      order_type=OrderType.MARKET, strategy=strat.name)
            o.mark_signal(intended_price=ref)
            res = await router.submit(o, equity=equity, positions={}, ref_price=ref)
            if res.status.value == "FILLED":
                in_pos, qty_held = False, 0.0
                equity += res.avg_fill_price * res.filled_qty

    print("\n=== LIVE-PATH EXECUTION (SimAdapter: injected latency + slippage) ===")
    print(router.latency_report().render())
    print(f"\nrtt_stats (submit->terminal): {router.rtt_stats()}")

    chk = ledger.verify_chain()
    print(f"\nledger: {args.ledger}  chain_ok={chk.ok}  events={chk.total}")

    # 3) Headline: backtest-assumed vs sim-realized slippage
    rep = router.latency_report()
    print("\n=== BACKTEST vs LIVE SLIPPAGE GAP ===")
    print(f"backtest modeled slippage : {bt.slippage_bps:.2f} bps (assumed)")
    if rep.slippage_bps.mean is not None:
        print(f"live-path realized slippage: {rep.slippage_bps.mean:.2f} bps (mean), "
              f"p95 {rep.slippage_bps.p95:.2f} bps")
        print(f"gap                        : {rep.slippage_bps.mean - bt.slippage_bps:+.2f} bps")
    print("\n(Replace SimAdapter with AlpacaAdapter to populate this from real paper fills.)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", default="donchian")
    ap.add_argument("--symbol", default="DEMO")
    ap.add_argument("--bars", type=int, default=500)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--commission-bps", type=float, default=1.0)
    ap.add_argument("--bt-slippage-bps", type=float, default=2.0)
    ap.add_argument("--live-slippage-bps", type=float, default=3.5)
    ap.add_argument("--ledger", default="data/demo_order_events.jsonl")
    # sleep=True by default so injected latencies actually elapse and the report
    # shows realistic non-zero numbers; --no-sleep for fast CI (latencies ~0).
    ap.add_argument("--no-sleep", dest="sleep", action="store_false")
    ap.set_defaults(sleep=True)
    args = ap.parse_args()
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
