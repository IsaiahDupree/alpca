"""
Run the live paper-trading loop against real Alpaca paper.

Wires: AlpacaAdapter (paper) -> ExecutionRouter (risk + latency + ledger)
       <- AlpacaBarPoller (or websocket) feeding bars to the strategy.

Prints a live latency + slippage report and the run summary. PAPER only (config
refuses live unless explicitly confirmed). Ctrl-C to stop.

Usage:
  python scripts/run_live.py --strategy donchian --symbol SPY --env-file /path/.env
  python scripts/run_live.py --strategy keltner --symbol AAPL --feed websocket --max-bars 30
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from alpca.config import load_config  # noqa: E402
from alpca.envfile import load_env_file  # noqa: E402
from alpca.execution.adapters.alpaca import AlpacaAdapter  # noqa: E402
from alpca.execution.order_event_log import OrderEventLog  # noqa: E402
from alpca.execution.router import ExecutionRouter  # noqa: E402
from alpca.risk.risk_engine import RiskEngine  # noqa: E402
from alpca.runtime.runner import LiveRunner  # noqa: E402
from alpca.strategies.registry import make  # noqa: E402


async def run(args) -> int:
    if args.env_file:
        names = load_env_file(args.env_file)
        shown = sorted(n for n in names if "ALPACA" in n or "APCA" in n)
        print(f"[env] loaded {len(names)} vars; alpaca-related: {shown}")

    cfg = load_config()
    print(cfg.describe())
    try:
        cfg.require_credentials()
    except RuntimeError as e:
        print(f"[abort] {e}", file=sys.stderr)
        return 1

    adapter = AlpacaAdapter(cfg)
    acct = await asyncio.to_thread(adapter.client.get_account)
    equity = float(acct.equity)
    print(f"[account] equity=${equity:,.0f} market_open={await adapter.is_market_open()}")

    log = OrderEventLog(args.ledger)
    risk = RiskEngine(cfg.risk, day_start_equity=equity)
    router = ExecutionRouter(adapter, risk, log, poll_interval_s=0.25, fill_timeout_s=10.0)
    runner = LiveRunner(make(args.strategy), args.symbol, router, starting_equity=equity)

    if args.feed == "websocket":
        from alpca.data.feed import AlpacaWebSocketFeed
        source = AlpacaWebSocketFeed(cfg, [args.symbol], max_bars=args.max_bars)
    else:
        from alpca.data.feed import AlpacaBarPoller
        source = AlpacaBarPoller(cfg, args.symbol, timeframe=args.timeframe,
                                 poll_interval_s=args.poll_interval, max_bars=args.max_bars)

    print(f"[run] {args.strategy} on {args.symbol} via {args.feed} "
          f"(max_bars={args.max_bars}); Ctrl-C to stop")
    try:
        await runner.run(source)
    except KeyboardInterrupt:
        runner.stop()

    print("\n=== SUMMARY ===")
    print(json.dumps(runner.summary(), indent=2, default=str))
    print("\n=== LATENCY ===")
    print(runner.latency_report().render())
    chk = log.verify_chain()
    print(f"\n[ledger] {args.ledger} chain_ok={chk.ok} events={chk.total}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", required=True)
    ap.add_argument("--symbol", default="SPY")
    ap.add_argument("--feed", choices=["poll", "websocket"], default="poll")
    ap.add_argument("--timeframe", default="1min")
    ap.add_argument("--poll-interval", type=float, default=5.0)
    ap.add_argument("--max-bars", type=int, default=30)
    ap.add_argument("--ledger", default="data/order_events.jsonl")
    ap.add_argument("--env-file")
    args = ap.parse_args()
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
