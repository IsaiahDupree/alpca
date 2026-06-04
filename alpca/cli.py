"""
Alpca command-line entry point.

  alpca config [--env-file F]                       config + credential status (no secrets)
  alpca strategies                                  list available strategies
  alpca backtest --strategy orb [--offline]         backtest (real Alpaca bars or synthetic)
  alpca parity   --strategy orb [--offline]         backtest-vs-live execution parity report
  alpca run      --strategy orb [--offline]         run the trading loop (offline sim or real paper)

Credentials are read from the environment / .env. Use --env-file to load them
from another file into THIS process only (never copied, never committed).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys

from alpca.config import load_config


def _maybe_load_env(args) -> None:
    if getattr(args, "env_file", None):
        from alpca.envfile import load_env_file
        names = load_env_file(args.env_file)
        shown = sorted(n for n in names if "ALPACA" in n or "APCA" in n)
        print(f"[env] loaded {len(names)} vars from {args.env_file}; alpaca-related: {shown}")


def cmd_config(args) -> int:
    _maybe_load_env(args)
    cfg = load_config()
    print(cfg.describe())
    print(f"risk: {cfg.risk}")
    if not cfg.has_credentials:
        print("\n[!] No Alpaca credentials. Set ALPACA_API_KEY / ALPACA_SECRET_KEY "
              "(see .env.example) or pass --env-file.", file=sys.stderr)
        return 1
    print("\n[ok] Credentials present. Paper mode." if cfg.paper else "\n[!] LIVE mode selected.")
    return 0


def cmd_strategies(_args) -> int:
    from alpca.strategies.registry import available
    print("available strategies:")
    for name in available():
        print(f"  - {name}")
    return 0


def _load_bars(args, cfg):
    from alpca.data.bars import fetch_alpaca_bars, synthetic_bars
    if args.offline:
        return synthetic_bars(args.symbol, n=args.bars, seed=args.seed), \
            f"synthetic(n={args.bars}, seed={args.seed})"
    cfg.require_credentials()
    bars = fetch_alpaca_bars(cfg, args.symbol, timeframe=args.timeframe, days=args.days)
    return bars, f"alpaca {args.timeframe} x{len(bars)} bars (last {args.days}d)"


def cmd_backtest(args) -> int:
    _maybe_load_env(args)
    from alpca.backtest.engine import run_backtest
    from alpca.strategies.registry import make
    cfg = load_config()
    try:
        bars, src = _load_bars(args, cfg)
    except RuntimeError as e:
        print(f"[abort] {e}\n(use --offline for synthetic bars)", file=sys.stderr)
        return 1
    if not bars:
        print(f"[abort] no bars for {args.symbol}", file=sys.stderr)
        return 1
    res = run_backtest(make(args.strategy), bars,
                       commission_bps=args.commission_bps, slippage_bps=args.slippage_bps)
    print(f"[data] {src}")
    print(json.dumps(res.summary(), indent=2))
    return 0


def cmd_parity(args) -> int:
    _maybe_load_env(args)
    from alpca.backtest.parity import run_parity
    cfg = load_config()
    try:
        bars, src = _load_bars(args, cfg)
    except RuntimeError as e:
        print(f"[abort] {e}\n(use --offline for synthetic bars)", file=sys.stderr)
        return 1
    if not bars:
        print(f"[abort] no bars for {args.symbol}", file=sys.stderr)
        return 1
    rep = run_parity(args.strategy, bars, symbol=args.symbol,
                     bt_slippage_bps=args.slippage_bps,
                     live_slippage_bps=args.live_slippage_bps)
    print(f"[data] {src}")
    print(rep.render())
    return 0


def cmd_run(args) -> int:
    _maybe_load_env(args)
    if not args.offline:
        print("real-paper run lives in scripts/run_live.py "
              "(needs credentials + a market data feed):\n"
              "  python scripts/run_live.py --strategy {s} --symbol {sym} --env-file <.env>"
              .format(s=args.strategy, sym=args.symbol), file=sys.stderr)
        return 2

    # offline: replay synthetic bars through the real runner + sim broker
    from alpca.config import RiskConfig
    from alpca.data.bars import synthetic_bars
    from alpca.data.feed import ReplayBarSource
    from alpca.execution.adapters.sim import SimAdapter
    from alpca.execution.order_event_log import OrderEventLog
    from alpca.execution.router import ExecutionRouter
    from alpca.risk.risk_engine import RiskEngine
    from alpca.runtime.runner import LiveRunner
    from alpca.strategies.registry import make

    async def go():
        bars = synthetic_bars(args.symbol, n=args.bars, seed=args.seed)
        log = OrderEventLog(args.ledger)
        risk = RiskEngine(RiskConfig(), day_start_equity=100_000)
        adapter = SimAdapter(seed=args.seed, sleep=False, slippage_bps_mean=3.5)
        router = ExecutionRouter(adapter, risk, log, fill_timeout_s=1.0)
        runner = LiveRunner(make(args.strategy), args.symbol, router, starting_equity=100_000)
        await runner.run(ReplayBarSource(bars))
        print(json.dumps(runner.summary(), indent=2, default=str))
        print("\n" + runner.latency_report().render())
        print(f"\n[ledger] {args.ledger} chain_ok={log.verify_chain().ok}")
    asyncio.run(go())
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="alpca", description="Alpaca latency-metrics paper-trading bot")
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("config", help="show config + credential status")
    sp.add_argument("--env-file")
    sp.set_defaults(func=cmd_config)

    sp = sub.add_parser("strategies", help="list available strategies")
    sp.set_defaults(func=cmd_strategies)

    def _common(sp):
        sp.add_argument("--strategy", required=True)
        sp.add_argument("--symbol", default="SPY")
        sp.add_argument("--timeframe", default="1hour")
        sp.add_argument("--days", type=int, default=30)
        sp.add_argument("--offline", action="store_true", help="use synthetic bars (no creds)")
        sp.add_argument("--bars", type=int, default=300)
        sp.add_argument("--seed", type=int, default=0)
        sp.add_argument("--commission-bps", type=float, default=1.0)
        sp.add_argument("--slippage-bps", type=float, default=2.0)
        sp.add_argument("--env-file")

    sp = sub.add_parser("backtest", help="run a backtest")
    _common(sp)
    sp.set_defaults(func=cmd_backtest)

    sp = sub.add_parser("parity", help="backtest-vs-live execution parity report")
    _common(sp)
    sp.add_argument("--live-slippage-bps", type=float, default=3.5)
    sp.set_defaults(func=cmd_parity)

    sp = sub.add_parser("run", help="run the trading loop (offline sim, or hint for real paper)")
    _common(sp)
    sp.add_argument("--ledger", default="data/order_events.jsonl")
    sp.set_defaults(func=cmd_run)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
