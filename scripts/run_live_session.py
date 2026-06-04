"""
Hardened LIVE PAPER trading session — one regular-hours session, then flat.

Unlike scripts/run_live.py (a bare loop), this is built to run UNATTENDED on a
schedule with strong guardrails:

  * PAPER-only (refuses a non-paper config) and refuses to start when the market
    is closed (exit 2) — same clock-safety as the calibration job.
  * Strict RiskConfig: small per-order notional cap, tight daily-loss auto-halt,
    concentration + orders/min caps. Position size is a small % of equity.
  * RTH-only: the runner only submits during regular hours (require_regular_hours).
  * FLATTEN on startup (clean slate vs any residue) AND on exit, so the session
    never leaves an open position overnight.
  * Wall-clock session DEADLINE (asyncio timeout) so it can never run past the
    close even if the bar feed stalls.
  * --dry-run: swaps the broker for the SimAdapter and replays recent REAL bars,
    so the full wiring (config -> risk -> runner -> fills -> accounting) can be
    verified offline with ZERO real orders. Use this any time, market open or not.

LIVE example (places real PAPER orders during RTH):
  python scripts/run_live_session.py --strategy donchian --symbol SPY
DRY example (no orders, runs anytime):
  python scripts/run_live_session.py --strategy donchian --symbol SPY --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from alpca.config import RiskConfig, load_config  # noqa: E402
from alpca.execution.order_event_log import OrderEventLog  # noqa: E402
from alpca.execution.router import ExecutionRouter  # noqa: E402
from alpca.risk.risk_engine import RiskEngine  # noqa: E402
from alpca.runtime.runner import LiveRunner  # noqa: E402
from alpca.strategies.registry import available, make  # noqa: E402


async def _flatten(adapter, symbol: str) -> bool:
    """Close any open position in `symbol` via Alpaca. Returns True if it acted.
    The SDK raises when there is no position to close — that means already flat."""
    try:
        await asyncio.to_thread(adapter.client.close_position, symbol)
        return True
    except Exception:
        return False  # no position / already flat


async def _run_live(args, cfg) -> int:
    from alpca.data.feed import AlpacaBarPoller
    from alpca.execution.adapters.alpaca import AlpacaAdapter

    adapter = AlpacaAdapter(cfg)
    if not await adapter.is_market_open():
        print("[closed] market is CLOSED — live session refuses to start. "
              "Schedule for regular hours.", file=sys.stderr)
        return 2

    acct = await asyncio.to_thread(adapter.client.get_account)
    equity = float(acct.equity)
    print(f"[account] equity=${equity:,.0f}  market_open=True")

    # clean slate: flatten any residue (e.g. from the calibration job) before we start
    acted = await _flatten(adapter, args.symbol)
    print(f"[startup-flatten] {args.symbol}: {'closed a residual position' if acted else 'already flat'}")

    risk = RiskEngine(_safe_risk(args), day_start_equity=equity)
    log = OrderEventLog(args.ledger)
    router = ExecutionRouter(adapter, risk, log, poll_interval_s=0.25, fill_timeout_s=10.0)
    runner = LiveRunner(_build_strategy(args), args.symbol, router,
                        starting_equity=equity, target_notional_pct=args.notional_pct,
                        require_regular_hours=True)

    source = AlpacaBarPoller(cfg, args.symbol, timeframe=args.timeframe,
                             poll_interval_s=args.poll_interval, max_bars=args.max_bars)

    print(f"[live] {args.strategy} on {args.symbol} ({args.timeframe}); "
          f"deadline {args.session_minutes}min; per-order cap ${args.max_notional:,.0f}; "
          f"pos~{args.notional_pct:.1%} of equity")
    try:
        await asyncio.wait_for(runner.run(source), timeout=args.session_minutes * 60)
    except asyncio.TimeoutError:
        runner.stop()
        print(f"[deadline] {args.session_minutes}min session deadline hit — stopping")
    except KeyboardInterrupt:
        runner.stop()

    # ALWAYS flatten on the way out — never hold overnight
    acted = await _flatten(adapter, args.symbol)
    print(f"[exit-flatten] {args.symbol}: {'flattened' if acted else 'already flat'}")
    _report(runner, log, args)
    return 0


async def _run_dry(args, cfg) -> int:
    from alpca.data.bars import fetch_alpaca_bars
    from alpca.data.feed import ReplayBarSource
    from alpca.execution.adapters.sim import SimAdapter

    print("[dry-run] NO real orders — SimAdapter over recent real bars")
    equity = 100_000.0
    if cfg.has_credentials:
        try:
            from alpca.execution.adapters.alpaca import AlpacaAdapter
            acct = await asyncio.to_thread(AlpacaAdapter(cfg).client.get_account)
            equity = float(acct.equity)
        except Exception as e:
            print(f"[dry-run] could not fetch live equity ({e}); using ${equity:,.0f}")
    try:
        bars = fetch_alpaca_bars(cfg, args.symbol, timeframe=args.timeframe, days=args.dry_days)
    except Exception as e:
        print(f"[dry-run] bar fetch failed: {e}", file=sys.stderr)
        return 1
    print(f"[dry-run] {len(bars)} real {args.timeframe} {args.symbol} bars, equity=${equity:,.0f}")

    risk = RiskEngine(_safe_risk(args), day_start_equity=equity)
    log = OrderEventLog(args.ledger)
    router = ExecutionRouter(SimAdapter.paper_calibrated(), risk, log, fill_timeout_s=2.0)
    runner = LiveRunner(_build_strategy(args), args.symbol, router,
                        starting_equity=equity, target_notional_pct=args.notional_pct,
                        require_regular_hours=True)
    await runner.run(ReplayBarSource(bars))
    _report(runner, log, args)
    return 0


def _safe_risk(args) -> RiskConfig:
    return RiskConfig(
        max_order_notional=args.max_notional,
        daily_loss_pct=args.daily_loss,
        max_concentration_pct=0.05,
        max_open_positions=2,
        max_orders_per_min=10,
        enforce_buying_power=True,
        allow_short=args.allow_short,
    )


def _build_strategy(args):
    """Make the strategy, then optionally wrap it in regime gates (ADX trend-strength
    and/or volatility band) so it sits out ranging 'chop' days that whipsaw breakouts."""
    strat = make(args.strategy)
    if args.adx_filter:
        from alpca.strategies.momentum import ADXTrendGate
        strat = ADXTrendGate(strat, period=args.adx_period, threshold=args.adx_threshold)
    if args.vol_gate:
        from alpca.strategies.momentum import VolRegimeGate
        strat = VolRegimeGate(strat, lookback=args.vol_lookback,
                              vol_floor=args.vol_floor, vol_cap=args.vol_cap)
    return strat


def _report(runner, log, args) -> None:
    print("\n=== SUMMARY ===")
    print(json.dumps(runner.summary(), indent=2, default=str))
    print("\n=== LATENCY ===")
    print(runner.latency_report().render())
    chk = log.verify_chain()
    print(f"\n[ledger] {args.ledger} chain_ok={chk.ok} events={chk.total}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", default="donchian", help=f"one of: {', '.join(available())}")
    ap.add_argument("--symbol", default="SPY")
    ap.add_argument("--timeframe", default="1min")
    ap.add_argument("--poll-interval", type=float, default=5.0)
    ap.add_argument("--max-bars", type=int, default=400, help="hard cap on bars consumed")
    ap.add_argument("--session-minutes", type=float, default=395.0, help="wall-clock deadline")
    ap.add_argument("--notional-pct", type=float, default=0.01, help="position size as fraction of equity")
    ap.add_argument("--max-notional", type=float, default=2500.0, help="per-order notional cap")
    ap.add_argument("--daily-loss", type=float, default=0.01, help="daily-loss auto-halt fraction")
    ap.add_argument("--allow-short", action="store_true")
    # regime gates — only take entries when the market is actually trending/volatile
    ap.add_argument("--adx-filter", action="store_true", help="gate entries by ADX trend strength")
    ap.add_argument("--adx-threshold", type=float, default=25.0, help="min ADX to allow an entry")
    ap.add_argument("--adx-period", type=int, default=14)
    ap.add_argument("--vol-gate", action="store_true", help="gate entries by a realized-vol band")
    ap.add_argument("--vol-lookback", type=int, default=20)
    ap.add_argument("--vol-floor", type=float, default=0.0)
    ap.add_argument("--vol-cap", type=float, default=float("inf"))
    ap.add_argument("--dry-run", action="store_true", help="SimAdapter + replay; no real orders")
    ap.add_argument("--dry-days", type=int, default=2)
    ap.add_argument("--ledger", default="data/live_session_events.jsonl")
    args = ap.parse_args()

    cfg = load_config()
    try:
        cfg.require_credentials()
    except RuntimeError as e:
        print(f"[abort] {e}", file=sys.stderr)
        return 1
    if not cfg.paper:
        print("[abort] refusing to run against a non-PAPER config.", file=sys.stderr)
        return 1
    if args.strategy not in available():
        print(f"[abort] unknown strategy '{args.strategy}'. available: {available()}", file=sys.stderr)
        return 1

    return asyncio.run(_run_dry(args, cfg) if args.dry_run else _run_live(args, cfg))


if __name__ == "__main__":
    raise SystemExit(main())
