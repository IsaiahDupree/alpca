"""
SWING deploy — daily-cadence, HOLD-OVERNIGHT live PAPER trading of a RISK-REDUCED basket.

This is the honest deployment of what the truth table found: rsi-mr / supertrend /
ema-momentum / ensemble are statistically significant, stable, lower-drawdown ways to be
LONG the market (risk-reduced BETA, not alpha). They only work on a DAILY/swing horizon
(on 1-min they bleed), so this runner — unlike run_live_session — does NOT flatten at the
close; it holds the position overnight, makes ONE decision per run, and is meant to be run
once a day.

Each run: fetch the recent daily bars, replay each strategy in the basket to get its current
desired direction (long/flat/short), set the target allocation = (mean direction) * --notional
of equity (so it scales smoothly: 2 of 3 strategies long -> ~67% invested), and submit a single
risk-gated market order for the delta vs the current live position. Honest, modest, finite.

  python scripts/run_swing.py --strategies rsi-mr,supertrend,ema-momentum --symbol SPY --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from alpca.config import RiskConfig, load_config  # noqa: E402
from alpca.execution.order import Order, OrderType, Side  # noqa: E402
from alpca.execution.order_event_log import OrderEventLog  # noqa: E402
from alpca.execution.router import ExecutionRouter  # noqa: E402
from alpca.risk.risk_engine import Position, RiskEngine  # noqa: E402
from alpca.strategies.registry import available, make  # noqa: E402


def _direction(strat) -> int:
    side = getattr(strat, "_side", None)
    if side == "SHORT":
        return -1
    if side == "LONG":
        return 1
    if side in ("", None):
        return 1 if getattr(strat, "_in_position", False) else 0
    return 0


def _desired(name: str, bars) -> int:
    strat = make(name)
    for b in bars:
        strat.on_bar(b)
    return _direction(strat)


async def run(args, cfg) -> int:
    from alpca.data.bars import fetch_alpaca_bars
    from alpca.execution.adapters.alpaca import AlpacaAdapter

    names = [s.strip() for s in args.strategies.split(",") if s.strip()]
    bad = [n for n in names if n not in available()]
    if bad:
        print(f"[abort] unknown strategies {bad}", file=sys.stderr)
        return 1

    adapter = AlpacaAdapter(cfg)
    bars = fetch_alpaca_bars(cfg, args.symbol, timeframe="1day", days=args.history_days)
    if len(bars) < 50:
        print(f"[abort] only {len(bars)} daily bars", file=sys.stderr)
        return 1
    price = bars[-1]["close"]

    dirs = {n: _desired(n, bars) for n in names}
    mean_dir = sum(dirs.values()) / len(dirs)
    target_frac = mean_dir * args.notional          # signed fraction of equity
    print(f"[basket] {args.symbol} @ ${price:,.2f}  directions={dirs}  "
          f"mean={mean_dir:+.2f} -> target {target_frac:+.0%} of equity")

    acct = await asyncio.to_thread(adapter.client.get_account)
    equity = float(acct.equity)
    positions = await asyncio.to_thread(adapter.client.get_all_positions)
    cur_qty = 0.0
    for p in positions:
        if p.symbol == args.symbol:
            cur_qty = float(p.qty)
    target_qty = round(target_frac * equity / price)
    delta = target_qty - cur_qty
    print(f"[position] equity ${equity:,.0f}  current {cur_qty:+.0f}  target {target_qty:+.0f}  "
          f"delta {delta:+.0f} sh (held overnight; NOT flattened)")

    if abs(delta) < 1:
        print("[done] already at target — no order.")
        return 0
    if not args.allow_short and target_qty < 0:
        print("[note] target is short but --allow-short off; clamping to flat.")
        delta = -cur_qty
        if abs(delta) < 1:
            return 0

    if args.dry_run:
        print(f"[dry-run] WOULD {'BUY' if delta > 0 else 'SELL'} {abs(delta):.0f} {args.symbol} "
              f"(no order placed).")
        return 0

    if not await adapter.is_market_open():
        print("[closed] market CLOSED — swing order deferred (run during RTH).", file=sys.stderr)
        return 2

    risk = RiskEngine(_safe_risk(args), day_start_equity=equity)
    log = OrderEventLog(args.ledger)
    router = ExecutionRouter(adapter, risk, log, poll_interval_s=0.25, fill_timeout_s=15.0)
    order = Order(symbol=args.symbol, side=Side.BUY if delta > 0 else Side.SELL,
                  qty=abs(delta), order_type=OrderType.MARKET, strategy="swing")
    order.mark_signal(intended_price=price)
    pos_view = {args.symbol: Position(args.symbol, cur_qty, price)} if cur_qty else {}
    res = await router.submit(order, equity=equity, positions=pos_view, ref_price=price, cash=float(acct.cash))
    print(f"[order] {order.side.value} {abs(delta):.0f} -> {res.status.value} "
          f"@ {res.avg_fill_price or '-'}  rtt {res.rtt_ms:.0f}ms" if hasattr(res, "rtt_ms")
          else f"[order] {order.side.value} {abs(delta):.0f} -> {res.status.value}")
    chk = log.verify_chain()
    print(f"[ledger] {args.ledger} chain_ok={chk.ok}")
    return 0


def _safe_risk(args) -> RiskConfig:
    return RiskConfig(max_order_notional=args.max_notional, daily_loss_pct=0.05,
                      max_concentration_pct=1.0, max_open_positions=5,
                      max_orders_per_min=10, enforce_buying_power=True, allow_short=args.allow_short)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategies", default="rsi-mr,supertrend,ema-momentum")
    ap.add_argument("--symbol", default="SPY")
    ap.add_argument("--notional", type=float, default=0.5, help="max fraction of equity when basket is unanimous")
    ap.add_argument("--max-notional", type=float, default=60_000.0, help="per-order notional cap")
    ap.add_argument("--history-days", type=int, default=250)
    ap.add_argument("--allow-short", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--ledger", default="data/swing_events.jsonl")
    args = ap.parse_args()

    cfg = load_config()
    try:
        cfg.require_credentials()
    except RuntimeError as e:
        print(f"[abort] {e}", file=sys.stderr)
        return 1
    if not cfg.paper:
        print("[abort] refusing non-PAPER config.", file=sys.stderr)
        return 1
    return asyncio.run(run(args, cfg))


if __name__ == "__main__":
    raise SystemExit(main())
