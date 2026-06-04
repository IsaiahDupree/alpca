"""
Live Alpaca PAPER smoke test — places ONE tiny real order through the full Alpca
stack (RiskEngine -> ExecutionRouter -> AlpacaAdapter), records the complete
signal->submit->ack->fill lifecycle to the hash-chained ledger, and prints the
latency report. This is the headline demonstration: real execution latency,
measured.

Safety:
  - PAPER only (config refuses live unless explicitly confirmed).
  - Tiny default qty (1 share). Override with --qty / --symbol.
  - Market OPEN  -> marketable LIMIT (ref*1.001): fills, so we get full
    signal->fill latency.
  - Market CLOSED -> resting LIMIT far below market (ref*0.80): accepted but
    won't fill, so we get the real submit->ack round trip; auto-canceled after.

Usage:
  python scripts/smoke_alpaca_paper.py --env-file /path/to/.env
  python scripts/smoke_alpaca_paper.py --symbol SPY --qty 1 --wait 30
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
from alpca.execution.order import Order, OrderType, Side  # noqa: E402
from alpca.execution.order_event_log import OrderEventLog  # noqa: E402
from alpca.execution.router import ExecutionRouter  # noqa: E402
from alpca.risk.risk_engine import RiskEngine  # noqa: E402


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
    is_open = await adapter.is_market_open()
    print(f"[account] equity=${equity:,.0f} buying_power=${float(acct.buying_power):,.0f} "
          f"market_open={is_open}")

    # reference price from latest quote
    ref_price = None
    try:
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockLatestQuoteRequest
        data = StockHistoricalDataClient(cfg.api_key, cfg.secret_key)
        q = await asyncio.to_thread(
            data.get_stock_latest_quote, StockLatestQuoteRequest(symbol_or_symbols=args.symbol))
        quote = q[args.symbol]
        bid, ask = float(quote.bid_price or 0), float(quote.ask_price or 0)
        ref_price = (bid + ask) / 2 if (bid and ask) else (ask or None)
        print(f"[quote] {args.symbol} bid={bid} ask={ask} mid={ref_price}")
    except Exception as e:
        print(f"[quote] unavailable ({e}); proceeding without a ref price")

    log = OrderEventLog(args.ledger)
    risk = RiskEngine(cfg.risk, day_start_equity=equity)
    router = ExecutionRouter(adapter, risk, log,
                             poll_interval_s=0.25, fill_timeout_s=float(args.wait))

    if ref_price and is_open:
        order = Order(symbol=args.symbol, side=Side.BUY, qty=float(args.qty),
                      order_type=OrderType.LIMIT, limit_price=round(ref_price * 1.001, 2),
                      strategy="smoke")
    elif ref_price and not is_open:
        order = Order(symbol=args.symbol, side=Side.BUY, qty=float(args.qty),
                      order_type=OrderType.LIMIT, limit_price=round(ref_price * 0.80, 2),
                      strategy="smoke")
        args.cancel_unfilled = True  # never leave a resting order
    else:
        order = Order(symbol=args.symbol, side=Side.BUY, qty=float(args.qty),
                      order_type=OrderType.MARKET, strategy="smoke")

    order.mark_signal(intended_price=ref_price)
    print(f"[submit] {order.side.value} {order.qty} {order.symbol} "
          f"{order.order_type.value} coid={order.client_order_id}")

    res = await router.submit(order, equity=equity, positions={}, ref_price=ref_price)

    print(f"[result] status={res.status.value} broker_id={res.broker_order_id} "
          f"reject={res.reject_reason}")
    lat = {
        "signal_to_submit_ms": res.signal_to_submit_ms,
        "submit_to_ack_ms": res.submit_to_ack_ms,
        "ack_to_fill_ms": res.ack_to_fill_ms,
        "submit_to_fill_ms": res.submit_to_fill_ms,
        "signal_to_fill_ms": res.signal_to_fill_ms,
        "rtt_ms": res.metadata.get("rtt_ms"),
        "slippage_bps": res.slippage_bps,
        "avg_fill_price": res.avg_fill_price,
        "intended_price": res.intended_price,
    }
    print("[latency] " + json.dumps({k: (round(v, 1) if isinstance(v, float) else v)
                                     for k, v in lat.items()}))

    chk = log.verify_chain()
    print(f"[ledger] {args.ledger} chain_ok={chk.ok} events={chk.total}")
    print("\n" + router.latency_report().render())

    if not res.status.is_terminal and args.cancel_unfilled:
        await adapter.cancel(res)
        print("[cleanup] canceled unfilled order")

    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps({
            "status": res.status.value,
            "broker_order_id": res.broker_order_id,
            "latency_ms": lat,
            "rtt_stats": router.rtt_stats(),
            "ledger_chain_ok": chk.ok,
            "ledger_events": chk.total,
            "market_open": is_open,
        }, indent=2, default=str))
        print(f"[json] wrote {args.json}")

    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env-file")
    ap.add_argument("--symbol", default="SPY")
    ap.add_argument("--qty", default="1")
    ap.add_argument("--wait", default="20", help="seconds to poll for a fill")
    ap.add_argument("--ledger", default="data/order_events.jsonl")
    ap.add_argument("--json")
    ap.add_argument("--cancel-unfilled", action="store_true")
    args = ap.parse_args()
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
