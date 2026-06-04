"""
LIVE calibration collector — places tiny round-trip PAPER orders during regular
trading hours and records each real fill (intended mid, actual fill, latencies,
volume) to the calibration store. Run this when the market is OPEN; it does
nothing (and says so) when closed.

Each cycle: read the NBBO mid, BUY `qty` shares marketable, record the buy fill,
then SELL them back, record the sell fill. Spacing several seconds apart and
across a few different sizes gives the fitter the participation range it needs to
separate spread from market impact.

SAFETY:
  - PAPER only (config refuses live).
  - Tiny qty (default 1). A hard per-cycle notional cap aborts if a share would
    cost more than --max-notional.
  - Flattens what it opens; on any error it cancels working orders and tries to
    flatten before exiting.

Usage (Tuesday, market open):
  python scripts/calibrate_paper.py --env-file <.env> --symbol SPY \
      --cycles 20 --sizes 1,2,3 --spacing 5
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from alpca.calibration.records import CalibrationRecord, CalibrationStore  # noqa: E402
from alpca.config import load_config  # noqa: E402
from alpca.envfile import load_env_file  # noqa: E402
from alpca.execution.adapters.alpaca import AlpacaAdapter  # noqa: E402
from alpca.execution.order import Order, OrderType, Side  # noqa: E402
from alpca.execution.order_event_log import OrderEventLog  # noqa: E402
from alpca.execution.router import ExecutionRouter  # noqa: E402
from alpca.risk.risk_engine import RiskEngine  # noqa: E402


async def _quote(cfg, symbol):
    """Return a quote dict {mid, bid, ask, bid_size, ask_size, quote_ts, vol} from
    the latest NBBO + a recent 1-min bar. The full bid/ask/sizes are persisted on
    each CalibrationRecord so the fitter can decompose spread vs market impact."""
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockLatestQuoteRequest, StockLatestBarRequest
    data = StockHistoricalDataClient(cfg.api_key, cfg.secret_key)
    q = await asyncio.to_thread(
        data.get_stock_latest_quote, StockLatestQuoteRequest(symbol_or_symbols=symbol))
    quote = q[symbol]
    bid, ask = float(quote.bid_price or 0), float(quote.ask_price or 0)
    mid = (bid + ask) / 2 if (bid and ask) else (ask or bid or 0.0)
    qts = getattr(quote, "timestamp", None)
    qts = qts.timestamp() if hasattr(qts, "timestamp") else 0.0
    vol = None
    try:
        b = await asyncio.to_thread(
            data.get_stock_latest_bar, StockLatestBarRequest(symbol_or_symbols=symbol,
                                                             feed=cfg.data_feed))
        bar = b.get(symbol) if isinstance(b, dict) else None
        vol = float(getattr(bar, "volume", 0) or 0) if bar else None
    except Exception:
        vol = None
    return {"mid": mid, "bid": bid or None, "ask": ask or None,
            "bid_size": float(getattr(quote, "bid_size", 0) or 0) or None,
            "ask_size": float(getattr(quote, "ask_size", 0) or 0) or None,
            "quote_ts": qts, "vol": vol}


async def _round_trip(router, adapter, cfg, store, symbol, qty, ledger, realized_vol=None):
    """One marketable BUY then SELL; record both fills. Returns (n_recorded)."""
    q0 = await _quote(cfg, symbol)
    if q0["mid"] <= 0:
        print(f"[skip] no quote for {symbol}")
        return 0

    recorded = 0
    acct = await asyncio.to_thread(adapter.client.get_account)
    equity = float(acct.equity)

    for side, lim_mult in ((Side.BUY, 1.002), (Side.SELL, 0.998)):
        # re-quote before each leg so intended price + NBBO are fresh
        q = await _quote(cfg, symbol)
        mid = q["mid"]
        if mid <= 0:
            break
        limit = round(mid * lim_mult, 2)
        order = Order(symbol=symbol, side=side, qty=float(qty),
                      order_type=OrderType.LIMIT, limit_price=limit, strategy="calib")
        order.mark_signal(intended_price=mid)
        res = await router.submit(order, equity=equity, positions={}, ref_price=mid,
                                  fill_timeout_s=15.0)
        if res.status.value == "FILLED" and res.avg_fill_price:
            store.append(CalibrationRecord(
                symbol=symbol, side=side.value, qty=float(qty),
                intended_price=mid, fill_price=float(res.avg_fill_price),
                bar_volume=q["vol"],
                signal_to_submit_ms=res.signal_to_submit_ms,
                submit_to_ack_ms=res.submit_to_ack_ms,
                ack_to_fill_ms=res.ack_to_fill_ms,
                signal_to_fill_ms=res.signal_to_fill_ms,
                ts=time.time(), broker_order_id=res.broker_order_id,
                realized_vol=realized_vol,
                bid=q["bid"], ask=q["ask"], bid_size=q["bid_size"],
                ask_size=q["ask_size"], quote_ts=q["quote_ts"]))
            recorded += 1
            print(f"  [{side.value}] {qty}@{symbol} intended={mid:.4f} "
                  f"fill={res.avg_fill_price:.4f} slip={res.slippage_bps:+.2f}bps "
                  f"sub->ack={res.submit_to_ack_ms}ms ack->fill={res.ack_to_fill_ms}ms")
        else:
            print(f"  [{side.value}] not filled (status={res.status.value}); "
                  f"canceling + continuing")
            if not res.status.is_terminal:
                await adapter.cancel(res)
    return recorded


async def run(args) -> int:
    if args.env_file:
        load_env_file(args.env_file)
    cfg = load_config()
    print(cfg.describe())
    try:
        cfg.require_credentials()
    except RuntimeError as e:
        print(f"[abort] {e}", file=sys.stderr)
        return 1

    adapter = AlpacaAdapter(cfg)
    is_open = await adapter.is_market_open()
    if not is_open and not args.force:
        print("[closed] market is CLOSED — calibration needs live fills. "
              "Re-run during regular hours (or --force to attempt anyway).")
        return 2

    # per-share notional guard
    q0 = await _quote(cfg, args.symbol)
    mid = q0["mid"]
    if mid * max(int(s) for s in args.sizes.split(",")) > args.max_notional:
        print(f"[abort] a max-size order (~${mid * max(int(s) for s in args.sizes.split(',')):.0f}) "
              f"exceeds --max-notional ${args.max_notional}", file=sys.stderr)
        return 1

    # realized vol (annualized) at signal time — captured once (σ is stable over the
    # few minutes of a run) and stamped on every record for the Almgren impact fit.
    realized_vol = None
    try:
        from alpca.calibration.volatility import compute_rolling_volatility
        from alpca.data.bars import fetch_alpaca_bars
        recent = fetch_alpaca_bars(cfg, args.symbol, timeframe="1min", days=12)
        realized_vol = compute_rolling_volatility(recent, lookback_days=10)
        print(f"[vol] realized σ (10d, 1-min, annualized) ≈ {realized_vol:.3f}")
    except Exception as e:
        print(f"[vol] could not compute realized vol ({e}); records will omit σ")

    store = CalibrationStore(args.store)
    ledger = OrderEventLog(args.ledger)
    risk = RiskEngine(cfg.risk)
    router = ExecutionRouter(adapter, risk, ledger, poll_interval_s=0.2, fill_timeout_s=15.0)

    sizes = [int(s) for s in args.sizes.split(",")]
    total = 0
    try:
        for i in range(args.cycles):
            qty = sizes[i % len(sizes)]
            print(f"[cycle {i+1}/{args.cycles}] {qty} {args.symbol}")
            total += await _round_trip(router, adapter, cfg, store, args.symbol, qty,
                                       ledger, realized_vol=realized_vol)
            if i < args.cycles - 1:
                await asyncio.sleep(args.spacing)
    except KeyboardInterrupt:
        print("[interrupt] stopping early")
    finally:
        # safety: cancel any working order + flatten the symbol
        try:
            pos = await asyncio.to_thread(adapter.client.get_all_positions)
            for p in pos:
                if p.symbol == args.symbol:
                    print(f"[cleanup] flattening {p.qty} {p.symbol}")
                    await asyncio.to_thread(adapter.client.close_position, p.symbol)
        except Exception as e:
            print(f"[cleanup] could not flatten ({e}); CHECK THE PAPER ACCOUNT")

    print(f"\n[done] recorded {total} fills -> {args.store} "
          f"(store now has {store.count()} total)")
    print(f"[ledger] {args.ledger} chain_ok={ledger.verify_chain().ok}")
    print("Next: python scripts/run_calibration_pipeline.py --env-file <.env>")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env-file")
    ap.add_argument("--symbol", default="SPY")
    ap.add_argument("--cycles", type=int, default=12)
    ap.add_argument("--sizes", default="1,2,3", help="comma-separated share sizes to vary")
    ap.add_argument("--spacing", type=float, default=5.0, help="seconds between cycles")
    ap.add_argument("--max-notional", type=float, default=5_000.0)
    ap.add_argument("--store", default="data/calibration_fills.jsonl")
    ap.add_argument("--ledger", default="data/calibration_orders.jsonl")
    ap.add_argument("--force", action="store_true", help="run even if market reads closed")
    args = ap.parse_args()
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
