"""
Alpaca paper connectivity + REST latency probe.

Proves credentials reach the PAPER account and measures real REST round-trip
latency for account/clock/quote calls. Places NO orders. Prints no secrets.

Usage:
    python scripts/probe_alpaca.py
    python scripts/probe_alpaca.py --env-file /path/to/.env   # load keys from elsewhere
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from alpca.config import load_config  # noqa: E402
from alpca.envfile import load_env_file  # noqa: E402


def _timed(label: str, fn, samples: int = 5):
    times = []
    result = None
    for _ in range(samples):
        t0 = time.perf_counter()
        result = fn()
        times.append((time.perf_counter() - t0) * 1000.0)
    times.sort()
    p50 = statistics.median(times)
    p95 = times[min(len(times) - 1, int(round(0.95 * (len(times) - 1))))]
    print(f"  {label:<22} n={samples}  p50={p50:7.1f}ms  p95={p95:7.1f}ms  "
          f"min={times[0]:7.1f}ms  max={times[-1]:7.1f}ms")
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env-file", help="path to a .env file to load credentials from")
    args = ap.parse_args()

    if args.env_file:
        names = load_env_file(args.env_file)
        # report names only, never values
        shown = [n for n in names if "ALPACA" in n or "APCA" in n]
        print(f"[env] loaded {len(names)} vars from {args.env_file}; "
              f"alpaca-related: {sorted(shown)}")

    cfg = load_config()
    print(cfg.describe())
    try:
        cfg.require_credentials()
    except RuntimeError as e:
        print(f"\n[abort] {e}", file=sys.stderr)
        return 1

    from alpaca.trading.client import TradingClient
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockLatestQuoteRequest

    trading = TradingClient(cfg.api_key, cfg.secret_key, paper=cfg.paper)
    data = StockHistoricalDataClient(cfg.api_key, cfg.secret_key)

    print("\nREST round-trip latency (paper endpoint):")
    acct = _timed("get_account", trading.get_account)
    clock = _timed("get_clock", trading.get_clock)

    spy = None
    try:
        def _quote():
            return data.get_stock_latest_quote(StockLatestQuoteRequest(symbol_or_symbols="SPY"))
        q = _timed("latest_quote(SPY)", _quote)
        try:
            spy = q["SPY"]
        except (KeyError, TypeError):
            spy = None
    except Exception as e:
        print(f"  latest_quote(SPY)      unavailable: {e}")

    print("\nAccount:")
    print(f"  status        = {getattr(acct, 'status', '?')}")
    print(f"  buying_power  = {getattr(acct, 'buying_power', '?')}")
    print(f"  cash          = {getattr(acct, 'cash', '?')}")
    print(f"  equity        = {getattr(acct, 'equity', '?')}")

    print("\nMarket clock:")
    print(f"  is_open       = {getattr(clock, 'is_open', '?')}")
    print(f"  next_open     = {getattr(clock, 'next_open', '?')}")
    print(f"  next_close    = {getattr(clock, 'next_close', '?')}")

    if spy is not None:
        print("\nSPY latest quote:")
        print(f"  bid {getattr(spy, 'bid_price', '?')} x {getattr(spy, 'bid_size', '?')}   "
              f"ask {getattr(spy, 'ask_price', '?')} x {getattr(spy, 'ask_size', '?')}")

    print("\n[ok] Paper connectivity verified. No orders were placed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
