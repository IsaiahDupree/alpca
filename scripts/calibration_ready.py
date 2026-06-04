"""
Calibration readiness preflight — verify EVERYTHING is in place so the live run
just works when the market opens. Safe to run anytime (no orders placed).

Checks, in order:
  1. package imports + venv deps (alpaca-py)
  2. credentials present + PAPER mode
  3. Alpaca account reachable (REST), prints equity + buying power
  4. market status (open/closed + next open) via get_clock
  5. a reference quote for the symbol is available
  6. the offline calibration fitter round-trips (sanity)
  7. write paths are writable

Exit 0 = ready (whether or not the market is open right now). Exit non-zero =
something needs fixing before the live run.

Usage:
  python scripts/calibration_ready.py --env-file <.env> --symbol SPY
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _ok(msg):
    print(f"  [ok]   {msg}")


def _warn(msg):
    print(f"  [warn] {msg}")


def _fail(msg):
    print(f"  [FAIL] {msg}")


async def run(args) -> int:
    problems = 0

    print("== 1. package + deps ==")
    try:
        import alpaca  # noqa
        from alpca.calibration.fit import calibrate  # noqa
        from alpca.calibration.records import CalibrationRecord, CalibrationStore  # noqa
        from alpca.execution.adapters.alpaca import AlpacaAdapter  # noqa
        _ok(f"alpaca-py {getattr(__import__('alpaca'), '__version__', '?')} + alpca calibration import")
    except Exception as e:
        _fail(f"import error: {e}")
        return 1  # nothing else will work

    if args.env_file:
        from alpca.envfile import load_env_file
        try:
            names = load_env_file(args.env_file)
            _ok(f"loaded env from {args.env_file} ({len(names)} vars)")
        except FileNotFoundError:
            _warn(f"--env-file {args.env_file} not found; using Alpca/.env / shell env")

    print("== 2. credentials ==")
    from alpca.config import load_config
    cfg = load_config()
    print("  " + cfg.describe())
    if not cfg.has_credentials:
        _fail("no ALPACA_API_KEY / ALPACA_SECRET_KEY — set them or pass --env-file")
        problems += 1
    elif not cfg.paper:
        _fail("config is NOT in paper mode — refusing (set ALPACA_PAPER=1)")
        problems += 1
    else:
        _ok("paper credentials present")

    if not cfg.has_credentials:
        # can't do live checks; still validate the offline fitter + paths below
        _offline_checks(args)
        print(f"\n[result] NOT READY — fix credentials. ({problems} problem(s))")
        return 1

    from alpca.execution.adapters.alpaca import AlpacaAdapter
    adapter = AlpacaAdapter(cfg)

    print("== 3. account reachable ==")
    try:
        acct = await asyncio.to_thread(adapter.client.get_account)
        _ok(f"account {getattr(acct,'status','?')}: equity=${float(acct.equity):,.0f} "
            f"buying_power=${float(acct.buying_power):,.0f}")
    except Exception as e:
        _fail(f"get_account failed: {e}")
        problems += 1

    print("== 4. market status ==")
    try:
        clock = await asyncio.to_thread(adapter.client.get_clock)
        is_open = bool(getattr(clock, "is_open", False))
        if is_open:
            _ok(f"market is OPEN now (next close {getattr(clock,'next_close','?')}) "
                f"— you can run calibrate_paper.py immediately")
        else:
            _ok(f"market is CLOSED (next open {getattr(clock,'next_open','?')}) "
                f"— schedule the live run for then")
    except Exception as e:
        _fail(f"get_clock failed: {e}")
        problems += 1

    print("== 5. reference quote ==")
    try:
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockLatestQuoteRequest
        data = StockHistoricalDataClient(cfg.api_key, cfg.secret_key)
        q = await asyncio.to_thread(
            data.get_stock_latest_quote, StockLatestQuoteRequest(symbol_or_symbols=args.symbol))
        quote = q[args.symbol]
        bid, ask = float(quote.bid_price or 0), float(quote.ask_price or 0)
        if bid or ask:
            _ok(f"{args.symbol} quote bid={bid} ask={ask}")
        else:
            _warn(f"{args.symbol} quote is empty (normal when closed; will populate at open)")
    except Exception as e:
        _warn(f"quote unavailable ({e}) — often just means market closed")

    _offline_checks(args)

    if problems == 0:
        print("\n[result] READY ✓ — when the market is open, run:\n"
              "  python scripts/run_calibration_pipeline.py --env-file <.env> --symbol "
              f"{args.symbol}")
        return 0
    print(f"\n[result] NOT READY — {problems} problem(s) above.")
    return 1


def _offline_checks(args) -> None:
    print("== 6. offline fitter sanity ==")
    try:
        import math
        from alpca.calibration.fit import calibrate
        from alpca.calibration.records import CalibrationRecord
        recs = [CalibrationRecord("SPY", "BUY", q, 100.0,
                                  100.0 * (1 + (2 + 15 * math.sqrt(q / 1e5)) / 1e4),
                                  bar_volume=1e5, submit_to_ack_ms=240, ack_to_fill_ms=60)
                for q in (10, 100, 1000, 5000)]
        res = calibrate(recs)
        _ok(f"fitter recovers half_spread={res.half_spread_bps}bps "
            f"impact={res.impact_coef_bps}bps (fitted={res.impact_fitted})")
    except Exception as e:
        _fail(f"offline fitter broken: {e}")

    print("== 7. write paths ==")
    try:
        d = Path("data")
        d.mkdir(exist_ok=True)
        probe = d / ".calib_write_probe"
        probe.write_text("ok")
        probe.unlink()
        _ok("data/ is writable")
    except Exception as e:
        _fail(f"data/ not writable: {e}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env-file")
    ap.add_argument("--symbol", default="SPY")
    args = ap.parse_args()
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
