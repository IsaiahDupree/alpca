"""
One-command market-hours calibration pipeline. When the market is open this:

  1. COLLECTS real paper fills (scripts/calibrate_paper.py logic) — tiny
     round-trip orders across a few sizes.
  2. FITS the FillModel coefficients (half-spread + sqrt-impact) and a SimAdapter
     latency preset from those fills.
  3. WRITES data/calibration.json — the calibrated parameters the backtester /
     runner can load.
  4. RUNS A PARITY CHECK: backtests a strategy with the OLD priors vs the NEWLY
     calibrated fill model on the same recent bars and prints the slippage/return
     gap, so you can see how much the calibration moved things.

Safe to run when closed: it stops after step 0 with a clear message (use the
provided cron/launchd snippet to fire it at the open).

Usage (Tuesday, market open):
  python scripts/run_calibration_pipeline.py --env-file <.env> --symbol SPY --cycles 16
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from alpca.calibration.fit import calibrate  # noqa: E402
from alpca.calibration.records import CalibrationStore  # noqa: E402
from alpca.config import load_config  # noqa: E402
from alpca.envfile import load_env_file  # noqa: E402


async def _collect(args, cfg) -> int:
    """Delegate to the collector's round-trip loop. Returns fills recorded."""
    from alpca.execution.adapters.alpaca import AlpacaAdapter
    from alpca.execution.order_event_log import OrderEventLog
    from alpca.execution.router import ExecutionRouter
    from alpca.risk.risk_engine import RiskEngine
    import scripts.calibrate_paper as collector

    adapter = AlpacaAdapter(cfg)
    if not await adapter.is_market_open():
        print("[closed] market is CLOSED — cannot collect live fills now.")
        print("         Schedule this script for the next open (see --print-cron).")
        return -1

    store = CalibrationStore(args.store)
    ledger = OrderEventLog(args.ledger)
    risk = RiskEngine(cfg.risk)
    router = ExecutionRouter(adapter, risk, ledger, poll_interval_s=0.2, fill_timeout_s=15.0)
    sizes = [int(s) for s in args.sizes.split(",")]
    total = 0
    try:
        for i in range(args.cycles):
            qty = sizes[i % len(sizes)]
            print(f"[collect {i+1}/{args.cycles}] {qty} {args.symbol}")
            total += await collector._round_trip(router, adapter, cfg, store,
                                                  args.symbol, qty, ledger)
            if i < args.cycles - 1:
                await asyncio.sleep(args.spacing)
    finally:
        try:
            pos = await asyncio.to_thread(adapter.client.get_all_positions)
            for p in pos:
                if p.symbol == args.symbol:
                    await asyncio.to_thread(adapter.client.close_position, p.symbol)
                    print(f"[cleanup] flattened {p.qty} {p.symbol}")
        except Exception as e:
            print(f"[cleanup] flatten failed ({e}); CHECK THE PAPER ACCOUNT")
    return total


def _parity_against_priors(args, result) -> dict:
    """Backtest with prior priors vs calibrated fill model on recent bars."""
    from alpca.backtest.engine import run_backtest
    from alpca.data.bars import fetch_alpaca_bars
    from alpca.execution.fills import FillModel
    from alpca.strategies.registry import make
    cfg = load_config()
    try:
        bars = fetch_alpaca_bars(cfg, args.symbol, timeframe="1hour", days=20)
    except Exception as e:
        return {"error": f"could not fetch bars for parity: {e}"}
    if not bars:
        return {"error": "no bars for parity"}
    prior_fm = FillModel(half_spread_bps=1.0, impact_coef_bps=8.0,
                         participation_cap=0.10, min_tick=0.01)
    calib_fm = result.to_fill_model()
    prior = run_backtest(make(args.strategy), bars, fill_model=prior_fm, commission_bps=0.0)
    calib = run_backtest(make(args.strategy), bars, fill_model=calib_fm, commission_bps=0.0)
    return {
        "n_bars": len(bars),
        "prior": {"half_spread_bps": prior_fm.half_spread_bps,
                  "impact_coef_bps": prior_fm.impact_coef_bps,
                  "total_return": round(prior.total_return, 4)},
        "calibrated": {"half_spread_bps": calib_fm.half_spread_bps,
                       "impact_coef_bps": calib_fm.impact_coef_bps,
                       "total_return": round(calib.total_return, 4)},
        "return_delta": round(calib.total_return - prior.total_return, 4),
    }


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

    # 1) collect (unless --fit-only, which fits whatever is already stored)
    if not args.fit_only:
        n = await _collect(args, cfg)
        if n < 0:
            return 2  # market closed
        print(f"[collected] {n} new fills")

    # 2) fit
    store = CalibrationStore(args.store)
    records = store.read_all()
    print(f"[fit] over {len(records)} stored fills")
    result = calibrate(records)
    print(json.dumps(result.to_dict(), indent=2, default=str))

    # 3) write calibrated config
    out = result.save(args.out)
    print(f"[write] calibrated parameters -> {out}")

    # 4) parity vs priors
    print("[parity] backtesting prior priors vs calibrated fill model...")
    parity = _parity_against_priors(args, result)
    print(json.dumps(parity, indent=2, default=str))

    print("\n[done] To USE the calibration in a backtest:\n"
          "  import json; from alpca.execution.fills import FillModel\n"
          f"  c = json.load(open('{out}'))\n"
          "  fm = FillModel(half_spread_bps=c['half_spread_bps'],\n"
          "                 impact_coef_bps=c['impact_coef_bps'],\n"
          "                 participation_cap=0.10, min_tick=0.01)\n"
          "  run_backtest(strategy, bars, fill_model=fm)")
    return 0


def _print_cron(args):
    py = str(Path(sys.executable))
    script = str(Path(__file__).resolve())
    envf = args.env_file or "/path/to/.env"
    print("# --- macOS launchd / cron: fire at 09:35 ET on the next trading day ---")
    print("# cron (machine clock must be ET, or adjust): 35 9 * * 2  (Tuesday)")
    print(f"35 9 * * 2  cd {Path(script).parents[1]} && {py} {script} "
          f"--env-file {envf} --symbol {args.symbol} --cycles {args.cycles} >> "
          f"data/calibration_run.log 2>&1")
    print("# or use Alpca's own scheduler / the /schedule skill to run it once at the open.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env-file")
    ap.add_argument("--symbol", default="SPY")
    ap.add_argument("--strategy", default="donchian")
    ap.add_argument("--cycles", type=int, default=16)
    ap.add_argument("--sizes", default="1,2,3")
    ap.add_argument("--spacing", type=float, default=5.0)
    ap.add_argument("--store", default="data/calibration_fills.jsonl")
    ap.add_argument("--ledger", default="data/calibration_orders.jsonl")
    ap.add_argument("--out", default="data/calibration.json")
    ap.add_argument("--fit-only", action="store_true",
                    help="skip collection; fit whatever fills are already stored")
    ap.add_argument("--print-cron", action="store_true",
                    help="print a cron/launchd line to fire this at the open, then exit")
    args = ap.parse_args()
    if args.print_cron:
        _print_cron(args)
        return 0
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
