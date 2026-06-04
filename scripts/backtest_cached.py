"""
Backtest registered strategies on the LOCAL data cache (scripts/download_data.py).

Bars-only strategies (gap-fade, rsi-mr, breakout, z-score) run on <sym>_<tf>_bars;
microprice* strategies run on <sym>_<tf>_qbars (NBBO merged onto each bar). No
network, no orders — pure offline replay through the same LiveRunner the bot uses.

Usage:
  python scripts/backtest_cached.py --symbol QQQ --timeframe 1min \
      --strategies gap-fade,gap-fade-ls,rsi-mr,microprice
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _load(path: Path):
    if not path.exists():
        return None
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="QQQ")
    ap.add_argument("--timeframe", default="1min")
    ap.add_argument("--strategies", default="gap-fade,gap-fade-ls,rsi-mr,rsi-mr-ls,microprice,microprice-ls")
    ap.add_argument("--cache", default="data/cache")
    args = ap.parse_args()

    from alpca.backtest.runner_backtest import backtest_resting
    from alpca.strategies.registry import make

    cache = Path(args.cache)
    bars = _load(cache / f"{args.symbol}_{args.timeframe}_bars.jsonl")
    qbars = _load(cache / f"{args.symbol}_{args.timeframe}_qbars.jsonl")
    if not bars:
        print(f"[FAIL] no cached bars for {args.symbol} {args.timeframe} — run download_data.py first.")
        return 1
    print(f"{args.symbol} {args.timeframe}: {len(bars)} bars"
          + (f", {len(qbars)} qbars" if qbars else ", no qbars (run download with --quotes)"))
    print(f"{'strategy':<14}{'data':>7}{'trades':>8}{'return%':>10}{'win%':>8}"
          f"{'sharpe/bar':>12}{'maxDD%':>9}{'shorts':>8}")
    print("-" * 76)

    for name in [s.strip() for s in args.strategies.split(",") if s.strip()]:
        needs_q = name.startswith(("microprice", "ofi"))
        data = qbars if needs_q else bars
        if needs_q and not qbars:
            print(f"{name:<14}{'qbars':>7}   (no qbars cached — skip)")
            continue
        allow_short = name.endswith("-ls")
        try:
            res = backtest_resting(make(name), data, allow_short=allow_short)
        except Exception as e:
            print(f"{name:<14} ERROR: {e}")
            continue
        wr = res.win_rate
        sh = res.sharpe
        # count opened shorts via the trades' signed entry qty
        shorts = sum(1 for t in res.trades if t.qty < 0)
        print(f"{name:<14}{'qbars' if needs_q else 'bars':>7}{res.n_trades:>8}"
              f"{res.total_return * 100:>10.2f}"
              f"{('-' if wr is None else f'{wr*100:.0f}'):>8}"
              f"{('-' if sh is None else f'{sh:.3f}'):>12}"
              f"{res.max_drawdown * 100:>9.2f}{shorts:>8}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
