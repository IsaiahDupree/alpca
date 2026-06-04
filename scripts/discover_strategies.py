"""
Strategy discovery sweep — backtest EVERY registered strategy across the cached
symbols/timeframe through the runner-driven backtest (signed positions, real fill
model, per-session reset), then rank by mean return so the profitable ones surface.

The lesson from session-momentum: profitability lives at the right TIMEFRAME with
LOW turnover (1-min overtrades into costs). So sweep daily (trend/momentum shine)
and 1-min (session-anchored shine) separately.

Run:
  .venv/bin/python scripts/discover_strategies.py --timeframe 1day \
      --symbols SPY,QQQ,AAPL,MSFT,NVDA,IWM,TLT --cache "/Volumes/My Passport/AlpcaData/cache"
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from alpca.backtest.runner_backtest import backtest_resting  # noqa: E402
from alpca.strategies.registry import available, make  # noqa: E402


def _load(path: Path):
    return [json.loads(l) for l in path.open() if l.strip()]


def _metric(r, *names, default=0.0):
    for n in names:
        v = getattr(r, n, None)
        if v is not None:
            return v
    return default


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default="SPY,QQQ,AAPL,MSFT,NVDA,IWM,TLT")
    ap.add_argument("--timeframe", default="1day")
    ap.add_argument("--cache", default="data/cache")
    ap.add_argument("--only", default="", help="comma list to restrict strategies")
    args = ap.parse_args()

    cache = Path(args.cache)
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    strategies = [s.strip() for s in args.only.split(",") if s.strip()] or available()

    # load bars per symbol
    bars_by_sym = {}
    for sym in symbols:
        p = cache / f"{sym}_{args.timeframe}_bars.jsonl"
        if p.exists():
            bars_by_sym[sym] = _load(p)
    if not bars_by_sym:
        print(f"[fail] no cached {args.timeframe} bars under {cache}", file=sys.stderr)
        return 1
    print(f"[ok] {len(bars_by_sym)} symbols x {len(strategies)} strategies on {args.timeframe} "
          f"({', '.join(bars_by_sym)})\n")

    agg = {}
    for sym, bars in bars_by_sym.items():
        for name in strategies:
            try:
                r = backtest_resting(make(name), bars, allow_short=name.endswith("-ls"))
            except Exception:
                continue
            ret = _metric(r, "total_return")
            agg.setdefault(name, []).append((ret, _metric(r, "n_trades"),
                                             _metric(r, "max_drawdown"), _metric(r, "win_rate")))

    rows = []
    for name, lst in agg.items():
        rets = [x[0] for x in lst]
        rows.append({
            "name": name,
            "mean_ret": statistics.fmean(rets),
            "median_ret": statistics.median(rets),
            "n_prof": sum(1 for r in rets if r > 0),
            "n_sym": len(lst),
            "mean_trades": statistics.fmean(x[1] for x in lst),
            "mean_dd": statistics.fmean(x[2] for x in lst),
        })
    rows.sort(key=lambda r: -r["mean_ret"])

    print(f"{'strategy':<22}{'mean_ret':>10}{'median':>9}{'prof/n':>8}{'trades':>9}{'maxDD':>9}")
    print("-" * 67)
    for r in rows:
        flag = " *" if r["mean_ret"] > 0 and r["n_prof"] * 2 >= r["n_sym"] else ""
        print(f"{r['name']:<22}{r['mean_ret']*100:>9.1f}%{r['median_ret']*100:>8.1f}%"
              f"{r['n_prof']:>5}/{r['n_sym']:<2}{r['mean_trades']:>9.0f}{r['mean_dd']*100:>8.1f}%{flag}")
    prof = [r["name"] for r in rows if r["mean_ret"] > 0 and r["n_prof"] * 2 >= r["n_sym"]]
    print(f"\n[profitable & consistent] ({len(prof)}): {', '.join(prof) or 'none'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
