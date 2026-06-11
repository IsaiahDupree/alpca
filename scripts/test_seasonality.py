"""
Harness-test the calendar seasonality sleeves (turn-of-month, pre-FOMC) on SPY/QQQ vs
buy-and-hold. These are LOW-turnover, cash-parked-most-days overlays — their value is being
an EVENT-CLOCK leg (uncorrelated to price strategies) for the combiner, not standalone alpha.
Reports exposure, Sharpe, OOS, and significance.

Run: .venv/bin/python scripts/test_seasonality.py --cache "/Volumes/My Passport/AlpcaData/cache"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from alpca.backtest.evaluation import (  # noqa: E402
    buy_and_hold, sharpe_of, sharpe_pvalue, sharpe_tstat)
from alpca.backtest.seasonality import (  # noqa: E402
    backtest_seasonal, pre_fomc_position, turn_of_month_position)

PPY = 252.0


def oos(eq, frac=0.3):
    sp = int(len(eq) * (1 - frac))
    return sharpe_of(eq[:sp], PPY), sharpe_of(eq[sp:], PPY)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/Volumes/My Passport/AlpcaData/cache")
    ap.add_argument("--symbols", default="SPY,QQQ")
    ap.add_argument("--cost-bps", type=float, default=2.0)
    ap.add_argument("--out", default="data/seasonality_results.json")
    args = ap.parse_args()
    cache = Path(args.cache)
    out = {}

    for sym in args.symbols.split(","):
        p = cache / f"{sym}_1day_bars.jsonl"
        if not p.exists():
            continue
        bars = [json.loads(l) for l in p.open() if l.strip()]
        bh = buy_and_hold(bars, PPY)
        print(f"\n===== {sym} ({len(bars)} bars) =====")
        print(f"  buy-and-hold:   Sharpe {bh.sharpe:.2f}  ret {bh.total_return*100:+.0f}%  (exposure 100%)")
        legs = {}
        for name, pos in (("turn-of-month", turn_of_month_position(bars)),
                          ("pre-FOMC", pre_fomc_position(bars))):
            r = backtest_seasonal(bars, pos, name=name, cost_bps=args.cost_bps, periods_per_year=PPY)
            is_sh, oos_sh = oos(r.equity_curve)
            t, pv = sharpe_tstat(r.equity_curve), sharpe_pvalue(r.equity_curve)
            print(f"  {name:<14} Sharpe {r.sharpe:.2f}  ret {r.total_return*100:+.0f}%  "
                  f"exposure {r.exposure*100:.0f}%  IS {is_sh:.2f} OOS {oos_sh:.2f}  "
                  f"t {t:.2f} p {pv:.3f}  maxDD {r.max_drawdown*100:.1f}%")
            legs[name] = {"sharpe": r.sharpe, "ret": r.total_return, "exposure": r.exposure,
                          "is_sharpe": is_sh, "oos_sharpe": oos_sh, "tstat": t, "pval": pv,
                          "maxdd": r.max_drawdown}
        out[sym] = {"buy_hold_sharpe": bh.sharpe, "legs": legs}

    print("\nNOTE: seasonality is a risk-reduced, time-diversifying OVERLAY (in-market a small "
          "fraction of days). Its job is to be an uncorrelated leg in the combiner, not to beat "
          "B&H standalone. Our cache (~2021-2026) can't test the pre-2011-vs-post claim.")
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"[done] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
