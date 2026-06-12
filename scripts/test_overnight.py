"""
Case study: overnight→intraday cross-sectional REVERSAL on the 195-symbol daily universe.
Judges it the honest way — vs the momentum control (which should fail if the reversal is real),
in-sample vs out-of-sample, a COST sweep (the binding constraint for a book that turns over
~2x/day), and Deflated Sharpe for the project's trial count.

Run: .venv/bin/python scripts/test_overnight.py --cache "/Volumes/My Passport/AlpcaData/cache"
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from alpca.backtest.evaluation import (  # noqa: E402
    deflated_sharpe_ratio, probabilistic_sharpe_ratio, sharpe_of, sharpe_pvalue, sharpe_tstat)
from alpca.backtest.overnight import backtest_overnight_reversal  # noqa: E402

PPY = 252.0


def oos(eq, frac=0.3):
    sp = int(len(eq) * (1 - frac))
    return sharpe_of(eq[:sp], PPY), sharpe_of(eq[sp:], PPY)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/Volumes/My Passport/AlpcaData/cache")
    ap.add_argument("--top-frac", type=float, default=0.2)
    ap.add_argument("--n-trials", type=int, default=36, help="DSR deflation: project search breadth")
    ap.add_argument("--cost-bps", default="0.0,1.0,2.0,5.0",
                    help="per-leg cost sweep; the book turns over ~2x/day so this is the binding test")
    ap.add_argument("--out", default="data/overnight_results.json")
    args = ap.parse_args()
    cache = Path(args.cache)

    bars_by = {}
    for bf in cache.glob("*_1day_bars.jsonl"):
        sym = bf.name.replace("_1day_bars.jsonl", "")
        rows = [json.loads(l) for l in bf.open() if l.strip()]
        if rows and all(k in rows[0] for k in ("open", "close", "timestamp")):
            bars_by[sym] = rows
    print(f"[ok] {len(bars_by)} symbols loaded\n")
    if len(bars_by) < 10:
        print("[fail] universe too small"); return 1

    # ---- signal-lookback sweep, REVERSAL vs MOMENTUM control (2bps) ----
    print(f"{'lookback':>9}{'side':>10}{'sharpe':>8}{'IS':>7}{'OOS':>7}{'ret':>9}{'maxDD':>8}"
          f"{'days':>7}{'sig':>5}")
    print("-" * 70)
    rows = []
    for lb in (1, 2, 3, 5):
        for reverse, side in ((True, "reversal"), (False, "momentum")):
            r = backtest_overnight_reversal(bars_by, signal_lookback=lb, top_frac=args.top_frac,
                                            cost_bps=2.0, reverse=reverse, periods_per_year=PPY)
            if r.n_days < 60:
                continue
            is_sh, oos_sh = oos(r.equity_curve)
            t, pv = sharpe_tstat(r.equity_curve), sharpe_pvalue(r.equity_curve)
            sig = pv < 0.05 and abs(t) > 2.0
            rows.append({"lookback": lb, "side": side, "sharpe": r.sharpe, "is": is_sh,
                         "oos": oos_sh, "ret": r.total_return, "maxdd": r.max_drawdown,
                         "days": r.n_days, "sig": sig})
            print(f"{lb:>9}{side:>10}{r.sharpe:>8.2f}{is_sh:>7.2f}{oos_sh:>7.2f}"
                  f"{r.total_return*100:>8.0f}%{r.max_drawdown*100:>7.1f}%{r.n_days:>7}"
                  f"{('Y' if sig else ''):>5}")

    rev_rows = [x for x in rows if x["side"] == "reversal"]
    best = max(rev_rows, key=lambda x: x["sharpe"]) if rev_rows else None

    # ---- COST sweep on the best reversal config (the binding test) ----
    print("\n" + "=" * 70)
    print(f"COST SWEEP — best reversal (lookback {best['lookback'] if best else '?'}); the book turns "
          f"over ~2x/day so cost is the real test:")
    print(f"{'cost_bps':>9}{'sharpe':>8}{'OOS':>7}{'ret':>9}{'PSR':>7}{'DSR':>7}")
    print("-" * 50)
    pp = [x["sharpe"] / (PPY ** 0.5) for x in rev_rows]
    var_trials = (statistics.pvariance(pp) if len(pp) > 1 else 1e-5) or 1e-5
    cost_rows = []
    for cb in [float(x) for x in args.cost_bps.split(",")]:
        r = backtest_overnight_reversal(bars_by, signal_lookback=best["lookback"] if best else 1,
                                        top_frac=args.top_frac, cost_bps=cb, reverse=True, periods_per_year=PPY)
        _, oos_sh = oos(r.equity_curve)
        psr = probabilistic_sharpe_ratio(r.equity_curve)
        dsr = deflated_sharpe_ratio(r.equity_curve, n_trials=args.n_trials, sharpe_variance=var_trials)
        cost_rows.append({"cost_bps": cb, "sharpe": r.sharpe, "oos": oos_sh, "ret": r.total_return,
                          "psr": psr, "dsr": dsr})
        print(f"{cb:>9.1f}{r.sharpe:>8.2f}{oos_sh:>7.2f}{r.total_return*100:>8.0f}%{psr:>7.2f}{dsr:>7.2f}")

    print("\n" + "=" * 70)
    if best:
        gross = next((x for x in cost_rows if x["cost_bps"] == 0.0), None)
        live = next((x for x in cost_rows if x["cost_bps"] == 2.0), None)
        print(f"VERDICT: best reversal lookback={best['lookback']} — gross (0bps) Sharpe "
              f"{gross['sharpe']:.2f}, IS {best['is']:.2f}/OOS {best['oos']:.2f}.")
        print(f"  At a realistic 2bps/leg: Sharpe {live['sharpe']:.2f}, DSR {live['dsr']:.2f}. "
              f"The momentum control (long winners) should be NEGATIVE if the reversal is real.")
        mom = next((x for x in rows if x["side"] == "momentum" and x["lookback"] == best["lookback"]), None)
        if mom:
            print(f"  Momentum control (lookback {best['lookback']}): Sharpe {mom['sharpe']:.2f} "
                  f"({'CONFIRMS reversal — control is worse' if mom['sharpe'] < best['sharpe'] else 'WARNING: control not worse'}).")

    Path(args.out).write_text(json.dumps({"n_symbols": len(bars_by), "grid": rows,
                                          "cost_sweep": cost_rows}, indent=2))
    print(f"\n[done] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
