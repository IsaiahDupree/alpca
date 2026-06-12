"""
Case study: cross-sectional GAP REVERSION with a multi-day hold. The question vs Case 17 (which
had a real gross edge but died to ~2x/day turnover): does the gap-reversion family survive once
the hold is stretched so only ~1/hold of the book rotates daily? Judged with a hold sweep, a
gap-MOMENTUM control (should fail if reversion is real), a cost sweep, turnover, OOS, and DSR.

Run: .venv/bin/python scripts/test_gap_reversion.py --cache "/Volumes/My Passport/AlpcaData/cache"
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
from alpca.backtest.gap_reversion import backtest_gap_reversion  # noqa: E402

PPY = 252.0


def oos(eq, frac=0.3):
    sp = int(len(eq) * (1 - frac))
    return sharpe_of(eq[:sp], PPY), sharpe_of(eq[sp:], PPY)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/Volumes/My Passport/AlpcaData/cache")
    ap.add_argument("--max-symbols", type=int, default=195)
    ap.add_argument("--n-trials", type=int, default=39)
    ap.add_argument("--cost-bps", default="0.0,1.0,2.0,5.0")
    ap.add_argument("--out", default="data/gap_reversion_results.json")
    args = ap.parse_args()
    cache = Path(args.cache)

    syms = sorted(p.name.split("_1day_")[0] for p in cache.glob("*_1day_bars.jsonl"))[:args.max_symbols]
    bars_by = {}
    for s in syms:
        p = cache / f"{s}_1day_bars.jsonl"
        rows = [json.loads(l) for l in p.open() if l.strip()]
        if rows and all(key in rows[0] for key in ("open", "close", "timestamp")):
            bars_by[s] = rows
    print(f"[ok] {len(bars_by)} symbols loaded\n")
    if len(bars_by) < 10:
        print("[fail] universe too small"); return 1

    # ---- hold sweep, reversion vs momentum control (2bps) ----
    print(f"{'hold':>5}{'side':>10}{'sharpe':>8}{'IS':>7}{'OOS':>7}{'ret':>9}{'maxDD':>8}"
          f"{'turn/d':>8}{'sig':>5}")
    print("-" * 67)
    rows = []
    for hold in (1, 3, 5, 10, 20):
        for reverse, side in ((True, "reversion"), (False, "momentum")):
            r = backtest_gap_reversion(bars_by, hold=hold, cost_bps=2.0, reverse=reverse, periods_per_year=PPY)
            if r.n_days < 60:
                continue
            is_sh, oos_sh = oos(r.equity_curve)
            t, pv = sharpe_tstat(r.equity_curve), sharpe_pvalue(r.equity_curve)
            sig = pv < 0.05 and abs(t) > 2.0
            rows.append({"hold": hold, "side": side, "sharpe": r.sharpe, "is": is_sh, "oos": oos_sh,
                         "ret": r.total_return, "maxdd": r.max_drawdown, "turnover": r.avg_turnover, "sig": sig})
            print(f"{hold:>5}{side:>10}{r.sharpe:>8.2f}{is_sh:>7.2f}{oos_sh:>7.2f}{r.total_return*100:>8.0f}%"
                  f"{r.max_drawdown*100:>7.1f}%{r.avg_turnover:>8.2f}{('Y' if sig else ''):>5}")

    rev_rows = [x for x in rows if x["side"] == "reversion"]
    best = max(rev_rows, key=lambda x: x["sharpe"]) if rev_rows else None

    # ---- cost sweep on the best reversion hold ----
    print("\n" + "=" * 67)
    print(f"COST SWEEP — best reversion (hold {best['hold'] if best else '?'}, turn/day "
          f"{best['turnover']:.2f}); lower turnover is the only way past Case 17's cost wall:")
    print(f"{'cost_bps':>9}{'sharpe':>8}{'OOS':>7}{'ret':>9}{'PSR':>7}{'DSR':>7}")
    print("-" * 47)
    pp = [x["sharpe"] / (PPY ** 0.5) for x in rev_rows]
    var_trials = (statistics.pvariance(pp) if len(pp) > 1 else 1e-5) or 1e-5
    cost_rows = []
    for cb in [float(x) for x in args.cost_bps.split(",")]:
        r = backtest_gap_reversion(bars_by, hold=best["hold"] if best else 5, cost_bps=cb,
                                   reverse=True, periods_per_year=PPY)
        _, oos_sh = oos(r.equity_curve)
        psr = probabilistic_sharpe_ratio(r.equity_curve)
        dsr = deflated_sharpe_ratio(r.equity_curve, n_trials=args.n_trials, sharpe_variance=var_trials)
        cost_rows.append({"cost_bps": cb, "sharpe": r.sharpe, "oos": oos_sh, "ret": r.total_return,
                          "psr": psr, "dsr": dsr})
        print(f"{cb:>9.1f}{r.sharpe:>8.2f}{oos_sh:>7.2f}{r.total_return*100:>8.0f}%{psr:>7.2f}{dsr:>7.2f}")

    print("\n" + "=" * 67)
    if best:
        gross = next((x for x in cost_rows if x["cost_bps"] == 0.0), None)
        live = next((x for x in cost_rows if x["cost_bps"] == 2.0), None)
        mom = next((x for x in rows if x["side"] == "momentum" and x["hold"] == best["hold"]), None)
        print(f"VERDICT: best reversion hold={best['hold']} — gross {gross['sharpe']:.2f}, "
              f"at 2bps Sharpe {live['sharpe']:.2f}, OOS {live['oos']:.2f}, DSR {live['dsr']:.2f}.")
        if mom:
            print(f"  gap-momentum control (hold {best['hold']}): Sharpe {mom['sharpe']:.2f} "
                  f"({'reversion confirmed' if mom['sharpe'] < best['sharpe'] else 'control NOT worse — weak'}).")
        verdict = ("SURVIVES the cost wall (DSR>0.9 at 2bps) — a real low-turnover gap edge"
                   if live["dsr"] > 0.9 and live["oos"] > 0.3 else
                   "real gross edge but does NOT clear 2bps/DSR — another cost-wall casualty (lower"
                   " turnover than Case 17 but still not enough)" if gross["sharpe"] > 0.3 else
                   "no convincing gap-reversion edge")
        print(f"  -> {verdict}")

    Path(args.out).write_text(json.dumps({"n_symbols": len(bars_by), "grid": rows,
                                          "cost_sweep": cost_rows}, indent=2))
    print(f"\n[done] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
