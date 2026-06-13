"""
Case study: the accruals anomaly (Sloan) on REAL SEC EDGAR fundamentals — our first FUNDAMENTAL,
non-price/non-positioning edge. Long low-accrual (cash-backed) / short high-accrual, dollar-neutral,
annual rebalance (very low turnover). Judged honestly: anomaly vs control, cost sweep, OOS split,
per-calendar-year regime stability (EDGAR's multi-year depth makes this real), and DSR.

Run: .venv/bin/python scripts/test_accruals.py --cache "/Volumes/My Passport/AlpcaData/cache"
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from alpca.backtest.evaluation import (  # noqa: E402
    deflated_sharpe_ratio, probabilistic_sharpe_ratio, sharpe_of, sharpe_pvalue, sharpe_tstat)
from alpca.backtest.accruals import backtest_accruals  # noqa: E402

PPY = 252.0


def oos(eq, frac=0.3):
    sp = int(len(eq) * (1 - frac))
    return sharpe_of(eq[:sp], PPY), sharpe_of(eq[sp:], PPY)


def _eq(d, s=1.0):
    eq = [s]
    for x in d:
        eq.append(eq[-1] * (1 + x))
    return eq


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/Volumes/My Passport/AlpcaData/cache")
    ap.add_argument("--fundamentals", default="/Volumes/My Passport/AlpcaData/fundamentals_edgar")
    ap.add_argument("--n-trials", type=int, default=43)
    ap.add_argument("--cost-bps", default="0.0,1.0,2.0,5.0")
    ap.add_argument("--out", default="data/accruals_results.json")
    args = ap.parse_args()
    cache, fdir = Path(args.cache), Path(args.fundamentals)

    bars_by, fund_by = {}, {}
    for ff in fdir.glob("*_fund.json"):
        sym = ff.name.replace("_fund.json", "")
        bf = cache / f"{sym}_1day_bars.jsonl"
        rows = json.loads(ff.read_text())
        if rows and bf.exists():
            bars_by[sym] = [json.loads(l) for l in bf.open() if l.strip()]
            fund_by[sym] = rows
    print(f"[ok] {len(fund_by)} symbols with fundamentals + bars\n")
    if len(fund_by) < 10:
        print("[fail] too few symbols"); return 1

    # ---- top_frac sweep, anomaly vs control (2bps) ----
    print(f"{'top_frac':>8}{'side':>10}{'sharpe':>8}{'IS':>7}{'OOS':>7}{'ret':>9}{'maxDD':>8}{'turn/d':>8}{'sig':>5}")
    print("-" * 65)
    rows = []
    for tf in (0.1, 0.2, 0.3):
        for reverse, side in ((False, "anomaly"), (True, "control")):
            r = backtest_accruals(bars_by, fund_by, top_frac=tf, reverse=reverse, cost_bps=2.0, periods_per_year=PPY)
            if r.n_days < 60:
                continue
            is_sh, oos_sh = oos(r.equity_curve)
            t, pv = sharpe_tstat(r.equity_curve), sharpe_pvalue(r.equity_curve)
            sig = pv < 0.05 and abs(t) > 2.0
            rows.append({"top_frac": tf, "side": side, "sharpe": r.sharpe, "is": is_sh, "oos": oos_sh,
                         "ret": r.total_return, "maxdd": r.max_drawdown, "turnover": r.avg_turnover, "sig": sig})
            print(f"{tf:>8.1f}{side:>10}{r.sharpe:>8.2f}{is_sh:>7.2f}{oos_sh:>7.2f}{r.total_return*100:>8.0f}%"
                  f"{r.max_drawdown*100:>7.1f}%{r.avg_turnover:>8.3f}{('Y' if sig else ''):>5}")

    anom = [x for x in rows if x["side"] == "anomaly"]
    ctrl = [x for x in rows if x["side"] == "control"]
    best = max(anom, key=lambda x: x["sharpe"]) if anom else None

    # ---- cost sweep + per-year on the best anomaly config ----
    print("\n" + "=" * 65)
    pp = [x["sharpe"] / (PPY ** 0.5) for x in anom]
    var_trials = (statistics.pvariance(pp) if len(pp) > 1 else 1e-5) or 1e-5
    rbest = backtest_accruals(bars_by, fund_by, top_frac=best["top_frac"] if best else 0.2, cost_bps=2.0, periods_per_year=PPY)
    by = {}
    for ep, x in zip(rbest.dates, rbest.daily_returns):
        by.setdefault(time.gmtime(ep).tm_year, []).append(x)
    yr = {y: round(sharpe_of(_eq(v), PPY), 2) for y, v in by.items() if len(v) >= 30}
    print(f"COST SWEEP — best anomaly (top_frac {best['top_frac'] if best else 0.2}, turn/day {rbest.avg_turnover:.3f}):")
    print(f"{'cost_bps':>9}{'sharpe':>8}{'OOS':>7}{'ret':>9}{'PSR':>7}{'DSR':>7}")
    print("-" * 47)
    cost_rows = []
    for cb in [float(x) for x in args.cost_bps.split(",")]:
        r = backtest_accruals(bars_by, fund_by, top_frac=best["top_frac"] if best else 0.2, cost_bps=cb, periods_per_year=PPY)
        _, oos_sh = oos(r.equity_curve)
        psr = probabilistic_sharpe_ratio(r.equity_curve)
        dsr = deflated_sharpe_ratio(r.equity_curve, n_trials=args.n_trials, sharpe_variance=var_trials)
        cost_rows.append({"cost_bps": cb, "sharpe": r.sharpe, "oos": oos_sh, "ret": r.total_return, "psr": psr, "dsr": dsr})
        print(f"{cb:>9.1f}{r.sharpe:>8.2f}{oos_sh:>7.2f}{r.total_return*100:>8.0f}%{psr:>7.2f}{dsr:>7.2f}")

    print("\n" + "=" * 65)
    if best:
        live = next((x for x in cost_rows if x["cost_bps"] == 2.0), None)
        cb = next((x for x in ctrl if x["top_frac"] == best["top_frac"]), None)
        pos = sum(1 for s in yr.values() if s > 0)
        print(f"VERDICT (FUNDAMENTAL edge, EDGAR multi-year): best anomaly top_frac {best['top_frac']} — "
              f"Sharpe {best['sharpe']:.2f}, at 2bps {live['sharpe']:.2f} (OOS {live['oos']:.2f}, DSR {live['dsr']:.2f}), "
              f"turnover {rbest.avg_turnover:.3f}/day.")
        if cb:
            print(f"  control (long high-ACC) Sharpe {cb['sharpe']:.2f} "
                  f"({'anomaly confirmed (control worse)' if cb['sharpe'] < best['sharpe'] else 'control NOT worse — weak'}).")
        print(f"  per-calendar-year: " + ", ".join(f"{y}:{yr[y]:+.2f}" for y in sorted(yr)) + f"  -> +{pos}/{len(yr)} yrs")
        v = ("survives (DSR>0.9, OOS+, regime-robust, low turnover) -> CANDIDATE for the fresh-symbol holdout"
             if live["dsr"] > 0.9 and live["oos"] > 0.3 and pos >= len(yr) - 1 else
             "positive but not robust enough yet" if best["sharpe"] > 0.3 else
             "no convincing accruals edge on this universe/period")
        print(f"  -> {v}")

    Path(args.out).write_text(json.dumps({"n_symbols": len(fund_by), "grid": rows, "cost_sweep": cost_rows,
                                          "per_year": yr}, indent=2))
    print(f"\n[done] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
