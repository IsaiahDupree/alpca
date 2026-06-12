"""
Case study: lead-lag cross-predictability, WALK-FORWARD. The lead-lag map is fit on train and
traded forward on held-out test windows (genuinely OOS). The decisive control is the SHUFFLE
PLACEBO: random leader assignments. If the real map doesn't beat the placebo, the "edge" is
fitted noise. Cost is stressed because the daily signal => high turnover.

Run: .venv/bin/python scripts/test_lead_lag.py --cache "/Volumes/My Passport/AlpcaData/cache"
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
from alpca.backtest.lead_lag import backtest_lead_lag  # noqa: E402

PPY = 252.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/Volumes/My Passport/AlpcaData/cache")
    ap.add_argument("--max-symbols", type=int, default=195)
    ap.add_argument("--n-trials", type=int, default=38)
    ap.add_argument("--cost-bps", default="0.0,1.0,2.0,5.0")
    ap.add_argument("--out", default="data/lead_lag_results.json")
    args = ap.parse_args()
    cache = Path(args.cache)

    syms = sorted(p.name.split("_1day_")[0] for p in cache.glob("*_1day_bars.jsonl"))[:args.max_symbols]
    bars_by = {}
    for s in syms:
        p = cache / f"{s}_1day_bars.jsonl"
        rows = [json.loads(l) for l in p.open() if l.strip()]
        if rows:
            bars_by[s] = rows
    print(f"[ok] {len(bars_by)} symbols loaded\n")
    if len(bars_by) < 10:
        print("[fail] universe too small"); return 1

    # ---- real vs SHUFFLE PLACEBO across leader counts (2bps), walk-forward ----
    print(f"{'n_leaders':>9}{'variant':>10}{'sharpe':>8}{'ret':>9}{'maxDD':>8}{'days':>7}{'wins':>6}{'sig':>5}")
    print("-" * 64)
    rows = []
    for nl in (3, 5, 10):
        for shuf, tag in ((False, "real"), (True, "placebo")):
            r = backtest_lead_lag(bars_by, train=252, test=63, lag=1, n_leaders=nl,
                                  cost_bps=2.0, shuffle_leaders=shuf, periods_per_year=PPY)
            if r.n_days < 60:
                continue
            t, pv = sharpe_tstat(r.equity_curve), sharpe_pvalue(r.equity_curve)
            sig = pv < 0.05 and abs(t) > 2.0
            rows.append({"n_leaders": nl, "variant": tag, "sharpe": r.sharpe, "ret": r.total_return,
                         "maxdd": r.max_drawdown, "days": r.n_days, "windows": r.n_windows, "sig": sig})
            print(f"{nl:>9}{tag:>10}{r.sharpe:>8.2f}{r.total_return*100:>8.0f}%{r.max_drawdown*100:>7.1f}%"
                  f"{r.n_days:>7}{r.n_windows:>6}{('Y' if sig else ''):>5}")

    real_rows = [x for x in rows if x["variant"] == "real"]
    plac_rows = [x for x in rows if x["variant"] == "placebo"]
    best = max(real_rows, key=lambda x: x["sharpe"]) if real_rows else None
    best_plac = max(plac_rows, key=lambda x: x["sharpe"]) if plac_rows else None

    # ---- cost sweep on the best real config ----
    print("\n" + "=" * 64)
    print(f"COST SWEEP — best real config (n_leaders {best['n_leaders'] if best else '?'}); daily signal "
          f"=> high turnover:")
    print(f"{'cost_bps':>9}{'sharpe':>8}{'ret':>9}{'PSR':>7}{'DSR':>7}")
    print("-" * 41)
    pp = [x["sharpe"] / (PPY ** 0.5) for x in real_rows]
    var_trials = (statistics.pvariance(pp) if len(pp) > 1 else 1e-5) or 1e-5
    cost_rows = []
    for cb in [float(x) for x in args.cost_bps.split(",")]:
        r = backtest_lead_lag(bars_by, train=252, test=63, lag=1,
                              n_leaders=best["n_leaders"] if best else 5, cost_bps=cb,
                              shuffle_leaders=False, periods_per_year=PPY)
        psr = probabilistic_sharpe_ratio(r.equity_curve)
        dsr = deflated_sharpe_ratio(r.equity_curve, n_trials=args.n_trials, sharpe_variance=var_trials)
        cost_rows.append({"cost_bps": cb, "sharpe": r.sharpe, "ret": r.total_return, "psr": psr, "dsr": dsr})
        print(f"{cb:>9.1f}{r.sharpe:>8.2f}{r.total_return*100:>8.0f}%{psr:>7.2f}{dsr:>7.2f}")

    print("\n" + "=" * 64)
    if best and best_plac:
        edge = best["sharpe"] - best_plac["sharpe"]
        print(f"VERDICT: best real Sharpe {best['sharpe']:.2f} (n_leaders {best['n_leaders']}) vs best "
              f"placebo {best_plac['sharpe']:.2f} -> real beats placebo by {edge:+.2f}.")
        if edge <= 0.1:
            print("  The real lead-lag map does NOT clear its shuffled placebo -> the structure is fitted "
                  "noise, not information diffusion. REJECT.")
        else:
            live = next((x for x in cost_rows if x["cost_bps"] == 2.0), None)
            print(f"  Real beats placebo. At 2bps the daily-rebalanced edge is Sharpe "
                  f"{live['sharpe']:.2f}, DSR {live['dsr']:.2f} -> {'survives' if live['dsr'] > 0.9 else 'does NOT clear DSR — likely cost/overfit'}.")

    Path(args.out).write_text(json.dumps({"n_symbols": len(bars_by), "grid": rows,
                                          "cost_sweep": cost_rows}, indent=2))
    print(f"\n[done] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
