"""
Case study: 52-week-high momentum (George-Hwang). Judged honestly — hold sweep, the reversal
control (should fail if the near-high effect is real), SPY buy-and-hold (is it alpha or beta?),
a cost sweep, OOS split, per-calendar-year regime stability, and DSR.

Run: .venv/bin/python scripts/test_high_52w.py --cache "/Volumes/My Passport/AlpcaData/cache"
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
    deflated_sharpe_ratio, probabilistic_sharpe_ratio, sharpe_of, sharpe_pvalue, sharpe_tstat, buy_and_hold)
from alpca.backtest.high_52w import backtest_high_52w  # noqa: E402

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
    ap.add_argument("--max-symbols", type=int, default=195)
    ap.add_argument("--n-trials", type=int, default=42)
    ap.add_argument("--cost-bps", default="0.0,1.0,2.0,5.0")
    ap.add_argument("--out", default="data/high_52w_results.json")
    args = ap.parse_args()
    cache = Path(args.cache)
    syms = sorted(p.name.split("_1day_")[0] for p in cache.glob("*_1day_bars.jsonl"))[:args.max_symbols]
    bars = {}
    for s in syms:
        rows = [json.loads(l) for l in (cache / f"{s}_1day_bars.jsonl").open() if l.strip()]
        if rows:
            bars[s] = rows
    print(f"[ok] {len(bars)} symbols\n")
    if len(bars) < 10:
        print("[fail] universe too small"); return 1

    print(f"{'hold':>5}{'side':>10}{'sharpe':>8}{'IS':>7}{'OOS':>7}{'ret':>9}{'maxDD':>8}{'turn/d':>8}{'sig':>5}")
    print("-" * 67)
    rows = []
    for hold in (10, 20, 60, 120):
        for reverse, side in ((False, "near-high"), (True, "reversal")):
            r = backtest_high_52w(bars, window=252, hold=hold, cost_bps=2.0, reverse=reverse, periods_per_year=PPY)
            if r.n_days < 60:
                continue
            is_sh, oos_sh = oos(r.equity_curve)
            t, pv = sharpe_tstat(r.equity_curve), sharpe_pvalue(r.equity_curve)
            sig = pv < 0.05 and abs(t) > 2.0
            rows.append({"hold": hold, "side": side, "sharpe": r.sharpe, "is": is_sh, "oos": oos_sh,
                         "ret": r.total_return, "maxdd": r.max_drawdown, "turnover": r.avg_turnover, "sig": sig})
            print(f"{hold:>5}{side:>10}{r.sharpe:>8.2f}{is_sh:>7.2f}{oos_sh:>7.2f}{r.total_return*100:>8.0f}%"
                  f"{r.max_drawdown*100:>7.1f}%{r.avg_turnover:>8.3f}{('Y' if sig else ''):>5}")

    nh = [x for x in rows if x["side"] == "near-high"]
    best = max(nh, key=lambda x: x["sharpe"]) if nh else None

    # cost sweep + per-year on the best near-high config
    print("\n" + "=" * 67)
    pp = [x["sharpe"] / (PPY ** 0.5) for x in nh]
    var_trials = (statistics.pvariance(pp) if len(pp) > 1 else 1e-5) or 1e-5
    cost_rows = []
    rbest = backtest_high_52w(bars, window=252, hold=best["hold"] if best else 20, cost_bps=2.0, periods_per_year=PPY)
    by = {}
    for ep, x in zip(rbest.dates, rbest.daily_returns):
        by.setdefault(time.gmtime(ep).tm_year, []).append(x)
    yr = {y: round(sharpe_of(_eq(v), PPY), 2) for y, v in by.items() if len(v) >= 30}
    print(f"COST SWEEP — best near-high (hold {best['hold'] if best else 20}, turn/day {rbest.avg_turnover:.3f}):")
    print(f"{'cost_bps':>9}{'sharpe':>8}{'OOS':>7}{'ret':>9}{'PSR':>7}{'DSR':>7}")
    print("-" * 47)
    for cb in [float(x) for x in args.cost_bps.split(",")]:
        r = backtest_high_52w(bars, window=252, hold=best["hold"] if best else 20, cost_bps=cb, periods_per_year=PPY)
        _, oos_sh = oos(r.equity_curve)
        psr = probabilistic_sharpe_ratio(r.equity_curve)
        dsr = deflated_sharpe_ratio(r.equity_curve, n_trials=args.n_trials, sharpe_variance=var_trials)
        cost_rows.append({"cost_bps": cb, "sharpe": r.sharpe, "oos": oos_sh, "ret": r.total_return, "psr": psr, "dsr": dsr})
        print(f"{cb:>9.1f}{r.sharpe:>8.2f}{oos_sh:>7.2f}{r.total_return*100:>8.0f}%{psr:>7.2f}{dsr:>7.2f}")

    # buy-and-hold benchmark (alpha vs beta?)
    spy = cache / "SPY_1day_bars.jsonl"
    bh = None
    if spy.exists():
        bh = buy_and_hold([json.loads(l) for l in spy.open() if l.strip()], PPY)

    print("\n" + "=" * 67)
    if best:
        live = next((x for x in cost_rows if x["cost_bps"] == 2.0), None)
        mom = next((x for x in rows if x["side"] == "reversal" and x["hold"] == best["hold"]), None)
        print(f"VERDICT: best near-high hold={best['hold']} — Sharpe {best['sharpe']:.2f}, at 2bps {live['sharpe']:.2f} "
              f"(OOS {live['oos']:.2f}, DSR {live['dsr']:.2f}).")
        if bh:
            print(f"  vs SPY buy-and-hold Sharpe {bh.sharpe:.2f} — market-neutral L/S so it should be JUDGED ON "
                  f"ITS OWN Sharpe (no beta), but compare: {'beats' if best['sharpe']>bh.sharpe else 'below'} B&H.")
        if mom:
            print(f"  reversal control (hold {best['hold']}): Sharpe {mom['sharpe']:.2f} "
                  f"({'near-high confirmed (control worse)' if mom['sharpe'] < best['sharpe'] else 'control NOT worse — weak'}).")
        print(f"  per-calendar-year (hold {best['hold']}, 2bps): " + ", ".join(f"{y}:{yr[y]:+.2f}" for y in sorted(yr)))
        pos = sum(1 for s in yr.values() if s > 0)
        v = ("survives (DSR>0.9, OOS+, regime-robust) -> candidate for a fresh-symbol holdout"
             if live["dsr"] > 0.9 and live["oos"] > 0.3 and pos >= len(yr) - 1 else
             "positive but not robust enough -> not a validated edge")
        print(f"  -> positive {pos}/{len(yr)} years; {v}")

    Path(args.out).write_text(json.dumps({"n_symbols": len(bars), "grid": rows, "cost_sweep": cost_rows,
                                          "per_year": yr}, indent=2))
    print(f"\n[done] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
