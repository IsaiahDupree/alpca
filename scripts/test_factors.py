"""
Batch-test the untested cross-sectional factors through the generic engine + the honest bar:
main universe + a DISJOINT fresh universe (real fresh-symbol holdout) + per-year regime stability +
cost + DSR + the deterministic falsification gate. One row per factor.

Factors (zero new data): asset_growth, net_issuance, roa, max, idiovol, residual_mom, volmanaged_mom.
(gross_profitability needs a Revenues/COGS fetch; short-interest CHANGE uses FINRA — separate scripts.)

Run: .venv/bin/python scripts/test_factors.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from alpca.backtest.factor import (  # noqa: E402
    backtest_factor, asset_growth_signal, net_issuance_signal, roa_signal, max_return_signal,
    idiosyncratic_vol_signal, residual_momentum_signal, vol_managed_momentum_signal)
from alpca.backtest.evaluation import deflated_sharpe_ratio, sharpe_of  # noqa: E402
from alpca.ai.strategy_gate import falsification_gate  # noqa: E402

PPY = 252.0
# name -> (builder(funds, bench), long_high, family)  ; long_high=False means short the high values
FACTORS = {
    "asset_growth":   (lambda f, b: asset_growth_signal(f),        False, "fund"),
    "net_issuance":   (lambda f, b: net_issuance_signal(f),        False, "fund"),
    "roa":            (lambda f, b: roa_signal(f),                 True,  "fund"),
    "max_lottery":    (lambda f, b: max_return_signal(21),         False, "price"),
    "idio_vol":       (lambda f, b: idiosyncratic_vol_signal(b, 120), False, "price"),
    "residual_mom":   (lambda f, b: residual_momentum_signal(b, 120, 21), True, "price"),
    "volmanaged_mom": (lambda f, b: vol_managed_momentum_signal(120, 21, 60), True, "price"),
}


def _load(c):
    return {p.name.split("_1day_")[0]: [json.loads(l) for l in p.open() if l.strip()]
            for p in Path(c).glob("*_1day_bars.jsonl")}


def _load_funds(d):
    d = Path(d)
    return {p.name.replace("_fund.json", ""): json.loads(p.read_text()) for p in d.glob("*_fund.json")} if d.exists() else {}


def _eq(daily, s=1.0):
    e = [s]
    for x in daily:
        e.append(e[-1] * (1 + x))
    return e


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/Volumes/My Passport/AlpcaData/cache")
    ap.add_argument("--cache-fresh", default="/Volumes/My Passport/AlpcaData/cache_fresh")
    ap.add_argument("--fundamentals", default="/Volumes/My Passport/AlpcaData/fundamentals_edgar")
    ap.add_argument("--fundamentals-fresh", default="/Volumes/My Passport/AlpcaData/fundamentals_edgar_fresh")
    ap.add_argument("--n-trials", type=int, default=45)
    ap.add_argument("--out", default="data/factors_results.json")
    args = ap.parse_args()

    main_b, fresh_b = _load(args.cache), _load(args.cache_fresh)
    fm, ff = _load_funds(args.fundamentals), _load_funds(args.fundamentals_fresh)
    spy = main_b.get("SPY")
    print(f"[ok] main {len(main_b)} (+{len(fresh_b)} fresh) · funds {len(fm)} (+{len(ff)})\n")
    print(f"{'factor':>15}{'main':>7}{'OOS':>7}{'fresh':>7}{'+yrs':>6}{'DSR':>6}{'turn':>7}{'rail':>6}")
    print("-" * 62)
    rows = []
    for name, (builder, long_high, fam) in FACTORS.items():
        r = backtest_factor(main_b, builder(fm, spy), name=name, top_frac=0.2, rebalance_days=21,
                            cost_bps=2.0, long_high=long_high, periods_per_year=PPY)
        eq = r.equity_curve
        if len(eq) < 60:
            print(f"{name:>15}  (insufficient data)"); continue
        sp = int(len(eq) * 0.7)
        oos = sharpe_of(eq[sp:], PPY)
        by = {}
        for ep, x in zip(r.dates, r.daily_returns):
            by.setdefault(time.gmtime(ep).tm_year, []).append(x)
        yr = {y: round(sharpe_of(_eq(v), PPY), 2) for y, v in by.items() if len(v) >= 30}
        fr = backtest_factor(fresh_b, builder(ff, spy), name=name, top_frac=0.2, rebalance_days=21,
                             cost_bps=2.0, long_high=long_high, periods_per_year=PPY)
        dsr = deflated_sharpe_ratio(eq, n_trials=args.n_trials, sharpe_variance=1e-4)
        result = {"name": name, "family": fam, "sharpe": round(r.sharpe, 3), "oos_sharpe": round(oos, 3),
                  "fresh_holdout_sharpe": round(fr.sharpe, 3), "per_year": yr, "dsr": round(dsr, 3),
                  "cost_2bps_sharpe": round(r.sharpe, 3), "turnover": round(r.avg_turnover, 3)}
        g = falsification_gate(result)
        pos = sum(1 for s in yr.values() if s > 0)
        rows.append({**result, "rail_pass": g.passed, "reasons": g.reasons})
        print(f"{name:>15}{r.sharpe:>7.2f}{oos:>7.2f}{fr.sharpe:>7.2f}{pos:>3}/{len(yr):<2}{dsr:>6.2f}"
              f"{r.avg_turnover:>7.3f}{('PASS' if g.passed else 'fail'):>6}")

    survivors = [r["name"] for r in rows if r["rail_pass"]]
    print(f"\n[done] rail survivors: {survivors or 'none'}  ({len(rows)} factors tested)")
    Path(args.out).write_text(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
