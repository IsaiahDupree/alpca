"""
Case 41 — THE FACTOR ZOO ON MID-CAPS ("find more like mid-cap value").

Value came alive on mid-caps (0.21, generalizing — Cases 38/40) when it was absent on large-caps. The
obvious question: which OTHER documented premia, dead on our liquid large-caps (Cases 26–34), come back
when we move to less-efficient mid-cap names? This reruns the whole zoo on the expanded mid-cap universe
through the SAME honest bar — full universe + a disjoint fresh-symbol holdout (internal i%3 split) +
per-year regime + cost + DSR + gate. A hit must (a) be positive on the full universe AND (b) keep a
positive holdout (generalize), like mid-cap value did.

Factors: asset_growth, net_issuance, roa, gross_profitability (fundamentals); max, idio_vol,
residual_mom, vol_managed_mom (price). Value + value+mom(.25) included as the known references.

Run: .venv/bin/python scripts/test_midcap_factors.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from alpca.backtest.factor import (  # noqa: E402
    backtest_factor, asset_growth_signal, net_issuance_signal, roa_signal, gross_profitability_signal,
    max_return_signal, idiosyncratic_vol_signal, residual_momentum_signal, vol_managed_momentum_signal)
from alpca.backtest.value import backtest_value_composite  # noqa: E402
from alpca.backtest.evaluation import deflated_sharpe_ratio, sharpe_of  # noqa: E402
from alpca.ai.strategy_gate import falsification_gate  # noqa: E402

PPY = 252.0
# name -> (builder(funds, spy), long_high, family)
FACTORS = {
    "asset_growth":   (lambda f, b: asset_growth_signal(f),          False, "fund"),
    "net_issuance":   (lambda f, b: net_issuance_signal(f),          False, "fund"),
    "roa":            (lambda f, b: roa_signal(f),                   True,  "fund"),
    "gross_profit":   (lambda f, b: gross_profitability_signal(f),   True,  "fund"),
    "max_lottery":    (lambda f, b: max_return_signal(21),           False, "price"),
    "idio_vol":       (lambda f, b: idiosyncratic_vol_signal(b, 120), False, "price"),
    "residual_mom":   (lambda f, b: residual_momentum_signal(b, 120, 21), True, "price"),
    "volmanaged_mom": (lambda f, b: vol_managed_momentum_signal(120, 21, 60), True, "price"),
}


def _load(c):
    return {p.name.split("_1day_")[0]: [json.loads(l) for l in p.open() if l.strip()]
            for p in Path(c).glob("*_1day_bars.jsonl")}


def _load_funds(d):
    d = Path(d)
    return {p.name.replace("_fund.json", ""): json.loads(p.read_text())
            for p in d.glob("*_fund.json")} if d.exists() else {}


def _eq(daily, s=1.0):
    e = [s]
    for x in daily:
        e.append(e[-1] * (1 + x))
    return e


def _metrics(r, n_trials):
    eq = r.equity_curve
    if len(eq) < 60:
        return None
    sp = int(len(eq) * 0.7)
    oos = sharpe_of(eq[sp:], PPY)
    by = {}
    for ep, x in zip(r.dates, r.daily_returns):
        by.setdefault(time.gmtime(ep).tm_year, []).append(x)
    yr = {y: round(sharpe_of(_eq(v), PPY), 2) for y, v in by.items() if len(v) >= 30}
    dsr = deflated_sharpe_ratio(eq, n_trials=n_trials, sharpe_variance=1e-4)
    return {"sharpe": r.sharpe, "oos": oos, "per_year": yr, "turnover": r.avg_turnover, "dsr": dsr}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/Volumes/My Passport/AlpcaData/cache_midcap")
    ap.add_argument("--fundamentals", default="/Volumes/My Passport/AlpcaData/fundamentals_edgar_midcap")
    ap.add_argument("--spy-cache", default="/Volumes/My Passport/AlpcaData/cache")
    ap.add_argument("--n-trials", type=int, default=80)
    ap.add_argument("--out", default="data/midcap_factors_results.json")
    args = ap.parse_args()

    bars = _load(args.cache)
    funds = _load_funds(args.fundamentals)
    spy = next((v for k, v in _load(args.spy_cache).items() if k == "SPY"), None)
    syms_all = sorted(bars)
    train_p = {s: bars[s] for i, s in enumerate(syms_all) if i % 3 != 0}
    hold_p = {s: bars[s] for i, s in enumerate(syms_all) if i % 3 == 0}
    print(f"[ok] mid-cap bars {len(bars)} · funds {len(funds)} · SPY {'yes' if spy else 'NO'}")
    print(f"[split] price universe train {len(train_p)} / holdout {len(hold_p)}\n")

    print(f"{'factor':>15}{'fam':>6}{'full':>7}{'OOS':>7}{'hold':>7}{'+yrs':>6}{'DSR':>6}{'rail':>6}")
    print("-" * 60)
    rows = []

    def run_factor(name, builder, long_high, fam):
        ffull = {s: funds[s] for s in syms_all if s in funds} if fam == "fund" else funds
        ftr = {s: funds[s] for s in train_p if s in funds} if fam == "fund" else funds
        fho = {s: funds[s] for s in hold_p if s in funds} if fam == "fund" else funds
        full_b = {s: bars[s] for s in syms_all}
        rf = backtest_factor(full_b, builder(ffull, spy), name=name, top_frac=0.2, rebalance_days=21,
                             cost_bps=2.0, long_high=long_high, periods_per_year=PPY)
        rh = backtest_factor(hold_p, builder(fho, spy), name=name, top_frac=0.2, rebalance_days=21,
                             cost_bps=2.0, long_high=long_high, periods_per_year=PPY)
        return rf, rh

    items = [(n, b, lh, fam) for n, (b, lh, fam) in FACTORS.items()]
    for name, builder, long_high, fam in items:
        if fam == "fund" and not funds:
            continue
        rf, rh = run_factor(name, builder, long_high, fam)
        m = _metrics(rf, args.n_trials)
        if m is None:
            print(f"{name:>15}  (insufficient)"); continue
        result = {"name": name, "sharpe": round(m["sharpe"], 3), "oos_sharpe": round(m["oos"], 3),
                  "fresh_holdout_sharpe": round(rh.sharpe, 3), "per_year": m["per_year"],
                  "dsr": round(m["dsr"], 3), "cost_2bps_sharpe": round(m["sharpe"], 3),
                  "turnover": round(m["turnover"], 3)}
        g = falsification_gate(result)
        pos = sum(1 for s in m["per_year"].values() if s > 0)
        gen = rh.sharpe > 0
        rows.append({**result, "family": fam, "rail_pass": g.passed, "generalizes": gen})
        print(f"{name:>15}{fam:>6}{m['sharpe']:>7.2f}{m['oos']:>7.2f}{rh.sharpe:>7.2f}"
              f"{pos:>3}/{len(m['per_year']):<2}{m['dsr']:>6.2f}{('PASS' if g.passed else '.'):>6}")

    # value references on the same split
    for label, mw in (("value", 0.0), ("value+mom.25", 0.25)):
        rf = backtest_value_composite({s: bars[s] for s in syms_all},
                                      {s: funds[s] for s in syms_all if s in funds},
                                      top_frac=0.2, rebalance_days=21, cost_bps=2.0,
                                      momentum_weight=mw, periods_per_year=PPY)
        rh = backtest_value_composite(hold_p, {s: funds[s] for s in hold_p if s in funds},
                                      top_frac=0.2, rebalance_days=21, cost_bps=2.0,
                                      momentum_weight=mw, periods_per_year=PPY)
        m = _metrics(rf, args.n_trials)
        if m:
            pos = sum(1 for s in m["per_year"].values() if s > 0)
            rows.append({"name": label, "sharpe": round(m["sharpe"], 3),
                         "fresh_holdout_sharpe": round(rh.sharpe, 3), "dsr": round(m["dsr"], 3),
                         "family": "value", "generalizes": rh.sharpe > 0, "per_year": m["per_year"]})
            print(f"{label:>15}{'val':>6}{m['sharpe']:>7.2f}{m['oos']:>7.2f}{rh.sharpe:>7.2f}"
                  f"{pos:>3}/{len(m['per_year']):<2}{m['dsr']:>6.2f}{'.':>6}")

    gens = [r["name"] for r in rows if r.get("generalizes") and r["sharpe"] > 0.15]
    print(f"\n[generalizing + full>0.15] {gens or 'none'}")
    print(f"[rail survivors] {[r['name'] for r in rows if r.get('rail_pass')] or 'none'}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rows, indent=2))
    print(f"[saved] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
