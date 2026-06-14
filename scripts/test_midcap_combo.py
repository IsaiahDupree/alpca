"""
Case 42 — MULTI-FACTOR MID-CAP COMBINER (chase regime-robustness past the rail).

Case 41 found THREE generalizing mid-cap legs — value, residual momentum, vol-managed momentum — each
real and out-of-sample-positive but each failing the rail on regime-robustness (only 3–4 of 6 years
positive) and DSR. Value and momentum are negatively correlated, so blending them should be MORE
regime-stable than any alone. The decisive test isn't a higher Sharpe — it's whether the blended
stream is positive in MORE years (clears the 60% regime bar) while keeping a positive fresh-symbol
holdout. We measure the cross-leg correlation matrix (the thing the whole combiner lives on), the
inverse-vol combined Sharpe vs the equal-weight null, and — the real metric — the per-year regime
profile of the blend.

Run: .venv/bin/python scripts/test_midcap_combo.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from alpca.backtest.factor import (  # noqa: E402
    backtest_factor, residual_momentum_signal, vol_managed_momentum_signal)
from alpca.backtest.value import backtest_value_composite  # noqa: E402
from alpca.backtest.combine import evaluate_combo  # noqa: E402
from alpca.backtest.evaluation import deflated_sharpe_ratio, sharpe_of  # noqa: E402

PPY = 252.0


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


def _per_year(dates, daily):
    by = {}
    for ep, x in zip(dates, daily):
        by.setdefault(time.gmtime(ep).tm_year, []).append(x)
    return {y: round(sharpe_of(_eq(v), PPY), 2) for y, v in by.items() if len(v) >= 30}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/Volumes/My Passport/AlpcaData/cache_midcap")
    ap.add_argument("--fundamentals", default="/Volumes/My Passport/AlpcaData/fundamentals_edgar_midcap")
    ap.add_argument("--spy-cache", default="/Volumes/My Passport/AlpcaData/cache")
    ap.add_argument("--n-trials", type=int, default=85)
    ap.add_argument("--out", default="data/midcap_combo_results.json")
    args = ap.parse_args()

    bars = _load(args.cache)
    funds = _load_funds(args.fundamentals)
    spy = next((v for k, v in _load(args.spy_cache).items() if k == "SPY"), None)
    fb = {s: funds[s] for s in bars if s in funds}
    print(f"[ok] mid-cap bars {len(bars)} · funds {len(fb)} · SPY {'yes' if spy else 'NO'}\n")

    # daily-return streams for the three generalizing legs on the FULL mid-cap universe (same calendar)
    rv = backtest_value_composite(bars, fb, top_frac=0.2, rebalance_days=21, cost_bps=2.0,
                                  momentum_weight=0.0, periods_per_year=PPY)
    rr = backtest_factor(bars, residual_momentum_signal(spy, 120, 21), name="resid_mom", top_frac=0.2,
                         rebalance_days=21, cost_bps=2.0, long_high=True, periods_per_year=PPY)
    rm = backtest_factor(bars, vol_managed_momentum_signal(120, 21, 60), name="volmom", top_frac=0.2,
                         rebalance_days=21, cost_bps=2.0, long_high=True, periods_per_year=PPY)

    legs = {"value": (rv.dates, rv.daily_returns), "resid_mom": (rr.dates, rr.daily_returns),
            "volmom": (rm.dates, rm.daily_returns)}
    print(f"{'leg':>12}{'sharpe':>8}{'+yrs':>8}")
    print("-" * 28)
    for nm, (d, dl) in legs.items():
        yr = _per_year(d, dl)
        pos = sum(1 for s in yr.values() if s > 0)
        print(f"{nm:>12}{sharpe_of(_eq(dl), PPY):>8.2f}{pos:>5}/{len(yr):<2}")

    # date-align the three streams (inner join on common dates) for an honest blend + per-year
    common = sorted(set(rv.dates) & set(rr.dates) & set(rm.dates))
    idx = {nm: dict(zip(d, dl)) for nm, (d, dl) in legs.items()}
    streams = {nm: [idx[nm][t] for t in common] for nm in legs}
    rep = evaluate_combo(streams, ppy=PPY)

    print(f"\n[correlation matrix]  names={rep.corr_names}")
    for row in rep.corr_matrix:
        print("   " + "  ".join(f"{x:+.2f}" for x in row))
    print(f"[avg |corr|] {rep.avg_abs_corr:.2f}")
    print(f"[combined] equal-weight {rep.equalweight_sharpe:.2f} · inverse-vol {rep.invvol_sharpe:.2f}"
          f" · weights { {k: round(v,2) for k,v in rep.invvol_weights.items()} }")

    # the REAL metric: per-year regime profile of the inverse-vol blend
    iv = rep.invvol_weights
    blended = [sum(iv[nm] * idx[nm][t] for nm in legs) for t in common]
    yr = _per_year(common, blended)
    pos = sum(1 for s in yr.values() if s > 0)
    dsr = deflated_sharpe_ratio(_eq(blended), n_trials=args.n_trials, sharpe_variance=1e-4)
    print(f"\n[BLEND regime] per-year {yr}  ->  {pos}/{len(yr)} positive · DSR {dsr:.2f}")
    regime_ok = pos / max(len(yr), 1) >= 0.6
    dsr_ok = dsr >= 0.9
    print(f"[verdict] regime-robust(>=60%)={regime_ok} · DSR>=0.9={dsr_ok} -> "
          f"{'CLEARS RAIL' if (regime_ok and dsr_ok) else 'still sub-rail'}")

    out = {"legs": {nm: {"sharpe": round(sharpe_of(_eq(dl), PPY), 3),
                         "per_year": _per_year(d, dl)} for nm, (d, dl) in legs.items()},
           "corr_names": rep.corr_names, "corr_matrix": rep.corr_matrix,
           "avg_abs_corr": round(rep.avg_abs_corr, 3),
           "equalweight_sharpe": round(rep.equalweight_sharpe, 3),
           "invvol_sharpe": round(rep.invvol_sharpe, 3),
           "invvol_weights": {k: round(v, 3) for k, v in rep.invvol_weights.items()},
           "blend_per_year": yr, "blend_pos_years": f"{pos}/{len(yr)}", "blend_dsr": round(dsr, 3),
           "clears_rail": bool(regime_ok and dsr_ok)}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\n[saved] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
