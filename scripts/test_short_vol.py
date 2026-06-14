"""
Case 49 — Short-volatility / variance-risk-premium (VRP) as a second-leg candidate.

The VRP (implied vol > realized vol on average → selling vol pays) is one of the most documented premia
and a GENUINELY different signal source from equity stat-arb (pairs) and trend (momentum) — so a priori
low correlation. The catch is the well-known TAIL TRAP: short-vol crashes in market stress (Feb-2018,
Mar-2020, 2022), and in stress the pairs spreads ALSO blow out, so the two can be correlated exactly when
it hurts. This tests it honestly: the borrow-free long-SVXY (short-vol ETF) sleeve AND a short-VXX sleeve,
with the full bar PLUS the two things that actually decide short-vol:
  (1) per-year incl. 2022 (the vol-spike year) and maxDD (the tail),
  (2) correlation with pairs over the OOS window AND TAIL correlation (on pairs' worst days),
  (3) the partial-2026 split (Case-48 lesson), and the honest combiner lift.

Run: .venv/bin/python scripts/test_short_vol.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from alpca.backtest.pairs import delisting_aware_walkforward  # noqa: E402
from alpca.backtest.combine import evaluate_combo, correlation  # noqa: E402
from alpca.backtest.evaluation import deflated_sharpe_ratio, max_drawdown_of, sharpe_of  # noqa: E402

PPY = 252.0


def _load(c):
    return {p.name.split("_1day_")[0]: [json.loads(l) for l in p.open() if l.strip()]
            for p in Path(c).glob("*_1day_bars.jsonl")}


def _eq(daily, s=1.0):
    e = [s]
    for x in daily:
        e.append(e[-1] * (1 + x))
    return e


def _ret_by_ts(bars):
    b = sorted(bars, key=lambda x: int(x["timestamp"]))
    out = {}
    for i in range(1, len(b)):
        p0 = float(b[i - 1]["close"])
        if p0 > 0:
            out[int(b[i]["timestamp"])] = float(b[i]["close"]) / p0 - 1.0
    return out


def _per_year(dmap):
    by = {}
    for t, x in dmap.items():
        by.setdefault(time.gmtime(t).tm_year, []).append(x)
    return {y: round(sharpe_of(_eq(v), PPY), 2) for y, v in sorted(by.items()) if len(v) >= 20}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vol-cache", default="/Volumes/My Passport/AlpcaData/cache_vol")
    ap.add_argument("--largecap", default="/Volumes/My Passport/AlpcaData/cache_largecap_sip")
    ap.add_argument("--out", default="data/short_vol_results.json")
    args = ap.parse_args()

    vol = _load(args.vol_cache)
    # short-vol sleeves: long SVXY (borrow-free ETF) and short VXX (= -VXX return)
    svxy = _ret_by_ts(vol["SVXY"]) if "SVXY" in vol else {}
    vxx = _ret_by_ts(vol["VXX"]) if "VXX" in vol else {}
    short_vxx = {t: -r for t, r in vxx.items()}
    sleeves = {"long_SVXY": svxy, "short_VXX": short_vxx}

    print("[short-vol sleeves] standalone honesty:")
    print(f"{'sleeve':>12}{'sharpe':>8}{'2022':>8}{'maxDD':>8}{'+yrs':>7}")
    print("-" * 43)
    out = {}
    for name, s in sleeves.items():
        if not s:
            continue
        ts = sorted(s)
        eq = _eq([s[t] for t in ts])
        yr = _per_year(s)
        pos = sum(1 for v in yr.values() if v > 0)
        out[name] = {"sharpe": round(sharpe_of(eq, PPY), 3), "maxdd": round(max_drawdown_of(eq), 3),
                     "per_year": yr, "y2022": yr.get(2022)}
        print(f"{name:>12}{sharpe_of(eq, PPY):>8.2f}{(yr.get(2022) or 0):>8.2f}"
              f"{max_drawdown_of(eq)*100:>7.1f}%{pos:>4}/{len(yr):<2}")

    # pairs WF OOS returns for the combiner + correlation
    print("\n[pairs] running delisting-aware walk-forward...")
    pr = delisting_aware_walkforward(_load(args.largecap), train=252, test=63, top_n=10, max_adf=-2.86)
    pairs = {int(t): r for t, r in zip(pr.dates, pr.daily_returns)}
    print(f"        WF {pr.sharpe:.2f} · {len(pairs)} OOS days")

    best = "long_SVXY" if out.get("long_SVXY") else "short_VXX"
    sv = sleeves[best]
    common = sorted(set(pairs) & set(sv))
    pv = [pairs[t] for t in common]; vv = [sv[t] for t in common]
    rho = correlation(pv, vv)
    # TAIL correlation: on the worst 10% of pairs days, does short-vol also lose?
    thr = np.percentile(pv, 10)
    tail_idx = [i for i in range(len(pv)) if pv[i] <= thr]
    tail_vol_mean = float(np.mean([vv[i] for i in tail_idx])) if tail_idx else 0.0
    tail_rho = correlation([pv[i] for i in tail_idx], [vv[i] for i in tail_idx]) if len(tail_idx) > 3 else 0.0
    print(f"\n[correlation] pairs vs {best}: full ρ={rho:+.3f} · "
          f"TAIL (pairs worst-10% days) short-vol mean {tail_vol_mean*100:+.2f}%/day, tail ρ={tail_rho:+.2f}")

    # honest combiner + partial-2026 split
    rep = evaluate_combo({"pairs": pv, best: vv}, ppy=PPY)
    w = rep.invvol_weights
    blend = {t: w["pairs"] * pairs[t] + w[best] * sv[t] for t in common}
    blend_no26 = {t: v for t, v in blend.items() if time.gmtime(t).tm_year < 2026}
    pairs_no26 = [pairs[t] for t in common if time.gmtime(t).tm_year < 2026]
    yr = _per_year(blend)
    pos = sum(1 for v in yr.values() if v > 0)
    print(f"[combined] inverse-vol {rep.invvol_sharpe:.2f} vs pairs-alone {sharpe_of(_eq(pv), PPY):.2f} "
          f"· per-year {yr} ({pos}/{len(yr)})")
    print(f"[ex-2026 check] combined {sharpe_of(_eq(list(blend_no26.values())), PPY):.2f} "
          f"vs pairs {sharpe_of(_eq(pairs_no26), PPY):.2f}")
    lift = rep.invvol_sharpe - sharpe_of(_eq(pv), PPY)
    robust = out[best]["per_year"]
    pos_recent = (robust.get(2024, 0) > 0) and (robust.get(2025, 0) > 0)
    verdict = ("CANDIDATE — lifts, low tail-ρ, robust recent" if (lift > 0.03 and abs(tail_rho) < 0.4 and pos_recent)
               else f"reject (lift={lift:+.2f}, tail_ρ={tail_rho:+.2f}, 2022={out[best].get('y2022')}, recent_pos={pos_recent})")
    print(f"\n[verdict] {verdict}")
    out["_combined"] = {"leg": best, "rho": round(rho, 4), "tail_rho": round(tail_rho, 3),
                        "tail_vol_mean_pct": round(tail_vol_mean * 100, 3),
                        "invvol_sharpe": round(rep.invvol_sharpe, 3),
                        "pairs_alone": round(sharpe_of(_eq(pv), PPY), 3), "lift": round(lift, 3),
                        "combined_per_year": yr, "verdict": verdict}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"[saved] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
