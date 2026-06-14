"""
Case 47 — the HONEST two-sleeve combined book (pairs basket + mid-cap momentum sleeve).

The combiner has been edge-supply-limited all program. We now have two stress-tested legs measured
uncorrelated (gate #1, ρ=−0.03): the deployed pairs basket (survivorship-stamped, Case 46) and the
borrow-free long/index-hedged mid-cap momentum sleeve (~0.23, Case 45). Gate #1 combined them but with
IN-SAMPLE pairs returns (inflated). This does it HONESTLY:
  - pairs leg = WALK-FORWARD OOS daily returns (delisting_aware_walkforward, deployed config), DATED
  - momentum leg = long/index-hedge daily returns on the mid-cap SIP universe, DATED
date-aligned (inner join), inverse-vol blended, vs the equal-weight null — with the correlation matrix,
per-year regime, drawdown, and DSR. This is the actual deployable two-sleeve number.

Run: .venv/bin/python scripts/test_two_sleeve_combiner.py
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
from alpca.backtest.factor import _price_ret, vol_managed_momentum_signal  # noqa: E402
from alpca.backtest.combine import evaluate_combo  # noqa: E402
from alpca.backtest.evaluation import deflated_sharpe_ratio, max_drawdown_of, sharpe_of  # noqa: E402

PPY = 252.0


def _load(c):
    out = {}
    for p in Path(c).glob("*_1day_bars.jsonl"):
        rows = [json.loads(l) for l in p.open() if l.strip()]
        if rows:
            out[p.name.split("_1day_")[0]] = rows
    return out


def _eq(daily, s=1.0):
    e = [s]
    for x in daily:
        e.append(e[-1] * (1 + x))
    return e


def _per_year(dmap):
    by = {}
    for t, x in dmap.items():
        by.setdefault(time.gmtime(t).tm_year, []).append(x)
    return {y: round(sharpe_of(_eq(v), PPY), 2) for y, v in by.items() if len(v) >= 30}


def momentum_long_hedge_returns(bars, spy_ret, *, top_frac=0.2, rebalance_days=21, cost_bps=2.0,
                                winsor=0.5):
    """Long top-quintile vol-managed-mom winners + short SPY (borrow-free), DATED daily returns."""
    syms = sorted(bars)
    master = sorted({int(b["timestamp"]) for s in syms for b in bars[s]})
    price, ret = _price_ret(bars, syms, master)
    if winsor:
        ret = np.clip(ret, -winsor, winsor)
    signal = vol_managed_momentum_signal(120, 21, 60)(master, syms, price)
    T, N = len(master), len(syms)
    k = max(1, int(round(N * top_frac)))
    spy = np.array([spy_ret.get(t, 0.0) for t in master])
    wl = np.zeros(N); spy_w = 0.0; prev = np.zeros(N)
    out = {}
    for t in range(1, T):
        if (t - 1) % rebalance_days == 0:
            s = signal[t - 1]; ok = np.isfinite(s)
            if ok.sum() >= 2 * k:
                order = np.argsort(np.where(ok, s, -np.inf))
                order = order[np.isin(order, np.where(ok)[0])]
                wl = np.zeros(N); wl[order[-k:]] = 1.0 / k; spy_w = -1.0
        turnover = float(np.abs(wl - prev).sum())
        r = float(np.nansum(wl * np.nan_to_num(ret[t]))) + spy_w * spy[t] - turnover * (cost_bps / 1e4)
        out[master[t]] = r
        prev = wl
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--largecap", default="/Volumes/My Passport/AlpcaData/cache_largecap_sip")
    ap.add_argument("--midcap", default="/Volumes/My Passport/AlpcaData/cache_midcap_sip")
    ap.add_argument("--spy-cache", default="/Volumes/My Passport/AlpcaData/cache")
    ap.add_argument("--out", default="data/two_sleeve_combiner.json")
    args = ap.parse_args()

    lc = _load(args.largecap)
    mc = _load(args.midcap)
    spy_bars = _load(args.spy_cache).get("SPY", [])
    spy_ret = {}
    sb = sorted(spy_bars, key=lambda b: int(b["timestamp"]))
    for i in range(1, len(sb)):
        if float(sb[i - 1]["close"]) > 0:
            spy_ret[int(sb[i]["timestamp"])] = float(sb[i]["close"]) / float(sb[i - 1]["close"]) - 1.0
    print(f"[ok] large-cap {len(lc)} · mid-cap {len(mc)} · SPY {len(spy_ret)} days\n")

    # leg 1: pairs basket WALK-FORWARD OOS (honest), dated
    print("[pairs] running delisting-aware walk-forward (deployed config top-10 / 5% ADF)...")
    pr = delisting_aware_walkforward(lc, train=252, test=63, top_n=10, max_adf=-2.86)
    pairs = {int(t): r for t, r in zip(pr.dates, pr.daily_returns)}
    print(f"        WF Sharpe {pr.sharpe:.2f} · {pr.n_windows} windows · {len(pairs)} OOS days")

    # leg 2: mid-cap momentum long/index-hedge, dated
    mom = momentum_long_hedge_returns(mc, spy_ret)
    print(f"[momentum] long/index-hedge · Sharpe {sharpe_of(_eq(list(mom.values())), PPY):.2f} · {len(mom)} days")

    # date-align
    common = sorted(set(pairs) & set(mom))
    print(f"[joined] {len(common)} common OOS days\n")
    if len(common) < 100:
        print("[abort] too few overlapping days"); return 1
    streams = {"pairs": [pairs[t] for t in common], "momentum": [mom[t] for t in common]}
    rep = evaluate_combo(streams, ppy=PPY)
    rho = rep.corr_matrix[0][1]
    print(f"[CORRELATION] pairs vs momentum  ρ = {rho:+.3f}")
    print(f"[legs] ann-Sharpe { {k: round(v['ann_sharpe'],2) for k,v in rep.legs.items()} }")
    print(f"[combined] equal-weight {rep.equalweight_sharpe:.2f} · inverse-vol {rep.invvol_sharpe:.2f}"
          f" · weights { {k: round(v,2) for k,v in rep.invvol_weights.items()} }")
    w = rep.invvol_weights
    blended = {t: w["pairs"] * pairs[t] + w["momentum"] * mom[t] for t in common}
    yr = _per_year(blended)
    pos = sum(1 for s in yr.values() if s > 0)
    dd = max_drawdown_of(_eq([blended[t] for t in common]))
    dsr = deflated_sharpe_ratio(_eq([blended[t] for t in common]), n_trials=90, sharpe_variance=1e-4)
    best_leg = max(v["ann_sharpe"] for v in rep.legs.values())
    print(f"[combined book] per-year {yr} -> {pos}/{len(yr)} positive · maxDD {dd*100:.1f}% · DSR {dsr:.2f}")
    lift = rep.invvol_sharpe - best_leg
    verdict = (f"DIVERSIFIES — combined {rep.invvol_sharpe:.2f} beats best leg {best_leg:.2f} "
               f"(+{lift:.2f}) at ρ={rho:+.2f}" if lift > 0.03 else
               f"combined {rep.invvol_sharpe:.2f} ≈ best leg {best_leg:.2f} (momentum too thin to lift much)")
    print(f"\n[verdict] {verdict}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({
        "rho": round(rho, 4), "n_common_days": len(common),
        "pairs_wf_sharpe": round(pr.sharpe, 3),
        "leg_ann_sharpe": {k: round(v["ann_sharpe"], 3) for k, v in rep.legs.items()},
        "equalweight_sharpe": round(rep.equalweight_sharpe, 3),
        "invvol_sharpe": round(rep.invvol_sharpe, 3),
        "invvol_weights": {k: round(v, 3) for k, v in rep.invvol_weights.items()},
        "combined_per_year": yr, "combined_pos_years": f"{pos}/{len(yr)}",
        "combined_maxdd": round(dd, 4), "combined_dsr": round(dsr, 3),
        "lift_over_best_leg": round(lift, 3), "verdict": verdict}, indent=2))
    print(f"[saved] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
