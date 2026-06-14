"""
Gate #1 — is the mid-cap value+momentum edge UNCORRELATED with the deployed large-cap pairs basket?

The mid-cap blend (Case 42) is a genuine second-edge candidate, but it only EARNS a place in the
master combiner if it diversifies the edge we already trade. Mechanistically they share nothing — the
pairs basket is statistical mean-reversion of cointegrated price residuals on LARGE-caps; the mid-cap
blend is cross-sectional fundamental cheapness + trend on a DISJOINT mid-cap universe — so the prior is
rho ~ 0. But "prior" isn't "measured." This builds both daily-return streams on the SAME calendar,
date-joins them, and reports the correlation + the two-sleeve inverse-vol book.

The pairs leg here is a single full-calendar screen (NOT the walk-forward) — fine for a CORRELATION
estimate, which depends on daily-P&L co-movement, not on the walk-forward honesty that the SHARPE needs.

Run: .venv/bin/python scripts/test_pairs_midcap_corr.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from alpca.backtest.pairs import screen_pairs, backtest_pairs, align  # noqa: E402
from alpca.backtest.factor import backtest_factor, vol_managed_momentum_signal  # noqa: E402
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


def _per_year(dmap):
    by = {}
    for t, x in dmap.items():
        by.setdefault(time.gmtime(t).tm_year, []).append(x)
    return {y: round(sharpe_of(_eq(v), PPY), 2) for y, v in by.items() if len(v) >= 30}


def pairs_basket_dated(bars, *, top_n=10, max_adf=-2.86, cost_bps=2.0):
    """Equal-weight basket of the top cointegrated pairs (validated config) -> {ts: daily_return}."""
    syms = sorted(bars)
    screened = screen_pairs(syms, bars, min_overlap=200, max_half_life=30.0, min_half_life=3.0,
                            max_adf=max_adf)
    per_pair = []   # list of {ts: ret}
    for r in screened[:top_n]:
        a_b, b_b = bars[r["a"]], bars[r["b"]]
        ts = [t for t, _, _ in align(a_b, b_b)]
        lb = int(max(20, min(120, r["half_life"] * 3)))
        res = backtest_pairs(a_b, b_b, lookback=lb, entry_z=2.0, exit_z=0.5, cost_bps=cost_bps,
                             hedge=r["hedge"])
        eq = res.equity_curve
        if len(eq) >= len(ts) and len(ts) > 1:
            rets = {ts[i]: (eq[i] / eq[i - 1] - 1.0) for i in range(1, len(ts)) if eq[i - 1] > 0}
            per_pair.append(rets)
    if not per_pair:
        return {}, 0
    all_ts = sorted(set().union(*[set(p) for p in per_pair]))
    basket = {}
    for t in all_ts:
        vals = [p[t] for p in per_pair if t in p]
        if vals:
            basket[t] = sum(vals) / len(vals)
    return basket, len(per_pair)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--largecap", default="/Volumes/My Passport/AlpcaData/cache")
    ap.add_argument("--midcap", default="/Volumes/My Passport/AlpcaData/cache_midcap")
    ap.add_argument("--mid-funds", default="/Volumes/My Passport/AlpcaData/fundamentals_edgar_midcap")
    ap.add_argument("--out", default="data/pairs_midcap_corr.json")
    args = ap.parse_args()

    lc = _load(args.largecap)
    mc = _load(args.midcap)
    mf = _load_funds(args.mid_funds)
    spy = lc.get("SPY")
    print(f"[ok] large-cap {len(lc)} · mid-cap {len(mc)} · mid funds {len(mf)}\n")

    # leg 1: large-cap pairs basket (validated config)
    pb, npairs = pairs_basket_dated(lc)
    print(f"[pairs basket] {npairs} pairs · {len(pb)} days · Sharpe {sharpe_of(_eq(list({t:pb[t] for t in sorted(pb)}.values())), PPY):.2f}")

    # leg 2: mid-cap value + vol-managed-momentum, inverse-vol ~50/50 (the 5/6-year blend, Case 42)
    fb = {s: mf[s] for s in mc if s in mf}
    rv = backtest_value_composite(mc, fb, top_frac=0.2, rebalance_days=21, cost_bps=2.0,
                                  momentum_weight=0.0, periods_per_year=PPY)
    rm = backtest_factor(mc, vol_managed_momentum_signal(120, 21, 60), name="volmom", top_frac=0.2,
                         rebalance_days=21, cost_bps=2.0, long_high=True, periods_per_year=PPY)
    vmap, mmap = dict(zip(rv.dates, rv.daily_returns)), dict(zip(rm.dates, rm.daily_returns))
    common_mid = sorted(set(vmap) & set(mmap))
    mid_combo = evaluate_combo({"value": [vmap[t] for t in common_mid],
                                "volmom": [mmap[t] for t in common_mid]}, ppy=PPY)
    w = mid_combo.invvol_weights
    midblend = {t: w["value"] * vmap[t] + w["volmom"] * mmap[t] for t in common_mid}
    print(f"[mid blend] {len(midblend)} days · Sharpe {sharpe_of(_eq([midblend[t] for t in common_mid]), PPY):.2f}")

    # date-join the two sleeves
    common = sorted(set(pb) & set(midblend))
    print(f"[joined on calendar] {len(common)} common days\n")
    if len(common) < 100:
        print("[abort] too few overlapping days"); return 1
    streams = {"pairs_basket": [pb[t] for t in common], "midcap_valmom": [midblend[t] for t in common]}
    rep = evaluate_combo(streams, ppy=PPY)
    rho = rep.corr_matrix[0][1]
    print(f"[CORRELATION pairs vs mid-cap blend]  rho = {rho:+.3f}")
    print(f"[two-sleeve book] each-leg ann-Sharpe { {k: round(v['ann_sharpe'],2) for k,v in rep.legs.items()} }")
    print(f"  equal-weight {rep.equalweight_sharpe:.2f} · inverse-vol {rep.invvol_sharpe:.2f}"
          f" · weights { {k: round(v,2) for k,v in rep.invvol_weights.items()} }")
    blended = {t: rep.invvol_weights["pairs_basket"] * pb[t] +
                  rep.invvol_weights["midcap_valmom"] * midblend[t] for t in common}
    yr = _per_year(blended)
    pos = sum(1 for s in yr.values() if s > 0)
    dsr = deflated_sharpe_ratio(_eq([blended[t] for t in common]), n_trials=85, sharpe_variance=1e-4)
    print(f"[combined regime] per-year {yr} -> {pos}/{len(yr)} positive · DSR {dsr:.2f}")
    diversifies = abs(rho) < 0.3 and rep.invvol_sharpe > max(v["ann_sharpe"] for v in rep.legs.values())
    print(f"\n[verdict] {'DIVERSIFIES — combined Sharpe beats both legs at low corr -> real 2nd leg' if diversifies else 'check: see numbers above'}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({
        "rho": round(rho, 4), "n_common_days": len(common),
        "leg_ann_sharpe": {k: round(v["ann_sharpe"], 3) for k, v in rep.legs.items()},
        "equalweight_sharpe": round(rep.equalweight_sharpe, 3),
        "invvol_sharpe": round(rep.invvol_sharpe, 3),
        "invvol_weights": {k: round(v, 3) for k, v in rep.invvol_weights.items()},
        "combined_per_year": yr, "combined_pos_years": f"{pos}/{len(yr)}", "combined_dsr": round(dsr, 3),
        "diversifies": bool(diversifies)}, indent=2))
    print(f"[saved] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
