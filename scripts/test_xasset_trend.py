"""
Case 51 — Cross-asset TREND / managed-futures (time-series momentum) as leg #3 — and a short-vol HEDGE.

Short-vol (Case 49) is the second leg but carries an un-sampled crash tail. The ideal third leg is CRISIS
ALPHA: cross-asset trend-following (CTAs) profits precisely in vol spikes (2022 was a banner year — trend
shorted bonds + stocks, went long commodities) — the opposite of short-vol. So a trend leg could both
diversify the book AND hedge short-vol's tail.

Construction (canonical diversified managed-futures, NOT the single-asset long-only TSMOM that was
rejected as a vol-scaling illusion): per ETF, position = sign(trailing 12-month return), VOL-SCALED to
equal risk (target/realized vol), summed across a cross-asset basket (equities/bonds/gold/commods/dollar/
credit). It GOES SHORT (the timing, not just exposure, must earn its keep), so we benchmark vs
buy-and-hold the same basket to prove it's trend timing, not beta.

Bar: Sharpe + per-year (2022 = the crisis-alpha test) + maxDD + correlation with pairs AND short-vol
(ideally NEGATIVE with short-vol in stress) + the 3-leg combiner lift + the Case-48 partial-year split.

Run: .venv/bin/python scripts/test_xasset_trend.py
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
from alpca.backtest.evaluation import sharpe_of, max_drawdown_of, deflated_sharpe_ratio  # noqa: E402

PPY = 252.0


def _load(c):
    return {p.name.split("_1day_")[0]: [json.loads(l) for l in p.open() if l.strip()]
            for p in Path(c).glob("*_1day_bars.jsonl")}


def _eq(daily, s=1.0):
    e = [s]
    for x in daily:
        e.append(e[-1] * (1 + x))
    return e


def _per_year(dmap):
    by = {}
    for t, x in dmap.items():
        by.setdefault(time.gmtime(t).tm_year, []).append(x)
    return {y: round(sharpe_of(_eq(v), PPY), 2) for y, v in sorted(by.items()) if len(v) >= 20}


def trend_book(bars, *, lookback=252, vol_window=40, target_vol=0.10, long_only=False):
    """Diversified cross-asset TSMOM: per asset sign(trailing `lookback` return), vol-scaled to
    `target_vol`/realized, equal-risk summed. Returns {ts: book_return}. long_only=True for the
    no-short control (exposure without timing)."""
    syms = sorted(bars)
    master = sorted({int(b["timestamp"]) for s in syms for b in bars[s]})
    idx = {t: i for i, t in enumerate(master)}
    T, N = len(master), len(syms)
    px = np.full((T, N), np.nan); ret = np.zeros((T, N))
    for j, s in enumerate(syms):
        b = sorted(bars[s], key=lambda x: int(x["timestamp"]))
        for k in range(len(b)):
            i = idx[int(b[k]["timestamp"])]; px[i, j] = float(b[k]["close"])
            if k > 0 and float(b[k - 1]["close"]) > 0:
                ret[i, j] = float(b[k]["close"]) / float(b[k - 1]["close"]) - 1.0
    out = {}
    for t in range(lookback + 1, T):
        sig = np.zeros(N); wsum = 0.0
        for j in range(N):
            if not (np.isfinite(px[t - 1, j]) and np.isfinite(px[t - 1 - lookback, j]) and px[t - 1 - lookback, j] > 0):
                continue
            trail = px[t - 1, j] / px[t - 1 - lookback, j] - 1.0
            direction = (1.0 if trail > 0 else -1.0) if not long_only else 1.0
            rv = np.std(ret[t - vol_window:t, j]) * np.sqrt(PPY)
            if rv > 1e-6:
                sig[j] = direction * (target_vol / rv); wsum += abs(sig[j])
        if wsum > 0:
            out[master[t]] = float(np.sum(sig * ret[t]) / wsum)   # gross-normalized book return
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--xasset", default="/Volumes/My Passport/AlpcaData/cache_xasset")
    ap.add_argument("--largecap", default="/Volumes/My Passport/AlpcaData/cache_largecap_sip")
    ap.add_argument("--vol-cache", default="/Volumes/My Passport/AlpcaData/cache_vol")
    ap.add_argument("--out", default="data/xasset_trend_results.json")
    args = ap.parse_args()

    xb = _load(args.xasset)
    print(f"[ok] cross-asset basket: {sorted(xb)} ({len(xb)} ETFs)")
    trend = trend_book(xb)
    bh = {}  # equal-weight buy-and-hold basket (beta benchmark)
    master = sorted({int(b["timestamp"]) for s in xb for b in xb[s]})
    for s in xb:
        b = sorted(xb[s], key=lambda x: int(x["timestamp"]))
        for k in range(1, len(b)):
            if float(b[k - 1]["close"]) > 0:
                t = int(b[k]["timestamp"]); bh[t] = bh.get(t, 0.0) + (float(b[k]["close"]) / float(b[k - 1]["close"]) - 1.0) / len(xb)
    lo = trend_book(xb, long_only=True)
    print(f"[trend] Sharpe {sharpe_of(_eq([trend[t] for t in sorted(trend)]), PPY):.2f} · "
          f"maxDD {max_drawdown_of(_eq([trend[t] for t in sorted(trend)]))*100:.1f}% · per-year {_per_year(trend)}")
    print(f"[controls] buy&hold basket Sharpe {sharpe_of(_eq([bh[t] for t in sorted(bh)]), PPY):.2f} · "
          f"long-only(no timing) Sharpe {sharpe_of(_eq([lo[t] for t in sorted(lo)]), PPY):.2f} "
          f"(trend must beat both to be real timing, not beta/vol-scaling)")

    # legs for correlation + combiner
    pr = delisting_aware_walkforward(_load(args.largecap), train=252, test=63, top_n=10, max_adf=-2.86)
    pairs = {int(t): r for t, r in zip(pr.dates, pr.daily_returns)}
    volb = _load(args.vol_cache)
    sv = {}
    b = sorted(volb["SVXY"], key=lambda x: int(x["timestamp"]))
    for k in range(1, len(b)):
        if float(b[k - 1]["close"]) > 0:
            sv[int(b[k]["timestamp"])] = float(b[k]["close"]) / float(b[k - 1]["close"]) - 1.0

    cps = sorted(set(trend) & set(pairs)); csv = sorted(set(trend) & set(sv))
    rho_p = correlation([trend[t] for t in cps], [pairs[t] for t in cps])
    rho_v = correlation([trend[t] for t in csv], [sv[t] for t in csv])
    # tail: short-vol's worst days — does trend HEDGE (positive when short-vol craters)?
    svv = [sv[t] for t in csv]; thr = np.percentile(svv, 10)
    tail = [trend[t] for t in csv if sv[t] <= thr]
    print(f"\n[correlation] trend vs pairs ρ={rho_p:+.3f} · trend vs short-vol ρ={rho_v:+.3f}")
    print(f"[crisis-alpha] on short-vol's worst-10% days, trend mean {np.mean(tail)*100:+.2f}%/day "
          f"({'HEDGES the tail' if np.mean(tail) > 0 else 'does NOT hedge'})")

    # 3-leg combiner (pairs + short-vol + trend) vs the current 2-leg book
    common3 = sorted(set(pairs) & set(sv) & set(trend))
    rep3 = evaluate_combo({"pairs": [pairs[t] for t in common3], "short_vol": [sv[t] for t in common3],
                           "trend": [trend[t] for t in common3]}, ppy=PPY)
    w = rep3.invvol_weights
    blend3 = [w["pairs"] * pairs[t] + w["short_vol"] * sv[t] + w["trend"] * trend[t] for t in common3]
    blend2 = [0.92 * pairs[t] + 0.08 * sv[t] for t in common3]
    dd3 = max_drawdown_of(_eq(blend3))
    print(f"\n[3-leg combiner] weights { {k: round(v,2) for k,v in w.items()} } · Sharpe {rep3.invvol_sharpe:.2f} "
          f"· maxDD {dd3*100:.1f}% · DSR {deflated_sharpe_ratio(_eq(blend3), n_trials=100, sharpe_variance=1e-4):.2f}")
    print(f"[vs 2-leg book] pairs+short-vol Sharpe {sharpe_of(_eq(blend2), PPY):.2f} maxDD {max_drawdown_of(_eq(blend2))*100:.1f}%")
    by = {}
    for t, x in zip(common3, blend3):
        by.setdefault(time.gmtime(t).tm_year, []).append(x)
    yr3 = {y: round(sharpe_of(_eq(v), PPY), 2) for y, v in sorted(by.items()) if len(v) >= 20}
    print(f"[3-leg per-year] {yr3}")

    trend_sh = sharpe_of(_eq([trend[t] for t in sorted(trend)]), PPY)
    real_timing = trend_sh > max(sharpe_of(_eq([bh[t] for t in sorted(bh)]), PPY), sharpe_of(_eq([lo[t] for t in sorted(lo)]), PPY))
    lift = rep3.invvol_sharpe - sharpe_of(_eq(blend2), PPY)
    verdict = ("CANDIDATE — real timing + lifts 3-leg book + hedges short-vol tail" if (real_timing and lift > 0.03)
               else f"reject (real_timing={real_timing}, 3leg_lift={lift:+.2f})")
    print(f"\n[verdict] {verdict}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({
        "trend_sharpe": round(trend_sh, 3), "trend_per_year": _per_year(trend),
        "rho_pairs": round(rho_p, 4), "rho_shortvol": round(rho_v, 4),
        "tail_hedge_mean_pct": round(float(np.mean(tail)) * 100, 3),
        "combiner3_sharpe": round(rep3.invvol_sharpe, 3), "combiner3_weights": {k: round(v, 3) for k, v in w.items()},
        "book2_sharpe": round(sharpe_of(_eq(blend2), PPY), 3), "lift_3leg": round(lift, 3), "verdict": verdict}, indent=2))
    print(f"[saved] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
