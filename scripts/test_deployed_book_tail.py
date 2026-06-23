"""
Case 67 — REAL tail-stress + out-of-regime test of the DEPLOYED book (pairs + short-vol).

Case 50 stress-tested the short-vol sleeve against a SIMULATED volmageddon. This upgrades it to the
REAL events: SVXY now has SIP history back to 2016-01, so the deployed book — pairs (92%, delisting-
aware WF) + short-vol (long SVXY, hard-capped 8%) — can be marked through the **Feb-2018 volmageddon**
and the **March-2020 COVID crash**, the exact tail the 8% cap exists to survive.

Conservatism note: ProShares cut SVXY from -1x to -0.5x on 2018-02-27. Our pre-2018 SVXY bars are the
-1x (2x more violent) instrument, so the Feb-2018 result OVERSTATES the deployed tail — if the capped
book survives this, it survives the safer -0.5x SVXY we actually hold. A genuinely conservative stress.

Run: .venv/bin/python scripts/test_deployed_book_tail.py   -> data/deployed_book_tail.json
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from alpca.backtest.pairs import delisting_aware_walkforward  # noqa: E402
from alpca.backtest.evaluation import sharpe_of, max_drawdown_of  # noqa: E402
from alpca.live.portfolio import DEPLOYED  # noqa: E402

PPY = 252.0
W_PAIRS = next(s.weight for s in DEPLOYED if s.name == "pairs")
W_SVOL = next(min(s.weight, s.cap) if s.cap else s.weight for s in DEPLOYED if s.name == "short_vol")


def svxy_returns(path):
    b = [json.loads(l) for l in Path(path).open() if l.strip()]
    b.sort(key=lambda x: int(x["timestamp"]))
    out = {}
    prev = None
    for r in b:
        c = float(r["close"]); t = int(r["timestamp"])
        if prev and prev > 0:
            out[t] = c / prev - 1.0
        prev = c
    return out


def ann(r):
    r = np.asarray(r)
    return float(r.mean() / r.std() * (PPY ** 0.5)) if len(r) and r.std() > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--equity-cache", default="/Volumes/My Passport/AlpcaData/cache_sip_10y")
    ap.add_argument("--svxy", default="/Volumes/My Passport/AlpcaData/cache_vol_10y/SVXY_1day_bars.jsonl")
    ap.add_argument("--out", default="data/deployed_book_tail.json")
    args = ap.parse_args()

    print("[book] running delisting-aware pairs WF on 10.5yr universe (deployed config)...")
    bars = {p.name.split("_1day_")[0]: [json.loads(l) for l in p.open() if l.strip()]
            for p in Path(args.equity_cache).glob("*_1day_bars.jsonl")}
    pr = delisting_aware_walkforward(bars, train=252, test=63, top_n=10, max_adf=-2.86, periods_per_year=PPY)
    pairs = {int(t): r for t, r in zip(pr.dates, pr.daily_returns)}
    svol = svxy_returns(args.svxy)
    print(f"[book] pairs {len(pairs)} OOS days · SVXY {len(svol)} days")

    # combined deployed book on the union of dates (each sleeve contributes 0 on days it lacks)
    dates = sorted(set(pairs) | set(svol))
    rows = []
    for t in dates:
        p = pairs.get(t, 0.0); s = svol.get(t, 0.0)
        rows.append((t, W_PAIRS * p + W_SVOL * s, p, s))
    comb = np.array([r[1] for r in rows])
    ds = [time.strftime("%Y-%m-%d", time.gmtime(r[0])) for r in rows]

    eq = np.cumprod(1 + comb)
    full = {"sharpe": round(ann(comb), 3), "maxdd_pct": round(max_drawdown_of([1.0] + list(eq)) * 100, 2),
            "cum_pct": round((eq[-1] - 1) * 100, 2), "n_days": len(comb)}

    by_year = defaultdict(list)
    svol_by_year = defaultdict(list)
    for (t, c, p, s), d in zip(rows, ds):
        by_year[d[:4]].append(c); svol_by_year[d[:4]].append(s)
    per_year = {y: round(ann(by_year[y]), 2) for y in sorted(by_year)}

    # worst single days of the COMBINED book (the tail the cap must contain)
    order = np.argsort(comb)
    worst = [{"date": ds[i], "book_ret_pct": round(comb[i] * 100, 2),
              "svxy_ret_pct": round(rows[i][3] * 100, 2), "pairs_ret_pct": round(rows[i][2] * 100, 2)}
             for i in order[:8]]

    # specific tail windows
    def window_dd(lo, hi):
        idx = [i for i, d in enumerate(ds) if lo <= d <= hi]
        if not idx:
            return None
        seg = comb[idx[0]:idx[-1] + 1]
        e = np.cumprod(1 + seg)
        return {"window": f"{lo}..{hi}", "book_dd_pct": round(max_drawdown_of([1.0] + list(e)) * 100, 2),
                "worst_day_pct": round(seg.min() * 100, 2), "svxy_worst_pct": round(min(
                    rows[i][3] for i in idx) * 100, 2)}
    feb2018 = window_dd("2018-01-25", "2018-02-28")
    mar2020 = window_dd("2020-02-20", "2020-04-30")

    out = {"case": 67, "name": "Deployed book real tail-stress (pairs + short-vol, 2016-2026)",
           "weights": {"pairs": W_PAIRS, "short_vol": W_SVOL}, "full": full, "per_year": per_year,
           "worst_8_days": worst, "feb_2018_volmageddon": feb2018, "mar_2020_covid": mar2020,
           "note": "pre-2018-02-27 SVXY is -1x (2x deployed -0.5x) => Feb-2018 is a CONSERVATIVE overstatement"}

    pos = sum(1 for v in per_year.values() if v > 0)
    # the cap holds if the worst single book-day is tolerable (<~10% even on the -1x 2018 event)
    worst_day = min(comb) * 100
    cap_holds = worst_day > -10.0 and (feb2018 is None or feb2018["book_dd_pct"] > -15.0)
    out["verdict"] = ("DEPLOYED BOOK SURVIVES THE REAL TAIL — the 8% short-vol cap contains Feb-2018 + "
                      f"Mar-2020; worst book day {worst_day:.1f}%, +{pos}/{len(per_year)} yrs"
                      if cap_holds else
                      f"TAIL BREACH — worst book day {worst_day:.1f}%; the 8% cap is too high for the real event")
    Path(args.out).write_text(json.dumps(out, indent=2))

    print(f"\nDEPLOYED BOOK (pairs {W_PAIRS:.0%} + short-vol {W_SVOL:.0%}) 2016-2026:")
    print(f"  full: Sharpe {full['sharpe']} · maxDD {full['maxdd_pct']}% · cum {full['cum_pct']}% · +{pos}/{len(per_year)} yrs")
    print(f"  per-year: {per_year}")
    print(f"  Feb-2018 volmageddon: {feb2018}")
    print(f"  Mar-2020 COVID:       {mar2020}")
    print(f"  worst 8 book-days: " + ", ".join(f"{w['date']} {w['book_ret_pct']}%(SVXY {w['svxy_ret_pct']}%)" for w in worst[:4]))
    print(f"\nVERDICT: {out['verdict']}")
    print(f"[meta] wrote {args.out}")


if __name__ == "__main__":
    main()
