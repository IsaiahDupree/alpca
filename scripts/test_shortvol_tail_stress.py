"""
Case 50 — Explicit TAIL STRESS test of the short-vol sleeve (respect the un-sampled volmageddon).

Case 49's metrics (Sharpe 1.08, DSR 0.90, combined DD −5.5%) all UNDERSTATE short-vol's risk because
2021–2026 contained no volmageddon. The honest follow-through is not to wait for the spike but to SIMULATE
it: inject a catastrophic single-day SVXY shock into the combined book and measure the damage at our hard
cap — does the 8% cap actually protect the book, or do we need a smaller one?

Calibration of the shock (post-2018 SVXY is −0.5× short-VIX, so it can't go to ~0 like the original −1×
XIV that died −96% on 2018-02-05; a VIX-futures doubling ≈ −50% for the −0.5× fund):
  −40%  ≈ a bad spike (≈ Feb-2018-scale on the de-levered fund)
  −50%  ≈ the realistic worst single day for −0.5× SVXY
  −70%  ≈ beyond-historical stress (model risk / a gap through the −0.5× design)
Injected at the WORST moment — overlapped on the combined book's existing worst drawdown day (joint
stress: vol spikes are often when pairs spreads also widen).

Run: .venv/bin/python scripts/test_shortvol_tail_stress.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from alpca.backtest.pairs import delisting_aware_walkforward  # noqa: E402
from alpca.backtest.evaluation import max_drawdown_of, sharpe_of  # noqa: E402

PPY = 252.0


def _load(c):
    return {p.name.split("_1day_")[0]: [json.loads(l) for l in p.open() if l.strip()]
            for p in Path(c).glob("*_1day_bars.jsonl")}


def _eq(daily, s=1.0):
    e = [s]
    for x in daily:
        e.append(e[-1] * (1 + x))
    return e


def _rbt(bars):
    b = sorted(bars, key=lambda x: int(x["timestamp"])); o = {}
    for i in range(1, len(b)):
        p0 = float(b[i - 1]["close"])
        if p0 > 0:
            o[int(b[i]["timestamp"])] = float(b[i]["close"]) / p0 - 1.0
    return o


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vol-cache", default="/Volumes/My Passport/AlpcaData/cache_vol")
    ap.add_argument("--largecap", default="/Volumes/My Passport/AlpcaData/cache_largecap_sip")
    ap.add_argument("--out", default="data/shortvol_tail_stress.json")
    args = ap.parse_args()

    svxy = _rbt(_load(args.vol_cache)["SVXY"])
    pr = delisting_aware_walkforward(_load(args.largecap), train=252, test=63, top_n=10, max_adf=-2.86)
    pairs = {int(t): r for t, r in zip(pr.dates, pr.daily_returns)}
    common = sorted(set(pairs) & set(svxy))
    pv = np.array([pairs[t] for t in common]); vv = np.array([svxy[t] for t in common])

    # worst observed SVXY day in-sample, for context
    print(f"[context] worst SVXY day in 2021-2026 sample: {vv.min()*100:.1f}%  (the window had NO volmageddon)")
    # the joint-stress injection point = the book's existing worst drawdown day (vols spike when pairs widen)
    print(f"\n{'shock':>8}{'svxy_wt':>9}{'1-day book hit':>16}{'stressed maxDD':>16}{'base maxDD':>12}")
    print("-" * 62)
    out = {"worst_svxy_in_sample": round(float(vv.min()), 4), "scenarios": []}
    for wt in (0.05, 0.08, 0.12):
        pairs_wt = 1.0 - wt
        book = pairs_wt * pv + wt * vv
        base_dd = max_drawdown_of(_eq(book.tolist()))
        # find the worst book day (joint stress proxy) and inject the shock THERE
        inj = int(np.argmin(book))
        for shock in (-0.40, -0.50, -0.70):
            stressed = book.copy()
            # replace SVXY's contribution that day with the shock (keep pairs' that-day return)
            stressed[inj] = pairs_wt * pv[inj] + wt * shock
            day_hit = stressed[inj]
            dd = max_drawdown_of(_eq(stressed.tolist()))
            out["scenarios"].append({"svxy_weight": wt, "shock": shock,
                                     "one_day_book_hit": round(float(day_hit), 4),
                                     "stressed_maxdd": round(float(dd), 4), "base_maxdd": round(float(base_dd), 4)})
            print(f"{shock*100:>7.0f}%{wt:>9.0%}{day_hit*100:>15.2f}%{dd*100:>15.1f}%{base_dd*100:>11.1f}%")
        print()

    # verdict: at the DEPLOYED 8% cap, is the worst realistic shock (-50%) survivable?
    s8 = [s for s in out["scenarios"] if s["svxy_weight"] == 0.08 and s["shock"] == -0.50][0]
    survivable = s8["stressed_maxdd"] > -0.15      # combined book stays within a ~15% drawdown
    verdict = (f"CAP OK — at the deployed 8% cap, a realistic -50% SVXY spike costs {s8['one_day_book_hit']*100:.1f}% "
               f"on the day and a {s8['stressed_maxdd']*100:.1f}% book drawdown — survivable, sizing protects it"
               if survivable else
               f"RE-CAP — even at 8% a -50% spike pushes the book to {s8['stressed_maxdd']*100:.1f}% DD; size smaller")
    print(f"[verdict] {verdict}")
    out["verdict"] = verdict
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"[saved] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
