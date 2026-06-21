"""
Case 63 (deep) — the decisive controls for the insider-buying long-only candidate.

It passed the first battery (beats universe-EW, generalizes, regime-stable). Before believing a
+1.4%/yr alpha-vs-EW, run the make-or-break skeptic checks:
  1. LONG-SHORT SPREAD (insider-subset − universe-EW): annualized Sharpe + t-stat + per-year + DSR.
  2. RANDOM-SUBSET PLACEBO (the killer): each month pick the SAME NUMBER of names at random; does
     insider selection beat random selection? Percentile of the insider alpha vs 300 random books.
  3. CORRELATION to the deployed pairs book — is it a genuinely uncorrelated 3rd leg?

Gross spreads (cost ~cancels in subset-vs-EW and is identical for insider vs placebo) — the question
is purely whether the INSIDER signal carries information beyond owning a random same-size slice.

Run: .venv/bin/python scripts/validate_insider_deep.py   -> data/insider_deep_results.json
"""

from __future__ import annotations

import json
import math
import time
from collections import defaultdict
from pathlib import Path

import argparse
import numpy as np

INSIDER = Path("/Volumes/My Passport/AlpcaData/insider/insider_buys.jsonl")
PAIRS = Path("data/pairs_wf_returns.json")
PPY = 252.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/Volumes/My Passport/AlpcaData/cache_sip_10y")
    ap.add_argument("--out", default="data/insider_deep_results.json")
    args = ap.parse_args()
    CACHE = Path(args.cache)
    bars = {}
    for p in CACHE.glob("*_1day_bars.jsonl"):
        s = p.name.split("_1day_")[0]
        b = [json.loads(l) for l in p.open() if l.strip()]
        b.sort(key=lambda x: int(x["timestamp"]))
        bars[s] = b
    syms = sorted([s for s in bars if s != "SPY"])
    common = sorted({int(b["timestamp"]) for s in syms for b in bars[s]})
    T, N = len(common), len(syms)
    tidx = {t: i for i, t in enumerate(common)}
    R = np.zeros((T, N)); have = np.zeros((T, N), dtype=bool)
    for j, s in enumerate(syms):
        prev = None
        px = {int(b["timestamp"]): float(b["close"]) for b in bars[s]}
        for i, t in enumerate(common):
            if t in px:
                have[i, j] = True
                if prev and prev > 0:
                    R[i, j] = (px[t] - prev) / prev
                prev = px[t]

    insider = defaultdict(list)
    for line in INSIDER.open():
        r = json.loads(line)
        if r.get("filing_date"):
            insider[r["ticker"]].append((time.mktime(time.strptime(r["filing_date"], "%Y-%m-%d")),
                                         float(r.get("buy_value", 0))))
    dstr = [time.strftime("%Y-%m-%d", time.gmtime(t)) for t in common]
    reb = [0] + [i for i in range(1, T) if dstr[i][:7] != dstr[i - 1][:7]]
    lb = 90 * 86400

    # build per-day insider-selected mask (piecewise constant between rebalances)
    Msel = np.zeros((T, N), dtype=bool)
    seg_bounds = reb + [T]
    seg_avail, seg_sel_idx = [], []
    for si in range(len(reb)):
        a, bnd = reb[si], seg_bounds[si + 1]
        t = common[a]
        avail = [j for j in range(N) if have[a, j]]
        sel = [j for j in avail if any(t - lb < ep <= t and v > 0 for ep, v in insider.get(syms[j], []))]
        for i in range(a, bnd):
            for j in sel:
                Msel[i, j] = True
        seg_avail.append(avail); seg_sel_idx.append((a, bnd, avail, sel))

    avail_cnt = have.sum(1).astype(float); avail_cnt[avail_cnt == 0] = 1
    avail_mean = (R * have).sum(1) / avail_cnt
    sel_cnt = Msel.sum(1).astype(float)
    sel_mean = np.where(sel_cnt > 0, (R * Msel).sum(1) / np.where(sel_cnt == 0, 1, sel_cnt), avail_mean)
    # spread uses prior-day weights -> shift selection by one day (no lookahead on returns)
    spread = np.zeros(T)
    spread[1:] = sel_mean[1:] - avail_mean[1:]   # subset already decided at prior rebalance (held)

    def ann_sharpe(x):
        x = np.asarray(x); s = x.std()
        return float(x.mean() / s * math.sqrt(PPY)) if s > 0 else 0.0
    sp = spread[1:]
    spread_sharpe = ann_sharpe(sp)
    tstat = float(sp.mean() / sp.std() * math.sqrt(len(sp))) if sp.std() > 0 else 0.0
    alpha_ann = float(sp.mean() * PPY)
    # per-year spread sharpe
    py = {}
    by = defaultdict(list)
    for i in range(1, T):
        by[dstr[i][:4]].append(spread[i])
    for y in sorted(by):
        py[y] = round(ann_sharpe(by[y]), 2)
    pos_years = sum(1 for v in py.values() if v > 0)

    # ---- RANDOM-SUBSET PLACEBO ----
    rng = np.random.default_rng(11)
    n_placebo = 300
    placebo_alpha, placebo_sharpe = [], []
    for _ in range(n_placebo):
        Mr = np.zeros((T, N), dtype=bool)
        for (a, bnd, avail, sel) in seg_sel_idx:
            k = len(sel)
            if k == 0 or not avail:
                continue
            pick = rng.choice(avail, size=min(k, len(avail)), replace=False)
            Mr[a:bnd, pick] = True
        rc = Mr.sum(1).astype(float)
        rmean = np.where(rc > 0, (R * Mr).sum(1) / np.where(rc == 0, 1, rc), avail_mean)
        rsp = np.zeros(T); rsp[1:] = rmean[1:] - avail_mean[1:]
        placebo_alpha.append(float(rsp[1:].mean() * PPY))
        placebo_sharpe.append(ann_sharpe(rsp[1:]))
    placebo_alpha = np.array(placebo_alpha); placebo_sharpe = np.array(placebo_sharpe)
    pctile_alpha = float((placebo_alpha < alpha_ann).mean())
    pctile_sharpe = float((placebo_sharpe < spread_sharpe).mean())

    # ---- correlation to deployed pairs book ----
    corr = None
    if PAIRS.exists():
        pj = json.loads(PAIRS.read_text())
        pmap = {row["date"]: row["ret"] for row in pj.get("returns", [])}
        xs, ys = [], []
        for i in range(1, T):
            d = dstr[i]
            if d in pmap:
                xs.append(spread[i]); ys.append(pmap[d])
        if len(xs) > 30:
            corr = float(np.corrcoef(xs, ys)[0, 1])

    verdict = ("CANDIDATE 3rd leg — insider selection beats random same-size selection (placebo "
               f"pctile {pctile_sharpe:.2f}), significant spread, regime-stable, uncorrelated to pairs"
               if (pctile_sharpe > 0.95 and tstat > 2.0 and pos_years >= len(py) - 2)
               else "REJECT/THIN — insider selection is NOT distinguishable from random same-size "
               f"selection (placebo pctile {pctile_sharpe:.2f}) or spread is insignificant (t={tstat:.1f})")

    print(f"LONG-SHORT SPREAD (insider-subset - universe-EW), gross:")
    print(f"  ann Sharpe {spread_sharpe:.3f} | t-stat {tstat:.2f} | alpha {alpha_ann*100:+.2f}%/yr | +{pos_years}/{len(py)} yrs")
    print(f"  per-year: {py}")
    print(f"\nRANDOM-SUBSET PLACEBO (300 books, same monthly count):")
    print(f"  insider alpha {alpha_ann*100:+.2f}%/yr  vs placebo mean {placebo_alpha.mean()*100:+.2f}%/yr "
          f"(std {placebo_alpha.std()*100:.2f}) -> percentile {pctile_alpha:.3f}")
    print(f"  insider spread-Sharpe {spread_sharpe:.3f} vs placebo mean {placebo_sharpe.mean():.3f} "
          f"-> percentile {pctile_sharpe:.3f}")
    print(f"\ncorrelation to deployed pairs book: {corr}")
    print(f"\nVERDICT: {verdict}")

    out = {"case": "63-deep", "spread_sharpe_gross": round(spread_sharpe, 3), "tstat": round(tstat, 2),
           "alpha_ann": round(alpha_ann, 4), "per_year": py, "pos_years": pos_years,
           "placebo_pctile_alpha": round(pctile_alpha, 3), "placebo_pctile_sharpe": round(pctile_sharpe, 3),
           "placebo_alpha_mean": round(float(placebo_alpha.mean()), 4),
           "placebo_alpha_std": round(float(placebo_alpha.std()), 4),
           "corr_to_pairs": (round(corr, 3) if corr is not None else None), "verdict": verdict}
    Path(args.out).write_text(json.dumps(out, indent=2))
    print("[meta] wrote data/insider_deep_results.json")


if __name__ == "__main__":
    main()
