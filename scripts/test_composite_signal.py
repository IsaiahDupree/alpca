"""
Case 68 — Multi-signal cross-sectional composite (the last tractable systematic hunt).

Combine the three signals that HAVE point-in-time history — momentum (12-1), value (B/P), and
short-interest (low days-to-cover) — into one z-scored cross-sectional rank, monthly L/S, and ask the
only question that matters: does combining BEAT the best single signal, or is it just averaging weak
parts? (Revision is snapshot-only -> excluded, already on its own forward track, Case 66.)

Universe: the 166 large-cap names with all three signals (price ∩ EDGAR fundamentals ∩ FINRA SI).
SURVIVORSHIP CAVEAT (explicit): fundamentals + SI have ZERO coverage of delisted names, so this can
only run on survivors — momentum is known to flatter on survivors (Case 65: 0.50 survivor -> 0.23
delisted), so a good composite here is suspect until fundamental/SI reach delisted names (they can't,
on free data). The ablation + fresh-symbol holdout are the honest guards we DO have.

Controls: ablation (composite vs each single + vs equal-weight market), beta-decomp vs SPY,
per-year / out-of-regime, cost, DSR (t-stat), fresh-symbol holdout (disjoint half of the 166).

Run: .venv/bin/python scripts/test_composite_signal.py   -> data/composite_signal.json
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from alpca.backtest.factor import backtest_factor  # noqa: E402

PPY = 252.0
DEQ = "/Volumes/My Passport/AlpcaData"


def _epoch(s, fmt):
    try:
        return time.mktime(time.strptime(s, fmt))
    except Exception:
        return None


def zscore_rows(M):
    """Cross-sectional z-score per row (nan-aware)."""
    Z = np.full_like(M, np.nan)
    for t in range(M.shape[0]):
        row = M[t]
        ok = np.isfinite(row)
        if ok.sum() >= 5:
            mu, sd = row[ok].mean(), row[ok].std()
            if sd > 0:
                Z[t, ok] = (row[ok] - mu) / sd
    return Z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=f"{DEQ}/cache_sip_10y")
    ap.add_argument("--fund", default=f"{DEQ}/fundamentals_edgar")
    ap.add_argument("--si", default=f"{DEQ}/short_interest_finra")
    ap.add_argument("--out", default="data/composite_signal.json")
    args = ap.parse_args()

    cache, fundd, sid = Path(args.cache), Path(args.fund), Path(args.si)
    fund_syms = {p.name.split("_")[0] for p in fundd.glob("*.json")}
    si_syms = {p.name.split("_si")[0] for p in sid.glob("*.json")}
    bars = {}
    for p in cache.glob("*_1day_bars.jsonl"):
        s = p.name.split("_1day_")[0]
        if s in fund_syms and s in si_syms:
            rows = [json.loads(l) for l in p.open() if l.strip()]
            if rows:
                bars[s] = sorted(rows, key=lambda x: int(x["timestamp"]))
    syms = sorted(bars)
    spy_bars = None
    spyp = cache / "SPY_1day_bars.jsonl"
    if spyp.exists():
        spy_bars = sorted((json.loads(l) for l in spyp.open() if l.strip()), key=lambda x: int(x["timestamp"]))
    master = sorted({int(b["timestamp"]) for s in syms for b in bars[s]})
    tidx = {t: i for i, t in enumerate(master)}
    T, N = len(master), len(syms)
    print(f"[composite] {N} names x {T} days (price ∩ fundamentals ∩ SI)")

    P = np.full((T, N), np.nan)
    for j, s in enumerate(syms):
        for b in bars[s]:
            P[tidx[int(b["timestamp"])], j] = float(b["close"])

    # ---- component signals (all oriented so HIGHER = more bullish; long_high=True) ----
    LB, SKIP = 251, 21
    mom = np.full((T, N), np.nan)
    for t in range(LB + SKIP, T):
        a, b0 = t - SKIP, t - SKIP - LB
        mom[t] = np.where((P[b0] > 0) & np.isfinite(P[a]) & np.isfinite(P[b0]), P[a] / P[b0] - 1.0, np.nan)

    # value B/P = book_equity / (price * shares), stepped at 10-K filed date (no lookahead)
    book = np.full((T, N), np.nan); shr = np.full((T, N), np.nan)
    for j, s in enumerate(syms):
        rows = sorted(json.load((fundd / f"{s}_fund.json").open()), key=lambda r: r.get("fy_end", ""))
        for r in rows:
            be, sh, ep = r.get("book_equity"), r.get("shares"), _epoch(r.get("filed", ""), "%Y-%m-%d")
            if be and sh and ep:
                k0 = next((m for m, t in enumerate(master) if t >= ep), None)
                if k0 is not None:
                    book[k0:, j] = be; shr[k0:, j] = sh
    val = np.where((P > 0) & np.isfinite(book) & (shr > 0), book / (P * shr), np.nan)

    # short interest: -days_to_cover (low SI bullish), stepped at settlement + ~5d publication lag
    si = np.full((T, N), np.nan)
    for j, s in enumerate(syms):
        recs = sorted(json.load((sid / f"{s}_si.json").open()), key=lambda r: r.get("settlement", ""))
        for r in recs:
            dtc = r.get("days_to_cover"); ep = _epoch(r.get("settlement", ""), "%m/%d/%Y")
            if dtc is not None and ep is not None:
                k0 = next((m for m, t in enumerate(master) if t >= ep + 5 * 86400), None)
                if k0 is not None:
                    si[k0:, j] = -float(dtc)

    zmom, zval, zsi = zscore_rows(mom), zscore_rows(val), zscore_rows(si)
    comp = np.nanmean(np.stack([zmom, zval, zsi]), axis=0)

    def sig_of(M):
        return lambda master_, syms_, price_: M

    def beta_alpha(daily, dates):
        if spy_bars is None:
            return None, None
        spx = {int(b["timestamp"]): float(b["close"]) for b in spy_bars}
        sr = {}
        ks = sorted(spx)
        for i in range(1, len(ks)):
            if spx[ks[i - 1]] > 0:
                sr[ks[i]] = spx[ks[i]] / spx[ks[i - 1]] - 1
        xs, ys = [], []
        for r, t in zip(daily, dates):
            if t in sr:
                xs.append(r); ys.append(sr[t])
        if len(xs) < 50:
            return None, None
        xs, ys = np.array(xs), np.array(ys)
        v = ys.var()
        b = float(np.cov(xs, ys)[0, 1] / v) if v > 0 else 0.0
        return round(b, 3), round(float((xs.mean() - b * ys.mean()) * PPY), 4)

    def evaluate(M, label):
        r = backtest_factor(bars, sig_of(M), name=label, top_frac=0.2, long_high=True, cost_bps=2.0)
        d = np.array(r.daily_returns)
        t = float(d.mean() / d.std() * math.sqrt(len(d))) if len(d) and d.std() > 0 else 0.0
        by = defaultdict(list)
        for ret, ts in zip(r.daily_returns, r.dates):
            by[time.strftime("%Y", time.gmtime(ts))].append(ret)
        py = {y: round(np.mean(by[y]) / (np.std(by[y]) + 1e-12) * math.sqrt(PPY), 2) for y in sorted(by)}
        oor = [ret for ret, ts in zip(r.daily_returns, r.dates) if int(time.strftime("%Y", time.gmtime(ts))) <= 2020]
        beta, alpha = beta_alpha(r.daily_returns, r.dates)
        return {"sharpe": round(r.sharpe, 3), "maxdd_pct": round(r.max_drawdown * 100, 2),
                "tstat": round(t, 2), "beta": beta, "alpha_ann": alpha,
                "oor_2016_2020": round(np.mean(oor) / (np.std(oor) + 1e-12) * math.sqrt(PPY), 2) if len(oor) > 30 else None,
                "pos_years": f"{sum(1 for v in py.values() if v>0)}/{len(py)}", "per_year": py}

    out = {"case": 68, "name": "Multi-signal cross-sectional composite (momentum+value+short-interest)",
           "universe": N, "survivorship_caveat": "166 survivors only; fundamentals+SI have 0 delisted coverage"}
    print("\n=== ABLATION (monthly L/S top-quintile, 166 survivors) ===")
    for M, lbl in [(zmom, "momentum"), (zval, "value_BP"), (zsi, "short_interest"), (comp, "COMPOSITE")]:
        out[lbl] = evaluate(M, lbl)
        e = out[lbl]
        print(f"  {lbl:14} Sh {e['sharpe']:+.2f} | t {e['tstat']:+.2f} | beta {e['beta']} | "
              f"OOR {e['oor_2016_2020']} | {e['pos_years']}yr | maxDD {e['maxdd_pct']}%")

    # fresh-symbol holdout on the composite (disjoint halves)
    tr_idx, ho_idx = list(range(0, N, 2)), list(range(1, N, 2))
    def subset(idx):
        sub = {syms[j]: bars[syms[j]] for j in idx}
        return sub, comp[:, idx]
    print("\n=== FRESH-SYMBOL HOLDOUT (composite, disjoint halves) ===")
    hold = {}
    for nm, idx in [("train", tr_idx), ("holdout", ho_idx)]:
        sub, M = subset(idx)
        r = backtest_factor(sub, lambda m, s, p: M, name=f"comp_{nm}", top_frac=0.2, long_high=True)
        hold[nm] = round(r.sharpe, 3)
        print(f"  {nm:8} composite Sharpe {r.sharpe:+.3f}")
    out["holdout"] = hold

    c, best_single = out["COMPOSITE"], max(out["momentum"]["sharpe"], out["value_BP"]["sharpe"], out["short_interest"]["sharpe"])
    beats_best = c["sharpe"] > best_single + 0.1
    significant = (c["tstat"] or 0) > 2.0
    generalizes = hold["holdout"] > 0.1
    out["verdict"] = {
        "beats_best_single": beats_best, "significant": significant, "generalizes_fresh": generalizes,
        "best_single_sharpe": round(best_single, 3),
        "summary": ("CANDIDATE — the composite beats its best single signal, is significant, and "
                    "generalizes to fresh symbols (but survivorship-limited — see caveat)"
                    if (beats_best and significant and generalizes) else
                    "REJECT — combining did not produce a significant, generalizing edge beyond the best "
                    "single signal (and the survivor-only universe flatters it anyway)")}
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\nbest single Sharpe {best_single:.2f} | composite {c['sharpe']:.2f} | beats_best={beats_best} "
          f"sig={significant} fresh={generalizes}")
    print(f"VERDICT: {out['verdict']['summary']}")
    print(f"[meta] wrote {args.out}")


if __name__ == "__main__":
    main()
