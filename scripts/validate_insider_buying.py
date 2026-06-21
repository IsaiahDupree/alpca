"""
Case 63 — validate INSIDER BUYING (long-only) through the honest battery.

This is the standout mined candidate: long-only -> NO borrow wall (the friction that killed PEAD,
SI-tilt, and every short-leg event edge). Claim (Quantpedia/SSRN): stocks with open-market insider
purchases earn abnormal forward returns. Signal from SEC bulk Form-4 data (free, no AV quota), keyed
on FILING_DATE = no lookahead.

Survivorship-robust by construction: run on the CLEAN 195-name large-cap 10.5yr SIP universe
(cache_sip_10y) rather than a backfilled small-cap basket (the adversary's #1 objection). The
decisive control is NOT "beat SPY" but "beat the UNIVERSE EQUAL-WEIGHT" — does selecting the
insider-buy names beat simply owning the whole universe? Plus fresh-symbol holdout + per-year.

Construction: monthly rebalance; a name is selected if it had net open-market insider buys filed in
the trailing `lookback` days; hold equal-weight to the next rebalance; 2bps cost on turnover; LONG ONLY.

Run: .venv/bin/python scripts/validate_insider_buying.py   -> data/insider_buying_results.json
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from alpca.backtest.evaluation import sharpe_of, max_drawdown_of  # noqa: E402

PPY = 252.0
COST = 2.0 / 1e4


def load_bars(cache: Path):
    bars = {}
    for p in cache.glob("*_1day_bars.jsonl"):
        sym = p.name.split("_1day_")[0]
        b = [json.loads(l) for l in p.open() if l.strip()]
        b.sort(key=lambda x: int(x["timestamp"]))
        bars[sym] = b
    return bars


def load_insider(path: Path):
    by_ticker = defaultdict(list)   # ticker -> [(filing_epoch, buy_value)]
    for line in path.open():
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        fd = r.get("filing_date")
        if not fd:
            continue
        ep = time.mktime(time.strptime(fd, "%Y-%m-%d"))
        by_ticker[r["ticker"]].append((ep, float(r.get("buy_value", 0))))
    for t in by_ticker:
        by_ticker[t].sort()
    return by_ticker


def ann_sharpe(r):
    eq = np.cumprod(1 + np.asarray(r))
    return sharpe_of([1.0] + list(eq), PPY)


def beta_alpha(strat, mkt):
    strat, mkt = np.asarray(strat), np.asarray(mkt)
    v = mkt.var()
    if v <= 0:
        return 0.0, 0.0
    beta = float(np.cov(strat, mkt)[0, 1] / v)
    return beta, float((strat.mean() - beta * mkt.mean()) * PPY)


def run(syms, bars, insider, spy_ret, common, *, lookback_days=90, hold_reb="month", min_buy=1.0):
    idx = {t: i for i, t in enumerate(common)}
    T = len(common)
    px = {s: {int(b["timestamp"]): float(b["close"]) for b in bars[s]} for s in syms}
    ret = {s: np.zeros(T) for s in syms}
    have = {s: np.zeros(T, dtype=bool) for s in syms}
    for s in syms:
        prev = None
        for i, t in enumerate(common):
            if t in px[s]:
                have[s][i] = True
                if prev is not None and prev > 0:
                    ret[s][i] = (px[s][t] - prev) / prev
                prev = px[s][t]
    # rebalance dates = first trading day of each month
    dstr = [time.strftime("%Y-%m-%d", time.gmtime(t)) for t in common]
    reb = [0] + [i for i in range(1, T) if dstr[i][:7] != dstr[i - 1][:7]]
    rebset = set(reb)
    lb = lookback_days * 86400
    sel_w = {s: np.zeros(T) for s in syms}
    uni_w = {s: np.zeros(T) for s in syms}
    cur_sel = []
    for i in range(T):
        if i in rebset:
            t = common[i]
            avail = [s for s in syms if have[s][i]]
            cur_sel = []
            for s in avail:
                buys = [v for (ep, v) in insider.get(s, []) if t - lb < ep <= t and v >= min_buy]
                if buys:
                    cur_sel.append(s)
            cur_uni = avail
        for s in syms:
            sel_w[s][i] = (1.0 / len(cur_sel)) if (cur_sel and s in cur_sel) else 0.0
            uni_w[s][i] = (1.0 / len(cur_uni)) if (cur_uni and s in cur_uni) else 0.0

    def book(w):
        daily = np.zeros(T)
        prev = {s: 0.0 for s in syms}
        for i in range(1, T):
            daily[i] = sum(w[s][i - 1] * ret[s][i] for s in syms)
            turn = sum(abs(w[s][i] - prev[s]) for s in syms)
            daily[i] -= turn * COST
            prev = {s: w[s][i] for s in syms}
        return daily

    sel = book(sel_w)
    uni = book(uni_w)
    # avg # selected
    navg = float(np.mean([sum(1 for s in syms if sel_w[s][i] > 0) for i in range(T)]))
    beta_spy, alpha_spy = beta_alpha(sel[1:], spy_ret[1:])
    # alpha vs the universe equal-weight (the real control)
    beta_uni, alpha_uni = beta_alpha(sel[1:], uni[1:])
    return {"sel": sel, "uni": uni, "n_avg_selected": round(navg, 1),
            "sel_sharpe": round(ann_sharpe(sel), 3), "uni_sharpe": round(ann_sharpe(uni), 3),
            "spy_sharpe": round(ann_sharpe(spy_ret), 3),
            "alpha_vs_spy_ann": round(alpha_spy, 4), "beta_vs_spy": round(beta_spy, 3),
            "alpha_vs_universe_ann": round(alpha_uni, 4),
            "sel_total_ret": round(float(np.prod(1 + sel) - 1), 4),
            "uni_total_ret": round(float(np.prod(1 + uni) - 1), 4),
            "sel_maxdd": round(max_drawdown_of([1.0] + list(np.cumprod(1 + sel))), 4),
            "dstr": dstr}


def peryear(daily, dstr):
    by = defaultdict(list)
    for i in range(1, len(daily)):
        by[dstr[i][:4]].append(daily[i])
    out = {}
    for y in sorted(by):
        e = np.cumprod(1 + np.array(by[y]))
        out[y] = round(sharpe_of([1.0] + list(e), PPY), 2)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/Volumes/My Passport/AlpcaData/cache_sip_10y")
    ap.add_argument("--insider", default="/Volumes/My Passport/AlpcaData/insider/insider_buys.jsonl")
    ap.add_argument("--out", default="data/insider_buying_results.json")
    args = ap.parse_args()

    bars = load_bars(Path(args.cache))
    insider = load_insider(Path(args.insider))
    syms = sorted([s for s in bars if s != "SPY"])
    spy = bars.get("SPY")
    # common calendar across the universe (union via SPY if present, else all)
    allts = sorted({int(b["timestamp"]) for s in bars for b in bars[s]})
    common = allts
    spy_px = {int(b["timestamp"]): float(b["close"]) for b in (spy or [])}
    spy_ret = np.zeros(len(common)); prev = None
    for i, t in enumerate(common):
        if t in spy_px:
            if prev and prev > 0:
                spy_ret[i] = (spy_px[t] - prev) / prev
            prev = spy_px[t]
    n_ins = sum(len(v) for v in insider.values())
    cov = len([s for s in syms if s in insider])
    print(f"[ok] {len(syms)} universe symbols, {cov} with insider data, {n_ins} insider-buy rows")

    full = run(syms, bars, insider, spy_ret, common)
    dstr = full["dstr"]
    py = peryear(full["sel"], dstr)
    pos_years = sum(1 for v in py.values() if v > 0)

    # fresh-symbol holdout: disjoint halves of the universe
    half = sorted(syms)
    train, hold = half[0::2], half[1::2]
    tr = run(train, bars, insider, spy_ret, common)
    ho = run(hold, bars, insider, spy_ret, common)

    print(f"\nFULL universe: selected~{full['n_avg_selected']} names | sel Sharpe {full['sel_sharpe']} "
          f"vs universe-EW {full['uni_sharpe']} vs SPY {full['spy_sharpe']}")
    print(f"  alpha vs SPY {full['alpha_vs_spy_ann']*100:+.1f}%/yr (beta {full['beta_vs_spy']}) | "
          f"alpha vs UNIVERSE-EW {full['alpha_vs_universe_ann']*100:+.1f}%/yr  <-- the real test")
    print(f"  per-year sel Sharpe: {py}  (+{pos_years}/{len(py)})")
    print(f"\nFRESH-SYMBOL HOLDOUT:")
    print(f"  TRAIN  half: sel {tr['sel_sharpe']} vs EW {tr['uni_sharpe']} | alpha-vs-EW {tr['alpha_vs_universe_ann']*100:+.1f}%/yr")
    print(f"  HOLDOUT half: sel {ho['sel_sharpe']} vs EW {ho['uni_sharpe']} | alpha-vs-EW {ho['alpha_vs_universe_ann']*100:+.1f}%/yr")

    beats_universe = full["alpha_vs_universe_ann"] > 0.01
    generalizes = ho["alpha_vs_universe_ann"] > 0.0 and tr["alpha_vs_universe_ann"] > 0.0
    regime_ok = pos_years >= len(py) - 2
    candidate = beats_universe and generalizes and regime_ok
    verdict = ("CANDIDATE — long-only insider-buy basket beats the universe equal-weight, generalizes "
               "to fresh symbols, and is regime-stable (no borrow wall -> potential 3rd leg)"
               if candidate else
               "REJECT — does not beat the universe equal-weight / fails fresh-symbol holdout / not "
               "regime-stable; the insider-buy subset is ~the universe beta")
    print(f"\nbeats-universe: {beats_universe} | generalizes(fresh): {generalizes} | regime-stable: {regime_ok}")
    print(f"VERDICT: {verdict}")

    out = {"case": 63, "name": "Insider buying (long-only) on large-cap 10.5yr SIP",
           "n_symbols": len(syms), "n_with_insider": cov, "n_insider_rows": n_ins,
           "full": {k: v for k, v in full.items() if k not in ("sel", "uni", "dstr")},
           "per_year_sel": py, "holdout": {
               "train_sel_sharpe": tr["sel_sharpe"], "train_alpha_vs_ew": tr["alpha_vs_universe_ann"],
               "holdout_sel_sharpe": ho["sel_sharpe"], "holdout_alpha_vs_ew": ho["alpha_vs_universe_ann"]},
           "candidate": bool(candidate), "verdict": verdict}
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\n[meta] wrote {args.out}")


if __name__ == "__main__":
    main()
