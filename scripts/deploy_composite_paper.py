"""
Multi-signal composite forward paper-track sleeve (Case 68) — the strongest non-deployed candidate.

The composite (momentum 12-1 + value B/P + short-interest, z-scored) backtested at Sharpe 0.64 /
t-stat 2.05 on 166 survivors — but its decisive control (survivorship) is UN-RUNNABLE on free data
(EDGAR/FINRA have no delisted coverage). A FORWARD TRACK is the survivorship-honest resolution: it
trades the universe as it exists going forward, including names that later delist, so the live curve
answers the one question the backtest can't.

LONG-ONLY top-quintile (no borrow wall). Monthly rebalance; each run marks the prior book to today's
close (realized_prev -> live OOS curve the paper-edge DB documents), then logs today's book. Uses the
full EDGAR fundamentals + FINRA SI caches (annual/bi-monthly, current enough) + daily-refreshed prices.
Zero capital — probation.

Run: .venv/bin/python scripts/deploy_composite_paper.py --cache "/Volumes/My Passport/AlpcaData/cache_sip_10y"
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TRACK = ROOT / "data" / "composite_forward_track.jsonl"
DEQ = "/Volumes/My Passport/AlpcaData"
LB, SKIP = 251, 21
TOP_FRAC = 0.2


def _epoch(s, fmt):
    try:
        return time.mktime(time.strptime(s, fmt))
    except Exception:
        return None


def prev_row():
    if not TRACK.exists():
        return None
    last = None
    for line in TRACK.open():
        if line.strip():
            last = line
    return json.loads(last) if last else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=f"{DEQ}/cache_sip_10y")
    ap.add_argument("--fund", default=f"{DEQ}/fundamentals_edgar")
    ap.add_argument("--si", default=f"{DEQ}/short_interest_finra")
    args = ap.parse_args()
    cache, fundd, sid = Path(args.cache), Path(args.fund), Path(args.si)

    fund_syms = {p.name.split("_")[0] for p in fundd.glob("*.json")}
    si_syms = {p.name.split("_si")[0] for p in sid.glob("*.json")}
    bars = {}
    for p in cache.glob("*_1day_bars.jsonl"):
        s = p.name.split("_1day_")[0]
        if s in fund_syms and s in si_syms:
            rows = [json.loads(l) for l in p.open() if l.strip()]
            if len(rows) > LB + SKIP + 5:
                bars[s] = sorted(rows, key=lambda x: int(x["timestamp"]))
    syms = sorted(bars)
    if len(syms) < 20:
        print(f"[composite] only {len(syms)} names with all signals — too few; abort")
        return
    last_close = {s: float(bars[s][-1]["close"]) for s in syms}
    today = time.strftime("%Y-%m-%d")
    asof = int(time.mktime(time.strptime(today, "%Y-%m-%d")))

    # ---- mark the PRIOR book to today's close ----
    realized = None
    prev = prev_row()
    if prev and prev.get("book"):
        rets = [last_close[s] / e - 1.0 for s, e in prev["book"].items() if s in last_close and e > 0]
        if rets:
            realized = sum(rets) / len(rets)

    # ---- rebalance monthly (carry the prior book within a month) ----
    rebalance = not prev or prev.get("date", "")[:7] != today[:7]
    if not rebalance and prev and prev.get("book"):
        book = prev["book"]
    else:
        # momentum 12-1
        mom = {}
        for s in syms:
            cl = [float(b["close"]) for b in bars[s]]
            if len(cl) > LB + SKIP and cl[-SKIP - LB] > 0:
                mom[s] = cl[-SKIP] / cl[-SKIP - LB] - 1.0
        # value B/P (latest filed) + short-interest (latest settlement, low DTC bullish)
        val, sib = {}, {}
        for s in syms:
            frows = sorted(json.load((fundd / f"{s}_fund.json").open()), key=lambda r: r.get("filed", ""))
            be = sh = None
            for r in frows:
                if r.get("book_equity") and r.get("shares") and _epoch(r.get("filed", ""), "%Y-%m-%d"):
                    be, sh = r["book_equity"], r["shares"]
            if be and sh and last_close[s] > 0:
                val[s] = be / (last_close[s] * sh)
            srows = sorted(json.load((sid / f"{s}_si.json").open()), key=lambda r: r.get("settlement", ""))
            if srows and srows[-1].get("days_to_cover") is not None:
                sib[s] = -float(srows[-1]["days_to_cover"])

        def z(d):
            xs = np.array(list(d.values()))
            mu, sd = xs.mean(), xs.std()
            return {k: ((v - mu) / sd if sd > 0 else 0.0) for k, v in d.items()}
        zmom, zval, zsi = z(mom), z(val), z(sib)
        comp = {}
        for s in syms:
            zs = [zz[s] for zz in (zmom, zval, zsi) if s in zz]
            if zs:
                comp[s] = sum(zs) / len(zs)
        ranked = sorted(comp.items(), key=lambda x: x[1], reverse=True)
        positive = [(s, v) for s, v in ranked if v > 0]
        k = max(1, int(TOP_FRAC * len(comp)))
        book = {s: last_close[s] for s, _ in positive[:k]} if len(positive) >= k else {}

    row = {"date": today, "asof": asof, "n_universe": len(syms), "n_long": len(book),
           "book": book, "realized_prev": realized, "rebalanced": bool(rebalance),
           "status": "trading" if book else "accumulating"}
    with TRACK.open("a") as f:
        f.write(json.dumps(row) + "\n")
    print(f"[composite] {today}: universe {len(syms)} · long {len(book)} "
          f"· realized_prev {realized} · rebalanced={rebalance}")


if __name__ == "__main__":
    main()
