"""
Analyst-revision-drift forward paper-track sleeve (Case 66) — LONG-ONLY (no borrow wall).

Signal: stocks whose consensus EPS estimates are being revised UP (positive 90-day estimate momentum
AND positive net revision breadth) tend to drift up (post-revision drift, the analyst-momentum
anomaly). We can't backtest it (AV serves only a current estimate snapshot, no PIT history), so this
is a pure FORWARD track: each run marks the prior logged book to today's close (realized return -> live
OOS curve), then logs today's long-only equal-weight book of top-signal names.

Honesty: NO backtest verdict exists — this STARTS THE CLOCK. It accumulates a live curve the
paper-edge DB documents, and only scores after enough marks. Needs the signal ledger
(data/revision_signal.jsonl, built by build_revision_signal.py) to reach adequate breadth first.

Run: .venv/bin/python scripts/deploy_revision_paper.py --cache "/Volumes/My Passport/AlpcaData/cache_sip_10y"
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SIGNAL = ROOT / "data" / "revision_signal.jsonl"
TRACK = ROOT / "data" / "revision_forward_track.jsonl"
MIN_NAMES = 8          # need a minimally diversified long book before trading
TOP_FRAC = 0.33        # long the top third by composite revision signal


def latest_signals():
    """Most recent snapshot per symbol from the dated ledger."""
    if not SIGNAL.exists():
        return {}
    by = {}
    for line in SIGNAL.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("est_mom_90d") is None and r.get("rev_breadth_30d") is None:
            continue
        # keep the newest by asof
        if r["symbol"] not in by or r.get("asof", 0) >= by[r["symbol"]].get("asof", 0):
            by[r["symbol"]] = r
    return by


def composite(r):
    em = r.get("est_mom_90d") or 0.0
    br = r.get("rev_breadth_30d") or 0.0
    return em + br      # both positive = estimates rising + analysts revising up


def close_price(cache: Path, sym: str):
    p = cache / f"{sym}_1day_bars.jsonl"
    if not p.exists():
        return None
    last = None
    for line in p.open():
        if line.strip():
            last = line
    if not last:
        return None
    try:
        return float(json.loads(last)["close"])
    except Exception:
        return None


def prev_book():
    if not TRACK.exists():
        return None
    last = None
    for line in TRACK.open():
        if line.strip():
            last = line
    return json.loads(last) if last else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/Volumes/My Passport/AlpcaData/cache_sip_10y")
    ap.add_argument("--fallback-cache", default="/Volumes/My Passport/AlpcaData/cache")
    args = ap.parse_args()
    cache, fb = Path(args.cache), Path(args.fallback_cache)
    today = time.strftime("%Y-%m-%d")
    asof = int(time.mktime(time.strptime(today, "%Y-%m-%d")))

    def px(sym):
        return close_price(cache, sym) or close_price(fb, sym)

    sigs = latest_signals()
    # ---- mark the PRIOR book to today's close -> realized return ----
    realized = None
    prev = prev_book()
    if prev and prev.get("book"):
        rets = []
        for sym, entry_px in prev["book"].items():
            now = px(sym)
            if entry_px and now and entry_px > 0:
                rets.append(now / entry_px - 1.0)
        if rets:
            realized = sum(rets) / len(rets)     # equal-weight long realized return

    # ---- form today's long book: top third by composite signal, signal>0 ----
    ranked = sorted(((s, composite(r)) for s, r in sigs.items()), key=lambda x: x[1], reverse=True)
    positive = [(s, v) for s, v in ranked if v > 0 and px(s) is not None]
    book = {}
    if len(positive) >= MIN_NAMES:
        k = max(MIN_NAMES, int(TOP_FRAC * len(positive)))
        for s, _ in positive[:k]:
            book[s] = px(s)

    row = {"date": today, "asof": asof, "n_signals": len(sigs), "n_positive": len(positive),
           "n_long": len(book), "book": book, "realized_prev": realized,
           "status": ("trading" if book else f"accumulating (need >={MIN_NAMES} positive-signal names, "
                      f"have {len(positive)}; signal ledger filling within AV quota)")}
    with TRACK.open("a") as f:
        f.write(json.dumps(row) + "\n")
    print(f"[revision] {today}: {len(sigs)} signals · {len(positive)} positive · long {len(book)} "
          f"· realized_prev {realized} · {row['status']}")


if __name__ == "__main__":
    main()
