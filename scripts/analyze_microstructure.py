"""
Phase-4 data-gated deadband calibration.

The microprice gate `k` and the OFI `entry`/`exit` deadbands shipped as heuristic
universal constants (see strategies/microstructure.py + order_flow.py CALIBRATION
NOTE). This tool replaces them with PER-SYMBOL values fit from the real cached NBBO
qbars, and screens out crossed/absurd quotes that would poison the fit.

SUBSTRATE: the dense, tick-level <sym>_quotes.jsonl stream (one row per NBBO
change). The 1-min <sym>_qbars.jsonl are NOT usable for this — attach_quotes_to_bars
broadcasts each sparse quote forward, so 1,200 "quoted" bars collapse to ~12 unique
books. The raw quote stream has ~12k-19k unique books per symbol.

COVERAGE (Phase-4 data gate, updated 2026-06-02): the cached stream is now a
REPRESENTATIVE multi-day sample — scripts/sample_quotes_multiday.py pulls the last
~10 weekdays × 3 regular-session ET windows (post-open / midday / pre-close),
~324k clean ticks and ~58k-113k unique books per symbol. p75|tilt| is stable across
SPY/QQQ/AAPL at ~0.50, so the microprice k=0.5 default is now a trustworthy fit (not
just a seed). NOTE the OFI numbers here are TICK-level (20-tick window); the deployed
L1OFI uses a 20-BAR window, so its deadband stays heuristic — see order_flow.py.

Method (no external deps):
  * Quote screen: keep ticks with bid<ask and spread_bps within [0, max_spread_bps].
    Absurd spreads (stale IEX top-of-book auction prints) are dropped — they bias
    the tilt toward 0.
  * microprice k: |tilt| has a heavy mass near 0 (flat book). A useful deadband
    sits at a high percentile so the gate only confirms on genuine imbalance. We
    report the 50/75/90th percentiles of |tilt| and recommend the 75th.
  * OFI entry/exit: roll the same normalized-OFI the L1OFI strategy computes
    (window=20) tick-to-tick and report |normOFI| percentiles; recommend
    entry=90th, exit=50th. NOTE the deployed L1OFI sees 1-min BARS (20-bar window =
    20 min), not ticks (20-tick window = seconds); the tick fit is a reference, and
    bar-level OFI calibration stays data-gated until per-bar NBBO history exists.

Writes a JSON report to data/microstructure_deadbands.json and prints a summary.
Run: .venv/bin/python scripts/analyze_microstructure.py --symbols SPY,QQQ,AAPL
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from alpca.strategies.microstructure import microprice_tilt  # noqa: E402
from alpca.strategies.order_flow import ofi_event  # noqa: E402


def _pctile(xs, p):
    """Linear-interpolation percentile of a sorted-able list, p in [0,100]."""
    if not xs:
        return None
    s = sorted(xs)
    if len(s) == 1:
        return s[0]
    rank = (p / 100.0) * (len(s) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(s) - 1)
    frac = rank - lo
    return s[lo] * (1 - frac) + s[hi] * frac


def _spread_bps(bid, ask):
    mid = (bid + ask) / 2.0
    return (ask - bid) / mid * 1e4 if mid > 0 else float("inf")


def analyze(symbol, cache, max_spread_bps, window):
    path = cache / f"{symbol}_quotes.jsonl"
    if not path.exists():
        return {"symbol": symbol, "error": "no raw quote stream cached"}
    rows = [json.loads(l) for l in path.open() if l.strip()]
    quoted = [r for r in rows if r.get("bid") is not None and r.get("ask") is not None]

    # quality screen
    clean, crossed, wide = [], 0, 0
    for r in quoted:
        b, a = r["bid"], r["ask"]
        if a <= b:
            crossed += 1
            continue
        sb = _spread_bps(b, a)
        if sb > max_spread_bps:
            wide += 1
            continue
        r["_spread_bps"] = sb
        clean.append(r)

    # microprice tilt distribution on clean book
    tilts = []
    for r in clean:
        t = microprice_tilt(r["bid"], r["ask"], r["bid_size"], r["ask_size"])
        if t is not None:
            tilts.append(abs(t))

    # normalized OFI (same kernel as L1OFI) over the clean stream, in time order
    clean_sorted = sorted(clean, key=lambda r: r.get("timestamp", 0))
    e_win, sz_win = deque(maxlen=window), deque(maxlen=window)
    prev = None
    ofis = []
    for r in clean_sorted:
        b, a, bs, az = r["bid"], r["ask"], r["bid_size"], r["ask_size"]
        if prev is not None:
            e = ofi_event(b, bs, a, az, prev[0], prev[1], prev[2], prev[3])
            e_win.append(e)
            sz_win.append(bs + az)
            if len(e_win) == window:
                denom = sum(sz_win) or 1.0
                ofis.append(abs(sum(e_win) / denom))
        prev = (b, bs, a, az)

    ts = [r.get("timestamp") for r in clean if r.get("timestamp")]
    span_hours = round((max(ts) - min(ts)) / 3600.0, 2) if ts else 0
    unique_books = len({(r["bid"], r["ask"], r["bid_size"], r["ask_size"]) for r in clean})

    return {
        "symbol": symbol,
        "ticks_total": len(rows),
        "quoted": len(quoted),
        "clean": len(clean),
        "unique_books": unique_books,
        "span_hours": span_hours,
        "rejected_crossed": crossed,
        "rejected_wide": wide,
        "spread_bps_p50": round(_pctile([r["_spread_bps"] for r in clean], 50) or 0, 3),
        "spread_bps_p90": round(_pctile([r["_spread_bps"] for r in clean], 90) or 0, 3),
        "tilt_abs_p50": round(_pctile(tilts, 50) or 0, 4),
        "tilt_abs_p75": round(_pctile(tilts, 75) or 0, 4),
        "tilt_abs_p90": round(_pctile(tilts, 90) or 0, 4),
        "ofi_abs_p50": round(_pctile(ofis, 50) or 0, 4),
        "ofi_abs_p90": round(_pctile(ofis, 90) or 0, 4),
        # recommendations
        "recommend_microprice_k": round(_pctile(tilts, 75) or 0.1, 3),
        "recommend_ofi_entry": round(_pctile(ofis, 90) or 0.15, 3),
        "recommend_ofi_exit": round(_pctile(ofis, 50) or 0.05, 3),
        "n_tilt": len(tilts),
        "n_ofi": len(ofis),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default="SPY,QQQ,AAPL")
    ap.add_argument("--cache", default="data/cache")
    ap.add_argument("--max-spread-bps", type=float, default=50.0,
                    help="reject quotes wider than this (default 50bps screens IEX garbage)")
    ap.add_argument("--window", type=int, default=20, help="OFI rolling window (matches L1OFI default)")
    ap.add_argument("--out", default="data/microstructure_deadbands.json")
    args = ap.parse_args()

    cache = Path(args.cache)
    reports = [analyze(s.strip(), cache, args.max_spread_bps, args.window)
               for s in args.symbols.split(",") if s.strip()]

    out = Path(args.out)
    out.write_text(json.dumps({"max_spread_bps": args.max_spread_bps,
                               "window": args.window,
                               "symbols": reports}, indent=2))

    print(f"{'sym':<5}{'clean':>8}{'uniq':>7}{'hrs':>6}{'spr50':>8}{'spr90':>8}"
          f"{'k(p75)':>8}{'ofiIn':>8}{'ofiOut':>8}")
    print("-" * 66)
    for r in reports:
        if "error" in r:
            print(f"{r['symbol']:<5} {r['error']}")
            continue
        print(f"{r['symbol']:<5}{r['clean']:>8}{r['unique_books']:>7}{r['span_hours']:>6.1f}"
              f"{r['spread_bps_p50']:>8.2f}{r['spread_bps_p90']:>8.2f}"
              f"{r['recommend_microprice_k']:>8.3f}{r['recommend_ofi_entry']:>8.3f}"
              f"{r['recommend_ofi_exit']:>8.3f}")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
