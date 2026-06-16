"""
Unified forward-track reporter — the single live view of the DEPLOYED PORTFOLIO.

Each sleeve (pairs / short-vol / momentum) logs its own track jsonl; this reads all of them, blends the
realized returns at the codified deployed weights (alpca/live/portfolio.py), and prints the combined live
OOS curve + per-sleeve breakdown. This is what to WATCH as the forward track bakes — especially the
combined drawdown (short-vol's tail is the risk). NO trading; pure read + report.

Run (after the daily forward_track.sh, or anytime):  .venv/bin/python scripts/report_forward_track.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from alpca.live.portfolio import DEPLOYED, deployed_weights, combine_tracks  # noqa: E402
from alpca.backtest.evaluation import sharpe_of, max_drawdown_of  # noqa: E402

PPY = 252.0
TRACKS = {"pairs": "data/pairs_forward_track.jsonl",
          "short_vol": "data/shortvol_forward_track.jsonl",
          "momentum": "data/momentum_forward_track.jsonl"}


def _read_track(path: str):
    p = Path(path)
    if not p.exists():
        return {}
    out = {}
    for line in p.read_text().splitlines():
        if not line.strip():
            continue
        e = json.loads(line)
        r = e.get("realized_prev")
        if isinstance(r, (int, float)) and e.get("asof"):
            out[int(e["asof"])] = r
    return out


def _curve_stats(daily):
    if len(daily) < 2:
        return None
    eq = [1.0]
    for x in daily:
        eq.append(eq[-1] * (1 + x))
    return {"cum_pct": (eq[-1] - 1) * 100, "sharpe": sharpe_of(eq, PPY),
            "maxdd_pct": max_drawdown_of(eq) * 100, "n": len(daily)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks", default=None, help="optional JSON map sleeve->path (else defaults)")
    args = ap.parse_args()
    tracks = json.loads(args.tracks) if args.tracks else TRACKS

    w = deployed_weights()
    print("=== DEPLOYED PORTFOLIO ===")
    for s in DEPLOYED:
        cap = f" (cap {s.cap:.0%})" if s.cap is not None else ""
        print(f"  {s.name:>10} [{s.role:<11}] weight {s.weight:>5.0%}{cap}  — {s.rationale}")
    print()

    track_returns = {k: _read_track(p) for k, p in tracks.items()}
    print("=== PER-SLEEVE LIVE TRACK (realized) ===")
    for k in tracks:
        tr = track_returns.get(k, {})
        st = _curve_stats([tr[t] for t in sorted(tr)]) if tr else None
        funded = "funded" if w.get(k, 0) > 0 else "PROBATION (0 capital)"
        if st:
            print(f"  {k:>10} [{funded}] {st['n']} marks · cum {st['cum_pct']:+.2f}% · "
                  f"Sharpe {st['sharpe']:.2f} · maxDD {st['maxdd_pct']:.1f}%")
        else:
            print(f"  {k:>10} [{funded}] {len(tr)} marks · need ≥2 realized returns to score")
    print()

    book = combine_tracks(track_returns)
    st = _curve_stats(book.daily_returns)
    print("=== COMBINED DEPLOYED BOOK (funded sleeves, deployed weights) ===")
    print(f"  funded weights: { {k: round(v,2) for k,v in book.weights.items()} }")
    if st:
        last = time.strftime('%Y-%m-%d', time.gmtime(book.dates[-1])) if book.dates else "?"
        print(f"  {st['n']} marked days (through {last}) · cumulative {st['cum_pct']:+.2f}% · "
              f"live Sharpe {st['sharpe']:.2f} · maxDD {st['maxdd_pct']:.1f}%")
        print("  ^ the number that adjudicates the whole program — watch the drawdown (short-vol tail).")
    else:
        print("  need ≥2 days with realized returns to mark the combined curve. Run the forward track "
              "daily (com.alpca.forwardtrack) — this fills in as it bakes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
