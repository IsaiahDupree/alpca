"""
Deploy the short-volatility / VRP second leg (Case 49) as a SHADOW FORWARD PAPER TRACK — the first
candidate that genuinely LIFTS the combined book (pairs 0.83 → 1.08, DSR 0.90, uncorrelated ρ=0.04).

The sleeve is a small long-SVXY position (post-2018 −0.5× short-VIX ETF; borrow-free, just hold it).
NO signal logic — the VRP is a static premium harvest. Each run marks the prior position to today's
close (realized OOS return) and logs the live curve.

TAIL DISCIPLINE (non-negotiable — short-vol is negatively skewed with an un-sampled catastrophic tail;
2021–2026 had no volmageddon): sized at a HARD-CAPPED tiny weight (default 8% notional, below the 12%
the inverse-vol combiner assigns) so a vol spike — the known failure mode — can't dominate the book.
This is a forward experiment with the tail respected, not a conviction bet. Run daily:
  .venv/bin/python scripts/deploy_shortvol_paper.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from alpca.backtest.evaluation import sharpe_of, max_drawdown_of  # noqa: E402

PPY = 252.0


def _load(c):
    out = {}
    for p in Path(c).glob("*_1day_bars.jsonl"):
        rows = [json.loads(l) for l in p.open() if l.strip()]
        if rows:
            out[p.name.split("_1day_")[0]] = rows
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vol-cache", default="/Volumes/My Passport/AlpcaData/cache_vol")
    ap.add_argument("--symbol", default="SVXY")
    ap.add_argument("--track", default="data/shortvol_forward_track.jsonl")
    ap.add_argument("--weight", type=float, default=0.08, help="HARD-CAPPED notional (tail discipline)")
    args = ap.parse_args()
    vol, track = _load(args.vol_cache), Path(args.track)
    bars = vol.get(args.symbol)
    if not bars:
        print(f"[deploy] no {args.symbol} bars in {args.vol_cache} — run the SIP refresh first."); return 1
    last = float(sorted(bars, key=lambda b: int(b["timestamp"]))[-1]["close"])
    asof = max(int(b["timestamp"]) for b in bars)
    print(f"[deploy] short-vol sleeve = long {args.symbol} @ HARD-CAPPED {args.weight:.0%} notional "
          f"(tail-respected; Case 49 — combined-lift leg, NOT a conviction bet).\n")

    prior = None
    if track.exists():
        lines = [l for l in track.read_text().splitlines() if l.strip()]
        if lines:
            prior = json.loads(lines[-1])
    realized = None
    if prior and prior.get("price"):
        realized = prior["weight"] * (last / prior["price"] - 1.0)
        print(f"[track] prior ({prior['date']}) marked to today: realized {realized*100:+.3f}%")

    entry = {"date": time.strftime("%Y-%m-%d", time.gmtime(asof)), "asof": asof, "symbol": args.symbol,
             "weight": args.weight, "price": last, "realized_prev": realized}
    with track.open("a") as f:
        f.write(json.dumps(entry) + "\n")
    print(f"[book] long {args.symbol} {args.weight:+.2%} @ {last:.2f}")

    rets = [json.loads(l).get("realized_prev") for l in track.read_text().splitlines() if l.strip()]
    rets = [x for x in rets if isinstance(x, (int, float))]
    if len(rets) >= 2:
        eq = [1.0]
        for x in rets:
            eq.append(eq[-1] * (1 + x))
        print(f"\n[forward-track] {len(rets)} marked periods · cumulative {(eq[-1]-1)*100:+.2f}% · "
              f"live Sharpe {sharpe_of(eq, PPY):.2f} · maxDD {max_drawdown_of(eq)*100:.1f}% "
              f"(watch the drawdown — the tail is the risk)")
    else:
        print(f"\n[forward-track] logged {entry['date']}; need ≥2 entries to mark a realized return.")
    print(f"[done] appended to {track}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
