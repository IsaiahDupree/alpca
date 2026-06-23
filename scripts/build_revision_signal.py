"""
Collect the analyst-revision-drift signal snapshots into a dated ledger (data/revision_signal.jsonl).

For each symbol, AV EARNINGS_ESTIMATES gives the current consensus + 90-days-ago + up/down revision
counts -> we store the estimate-momentum + revision-breadth signal, stamped with the snapshot date.
Because AV only serves a CURRENT snapshot (no PIT history), this builds the revision history GOING
FORWARD, one daily slice at a time. Quota-aware (AV free ~25/day shared with the earnings job):
--max-calls caps it; --resume skips symbols already snapshotted today.

Run: .venv/bin/python scripts/build_revision_signal.py --max-calls 12
Universe defaults to the earnings_av symbols (liquid large-caps we already track).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from alpca.data.estimates import fetch_estimates, av_key  # noqa: E402

OUT = ROOT / "data" / "revision_signal.jsonl"


def universe(earnings_dir: str):
    p = Path(earnings_dir)
    if p.exists():
        return sorted(f.name.replace("_earnings.json", "") for f in p.glob("*_earnings.json"))
    return ["AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "JPM", "V", "UNH", "XOM"]


def already_today():
    if not OUT.exists():
        return set()
    today = time.strftime("%Y-%m-%d")
    out = set()
    for line in OUT.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("date") == today:
            out.add(r["symbol"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--earnings", default="/Volumes/My Passport/AlpcaData/earnings_av")
    ap.add_argument("--symbols", default="")
    ap.add_argument("--max-calls", type=int, default=12, help="AV free ~25/day, shared w/ earnings job")
    ap.add_argument("--sleep", type=float, default=1.0)
    args = ap.parse_args()

    syms = [s.strip().upper() for s in args.symbols.split(",") if s.strip()] or universe(args.earnings)
    done = already_today()
    todo = [s for s in syms if s not in done][:args.max_calls]
    key = av_key()
    today = time.strftime("%Y-%m-%d")
    print(f"[rev] universe {len(syms)} · {len(done)} already today · fetching {len(todo)} (cap {args.max_calls})")

    n_ok = 0
    with OUT.open("a") as f:
        for s in todo:
            r = fetch_estimates(s, key)
            if r is None:
                print(f"  {s}: miss/rate-limited — stopping (quota likely exhausted)")
                break
            r["date"] = today
            r["asof"] = int(time.mktime(time.strptime(today, "%Y-%m-%d")))
            f.write(json.dumps(r) + "\n")
            n_ok += 1
            print(f"  {s}: est_mom_90d={r['est_mom_90d']} breadth={r['rev_breadth_30d']} "
                  f"(analysts {r['analyst_count']:.0f})")
            time.sleep(args.sleep)
    total = sum(1 for _ in OUT.open()) if OUT.exists() else 0
    print(f"[rev] +{n_ok} snapshots today -> {OUT} ({total} total rows)")


if __name__ == "__main__":
    main()
