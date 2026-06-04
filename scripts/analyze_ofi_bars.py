"""
Bar-level L1OFI deadband calibration — the Phase-4 item that was data-gated.

Reads the contiguous full-session qbars (<sym>_1min_qbars_fullsession.jsonl from
sample_quotes_fullsession.py) and runs the EXACT normalized-OFI computation the
deployed L1OFI strategy uses (alpca.strategies.order_flow.ofi_event over a rolling
`--window` of 1-min BARS), segmented per trading day so no window straddles a
session boundary. It collects the |normOFI| distribution and recommends:
  entry = p90  (only a top-decile sustained imbalance triggers an entry)
  exit  = p50  (fade back to the median = flat)
matching how L1OFI uses entry/exit. This is the BAR-level fit that the tick-level
analyze_microstructure.py explicitly could not provide.

Writes data/ofi_deadbands.json and prints a summary. Does NOT edit strategy defaults
(apply step is separate, after reviewing the numbers).

Run:
  .venv/bin/python scripts/analyze_ofi_bars.py --symbols SPY,QQQ,AAPL --window 20 \
      --cache "/Volumes/My Passport/AlpcaData/cache"
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from alpca.strategies.order_flow import ofi_event  # noqa: E402


def _pctile(xs, p):
    if not xs:
        return None
    s = sorted(xs)
    if len(s) == 1:
        return s[0]
    rank = (p / 100.0) * (len(s) - 1)
    lo = int(rank); hi = min(lo + 1, len(s) - 1); frac = rank - lo
    return s[lo] * (1 - frac) + s[hi] * frac


def _et_date(ts):
    # ET date label for day-segmentation (DST-correct enough via fixed -4/-5 not needed:
    # group by UTC-date-shifted-to-ET using a simple offset is risky, so use date in UTC
    # minus 4h which keeps a full RTH session (13:30-20:00 UTC) on one calendar date).
    return datetime.fromtimestamp(ts - 4 * 3600, tz=timezone.utc).date()


def analyze(symbol, cache, window):
    path = cache / f"{symbol}_1min_qbars_fullsession.jsonl"
    if not path.exists():
        return {"symbol": symbol, "error": "no full-session qbars cached"}
    bars = [json.loads(l) for l in path.open() if l.strip()]
    bars = [b for b in bars if b.get("bid") and b.get("ask")
            and b.get("bid_size") and b.get("ask_size")]
    bars.sort(key=lambda b: b.get("timestamp", 0))

    # segment by trading day so a 20-bar window never spans an overnight gap
    days = {}
    for b in bars:
        days.setdefault(_et_date(b["timestamp"]), []).append(b)

    ofis = []
    used_days = 0
    for _, day_bars in sorted(days.items()):
        if len(day_bars) <= window:
            continue
        used_days += 1
        e_win, sz_win = deque(maxlen=window), deque(maxlen=window)
        prev = None
        for b in day_bars:
            bid, ask, bs, az = b["bid"], b["ask"], b["bid_size"], b["ask_size"]
            if prev is not None:
                e = ofi_event(bid, bs, ask, az, prev[0], prev[1], prev[2], prev[3])
                e_win.append(e)
                sz_win.append(bs + az)
                if len(e_win) == window:
                    denom = sum(sz_win) or 1.0
                    ofis.append(abs(sum(e_win) / denom))
            prev = (bid, bs, ask, az)

    return {
        "symbol": symbol,
        "bars_quoted": len(bars),
        "days": used_days,
        "window_bars": window,
        "n_ofi": len(ofis),
        "ofi_abs_p50": round(_pctile(ofis, 50) or 0, 4),
        "ofi_abs_p75": round(_pctile(ofis, 75) or 0, 4),
        "ofi_abs_p90": round(_pctile(ofis, 90) or 0, 4),
        "ofi_abs_p95": round(_pctile(ofis, 95) or 0, 4),
        "recommend_entry": round(_pctile(ofis, 90) or 0.15, 3),
        "recommend_exit": round(_pctile(ofis, 50) or 0.05, 3),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default="SPY,QQQ,AAPL")
    ap.add_argument("--cache", default="data/cache")
    ap.add_argument("--window", type=int, default=20, help="rolling window in BARS (L1OFI default 20)")
    ap.add_argument("--out", default="data/ofi_deadbands.json")
    args = ap.parse_args()

    cache = Path(args.cache)
    reports = [analyze(s.strip(), cache, args.window) for s in args.symbols.split(",") if s.strip()]
    Path(args.out).write_text(json.dumps({"window_bars": args.window, "symbols": reports}, indent=2))

    print(f"{'sym':<5}{'days':>5}{'nOFI':>7}{'p50':>8}{'p75':>8}{'p90':>8}{'p95':>8}{'entry':>8}{'exit':>8}")
    print("-" * 65)
    for r in reports:
        if "error" in r:
            print(f"{r['symbol']:<5} {r['error']}"); continue
        print(f"{r['symbol']:<5}{r['days']:>5}{r['n_ofi']:>7}{r['ofi_abs_p50']:>8.3f}{r['ofi_abs_p75']:>8.3f}"
              f"{r['ofi_abs_p90']:>8.3f}{r['ofi_abs_p95']:>8.3f}{r['recommend_entry']:>8.3f}{r['recommend_exit']:>8.3f}")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
