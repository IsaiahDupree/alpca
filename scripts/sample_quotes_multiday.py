"""
Representative multi-day NBBO sampler for Phase-4 deadband calibration.

The plain downloader (download_data.py --quotes) pulls one contiguous recent window
and caps it at the FIRST N quotes — for a liquid name that is just the opening burst
of a single morning (or, worse, the pre-market). Deadbands fit on that are regime-
specific and untrustworthy (see the COVERAGE CAVEAT in analyze_microstructure.py).

This sampler instead builds a sample that spans:
  * the last --days WEEKDAYS (holidays that return 0 quotes are skipped), and
  * several intraday ET windows per day (--windows), inside the REGULAR session and
    deliberately AVOIDING the 09:30 open auction and 16:00 close,
so the |tilt| and |OFI| percentiles reflect typical trading conditions rather than
one open. Each (day, window) is capped at --limit quotes; results are concatenated,
de-duplicated, sorted by venue time, and written to <out>/<sym>_quotes.jsonl (the
exact file analyze_microstructure.py reads).

Historical IEX quotes work whether or not the market is open; NO orders are placed.
Creds load from Alpca/.env (PAPER). Feed = config.data_feed (iex).

Run:
  .venv/bin/python scripts/sample_quotes_multiday.py --symbols SPY,QQQ,AAPL \
      --days 10 --windows 10:00-10:15,12:30-12:45,15:30-15:45 --limit 12000 \
      --out "/Volumes/My Passport/AlpcaData/cache"
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ET = ZoneInfo("America/New_York")


def _parse_windows(spec: str):
    """'10:00-10:15,12:30-12:45' -> [((10,0),(10,15)), ((12,30),(12,45))]."""
    out = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        a, b = chunk.split("-")
        ah, am = (int(x) for x in a.split(":"))
        bh, bm = (int(x) for x in b.split(":"))
        out.append(((ah, am), (bh, bm)))
    return out


def _weekdays_back(n: int):
    """The last n weekdays (Mon-Fri), most recent first, as ET date objects.
    Starts from yesterday so the current (possibly-incomplete) day never skews it;
    intraday windows for a fully-elapsed today are still covered when n is large."""
    days = []
    # walk back from today; include today only if it's a weekday and fully past the
    # last window (handled by the caller via the 20-min delay). Start at today.
    d = datetime.now(ET).date()
    while len(days) < n:
        if d.weekday() < 5:  # 0=Mon .. 4=Fri
            days.append(d)
        d = d - timedelta(days=1)
    return days


def _fetch_window(cfg, sym, date_et, w_start, w_end, limit):
    from alpca.data.bars import fetch_alpaca_quotes
    s = datetime.combine(date_et, time(*w_start), tzinfo=ET)
    e = datetime.combine(date_et, time(*w_end), tzinfo=ET)
    # IEX free data is ~15 min delayed; don't request anything inside the last 20 min.
    cutoff = datetime.now(ET) - timedelta(minutes=20)
    if e > cutoff:
        return None  # window not yet available
    return fetch_alpaca_quotes(cfg, sym, start=s.astimezone(ZoneInfo("UTC")),
                               end=e.astimezone(ZoneInfo("UTC")), limit=limit)


def _key(r):
    return (r.get("timestamp"), r.get("bid"), r.get("ask"), r.get("bid_size"), r.get("ask_size"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default="SPY,QQQ,AAPL")
    ap.add_argument("--days", type=int, default=10, help="number of recent weekdays to sample")
    ap.add_argument("--windows", default="10:00-10:15,12:30-12:45,15:30-15:45",
                    help="comma list of ET intraday windows H:M-H:M (regular session)")
    ap.add_argument("--limit", type=int, default=12000, help="max quotes per (day,window)")
    ap.add_argument("--out", default="data/cache")
    args = ap.parse_args()

    from alpca.config import load_config
    cfg = load_config()
    if not cfg.has_credentials:
        print("[FAIL] no Alpaca credentials (Alpca/.env)."); return 1
    if not cfg.paper:
        print("[FAIL] refusing to run against a non-paper config."); return 1

    windows = _parse_windows(args.windows)
    weekdays = _weekdays_back(args.days)
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    out = Path(args.out)

    print(f"[ok] {cfg.describe()}")
    print(f"[ok] feed={cfg.data_feed}  symbols={symbols}")
    print(f"[ok] {len(weekdays)} weekdays {weekdays[-1]}..{weekdays[0]}  x {len(windows)} ET windows  "
          f"cap {args.limit}/window  -> {out}")

    grand = 0
    for sym in symbols:
        print(f"\n{sym}:")
        rows = {}
        days_hit = 0
        for d in weekdays:
            day_n = 0
            for (ws, we) in windows:
                try:
                    q = _fetch_window(cfg, sym, d, ws, we, args.limit)
                except Exception as ex:
                    print(f"  {d} {ws[0]:02d}:{ws[1]:02d}-{we[0]:02d}:{we[1]:02d}  FAIL: {ex}")
                    continue
                if q is None:
                    continue
                for r in q:
                    rows[_key(r)] = r
                day_n += len(q)
            if day_n:
                days_hit += 1
                print(f"  {d} ({['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][d.weekday()]}): +{day_n}")
        ordered = sorted(rows.values(), key=lambda r: r.get("timestamp", 0))
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"{sym}_quotes.jsonl"
        with path.open("w") as f:
            for r in ordered:
                f.write(json.dumps(r) + "\n")
        span = (ordered[-1]["timestamp"] - ordered[0]["timestamp"]) / 86400.0 if ordered else 0
        print(f"  -> {len(ordered)} unique quotes across {days_hit} days "
              f"(calendar span {span:.1f}d) -> {path.name}")
        grand += len(ordered)

    print(f"\n[done] {grand} quotes total -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
