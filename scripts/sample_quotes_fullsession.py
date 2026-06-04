"""
Full-session NBBO sampler — unlocks BAR-LEVEL OFI calibration.

The deadband sampler (sample_quotes_multiday.py) pulls 3 short windows/day, which is
right for the instantaneous microprice tilt but leaves big gaps between windows — so
the bar-level L1OFI (20 one-minute bars = 20 min rolling) can never fill its window.
This sampler instead covers the ENTIRE regular session (09:30-16:00 ET) at a fine
cadence so EVERY contiguous 1-min bar gets a fresh quote.

Efficiency: a windowed quote fetch returns the EARLIEST `limit` quotes, so one fetch
can't cover a long span without pulling millions of ticks. We instead make one small
fetch every `--step-min` minutes (default 3) across the session — ~130 tiny fetches/
day — giving a quote near every step boundary. With attach_quotes_to_bars(max_
staleness_s>step) every 1-min bar then finds a non-stale NBBO. Far cheaper than
downloading the full tick stream, and exactly enough resolution for 1-min bars.

Outputs (kept SEPARATE from the deadband-window files so neither clobbers the other):
  <out>/<sym>_quotes_fullsession.jsonl
  <out>/<sym>_1min_qbars_fullsession.jsonl     (contiguous bars + prevailing NBBO)

Historical IEX quotes work whether or not the market is open; NO orders are placed.

Run:
  .venv/bin/python scripts/sample_quotes_fullsession.py --symbols SPY,QQQ,AAPL \
      --days 5 --step-min 3 --limit 50 --out "/Volumes/My Passport/AlpcaData/cache"
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
UTC = ZoneInfo("UTC")
SESSION_OPEN = (9, 30)
SESSION_CLOSE = (16, 0)


def _weekdays_back(n: int):
    days, d = [], datetime.now(ET).date()
    while len(days) < n:
        if d.weekday() < 5:
            days.append(d)
        d -= timedelta(days=1)
    return days


def _session_steps(date_et, step_min: int):
    """Yield (start, end) UTC datetimes for each step-min chunk of the RTH session."""
    s = datetime.combine(date_et, time(*SESSION_OPEN), tzinfo=ET)
    close = datetime.combine(date_et, time(*SESSION_CLOSE), tzinfo=ET)
    cutoff = datetime.now(ET) - timedelta(minutes=20)  # IEX ~15min delay
    while s < close:
        e = min(close, s + timedelta(minutes=step_min))
        if e <= cutoff:
            yield s.astimezone(UTC), e.astimezone(UTC)
        s = e


def _key(r):
    return (r.get("timestamp"), r.get("bid"), r.get("ask"), r.get("bid_size"), r.get("ask_size"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default="SPY,QQQ,AAPL")
    ap.add_argument("--days", type=int, default=5)
    ap.add_argument("--step-min", type=int, default=3, help="minutes between fetches across the session")
    ap.add_argument("--limit", type=int, default=50, help="max quotes per step fetch")
    ap.add_argument("--staleness-s", type=float, default=210.0, help="max quote age when attaching to a bar")
    ap.add_argument("--out", default="data/cache")
    args = ap.parse_args()

    from alpca.config import load_config
    from alpca.data.bars import attach_quotes_to_bars, fetch_alpaca_bars, fetch_alpaca_quotes

    cfg = load_config()
    if not cfg.has_credentials:
        print("[FAIL] no Alpaca credentials (Alpca/.env)."); return 1
    if not cfg.paper:
        print("[FAIL] refusing to run against a non-paper config."); return 1

    weekdays = _weekdays_back(args.days)
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"[ok] {cfg.describe()}")
    print(f"[ok] feed={cfg.data_feed}  symbols={symbols}")
    print(f"[ok] {len(weekdays)} weekdays {weekdays[-1]}..{weekdays[0]}  step={args.step_min}min "
          f"cap {args.limit}/step  staleness={args.staleness_s:.0f}s  -> {out}")

    grand_q = grand_b = 0
    for sym in symbols:
        print(f"\n{sym}:")
        rows = {}
        bars_all = []
        for d in weekdays:
            day_q = 0
            for (s, e) in _session_steps(d, args.step_min):
                try:
                    q = fetch_alpaca_quotes(cfg, sym, start=s, end=e, limit=args.limit)
                except Exception as ex:
                    print(f"  {d} {s.time()} FAIL: {ex}")
                    continue
                for r in q:
                    rows[_key(r)] = r
                day_q += len(q)
            # full-session 1-min bars for this day
            ds = datetime.combine(d, time(*SESSION_OPEN), tzinfo=ET).astimezone(UTC)
            de = datetime.combine(d, time(*SESSION_CLOSE), tzinfo=ET).astimezone(UTC)
            try:
                bars = fetch_alpaca_bars(cfg, sym, timeframe="1min", start=ds, end=de)
            except Exception as ex:
                print(f"  {d} bars FAIL: {ex}"); bars = []
            bars_all.extend(bars)
            if day_q:
                print(f"  {d} ({['Mon','Tue','Wed','Thu','Fri'][d.weekday()]}): {day_q} quotes, {len(bars)} bars")

        quotes = sorted(rows.values(), key=lambda r: r.get("timestamp", 0))
        with (out / f"{sym}_quotes_fullsession.jsonl").open("w") as f:
            for r in quotes:
                f.write(json.dumps(r) + "\n")

        # contiguous qbars: attach prevailing NBBO to each 1-min bar (per-bar, no gaps)
        bars_all.sort(key=lambda b: b.get("timestamp", 0))
        qbars = attach_quotes_to_bars(bars_all, quotes, max_staleness_s=args.staleness_s)
        covered = sum(1 for b in qbars if b.get("bid") and b.get("ask"))
        with (out / f"{sym}_1min_qbars_fullsession.jsonl").open("w") as f:
            for b in qbars:
                f.write(json.dumps(b) + "\n")
        pct = 100.0 * covered / len(qbars) if qbars else 0
        print(f"  -> {len(quotes)} quotes | {len(qbars)} bars, {covered} with fresh NBBO ({pct:.0f}%)")
        grand_q += len(quotes); grand_b += len(qbars)

    print(f"\n[done] quotes={grand_q} qbars={grand_b} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
