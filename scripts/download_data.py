"""
Download REAL Alpaca historical data into a local cache for offline backtests.

BARS scale to years — they are paged in timeframe-appropriate chunks and written
incrementally (deduped by timestamp), so a multi-year 1-min pull is safe.
QUOTES cannot go multi-year (tick data is billions of rows), so NBBO quotes are
pulled for a recent window only, per-day for even coverage, and merged onto recent
bars to make microprice qbars.

Per symbol:
  - bars  -> <out>/<sym>_<tf>_bars.jsonl                       (full --years history)
  - quotes-> <out>/<sym>_quotes.jsonl              (--quotes; recent --quote-days)
  - qbars -> <out>/<sym>_<tf>_qbars.jsonl          (--quotes; recent bars + NBBO)

Historical data works whether or not the market is open; NO orders are ever placed.
Credentials load from Alpca/.env (PAPER). Feed = config.data_feed (iex default).

Usage:
  python scripts/download_data.py --symbols SPY,QQQ,AAPL --timeframe 1min --years 3 --quotes \
      --out "/Volumes/My Passport/AlpcaData/cache"
  python scripts/download_data.py --symbols SPY,QQQ,AAPL,MSFT,NVDA,IWM --timeframe 1day --years 5 \
      --out "/Volumes/My Passport/AlpcaData/cache"
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# How wide a window to request per chunk, by timeframe (keeps each request sane).
_TF_CHUNK_DAYS = {"1min": 30, "5min": 90, "15min": 180, "1hour": 3650, "1day": 3650}


def _write_jsonl(path: Path, rows) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return len(rows)


def _iter_chunks(start: datetime, end: datetime, chunk_days: int):
    cur = start
    while cur < end:
        nxt = min(end, cur + timedelta(days=chunk_days))
        yield cur, nxt
        cur = nxt


def _fetch_bars_chunked(cfg, sym, tf, start, end, out_path: Path) -> int:
    from alpca.data.bars import fetch_alpaca_bars
    chunk_days = _TF_CHUNK_DAYS.get(tf, 90)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    seen = set()
    n = 0
    with out_path.open("w") as f:
        for cs, ce in _iter_chunks(start, end, chunk_days):
            try:
                bars = fetch_alpaca_bars(cfg, sym, timeframe=tf, start=cs, end=ce)
            except Exception as e:
                print(f"    {cs.date()}..{ce.date()}  FAIL: {e}")
                continue
            wrote = 0
            for b in bars:
                ts = b.get("timestamp")
                if ts in seen:
                    continue
                seen.add(ts)
                f.write(json.dumps(b) + "\n")
                wrote += 1
                n += 1
            if wrote:
                print(f"    {cs.date()}..{ce.date()}: +{wrote}  (total {n})")
    return n


def _fetch_quotes_recent(cfg, sym, quote_days, per_day_limit, out_path: Path):
    from alpca.data.bars import fetch_alpaca_quotes
    end = datetime.now(timezone.utc) - timedelta(minutes=20)
    allq = []
    for d in range(quote_days):
        de = end - timedelta(days=d)
        ds = de - timedelta(days=1)
        try:
            q = fetch_alpaca_quotes(cfg, sym, start=ds, end=de, limit=per_day_limit)
        except Exception as e:
            print(f"    quotes {ds.date()}  FAIL: {e}")
            continue
        allq.extend(q)
        print(f"    quotes {ds.date()}..{de.date()}: +{len(q)}")
    allq.sort(key=lambda r: r["timestamp"])
    _write_jsonl(out_path, allq)
    return allq


def _coverage(qbars):
    from alpca.strategies.microstructure import microprice_signal
    have = sum(1 for b in qbars if b.get("bid") and b.get("ask"))
    nonflat = sum(1 for b in qbars
                  if microprice_signal(b.get("bid"), b.get("ask"),
                                       b.get("bid_size"), b.get("ask_size")) in ("bull", "bear"))
    return have, nonflat


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default="SPY")
    ap.add_argument("--timeframe", default="1min")
    ap.add_argument("--years", type=float, default=0.0, help="history length in years (takes precedence)")
    ap.add_argument("--days", type=int, default=5, help="used when --years is 0")
    ap.add_argument("--quotes", action="store_true", help="also pull recent NBBO quotes + build qbars")
    ap.add_argument("--quote-days", type=int, default=2)
    ap.add_argument("--quote-limit", type=int, default=200_000, help="max quotes per day")
    ap.add_argument("--out", default="data/cache")
    args = ap.parse_args()

    from alpca.config import load_config
    from alpca.data.bars import attach_quotes_to_bars, fetch_alpaca_bars

    cfg = load_config()
    if not cfg.has_credentials:
        print("[FAIL] no Alpaca credentials (Alpca/.env).")
        return 1
    if not cfg.paper:
        print("[FAIL] refusing to run against a non-paper config.")
        return 1

    end = datetime.now(timezone.utc) - timedelta(minutes=20)
    start = end - timedelta(days=args.years * 365.25) if args.years > 0 else end - timedelta(days=args.days)
    span = f"{args.years}y" if args.years > 0 else f"{args.days}d"
    out = Path(args.out)
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]

    print(f"[ok] {cfg.describe()}")
    print(f"[ok] feed={cfg.data_feed}  symbols={symbols}  tf={args.timeframe}  span={span} "
          f"({start.date()}..{end.date()})" + ("  +quotes" if args.quotes else ""))
    print(f"[ok] out -> {out}")

    grand = {"bars": 0, "quotes": 0, "qbars": 0}
    for sym in symbols:
        print(f"\n{sym}:")
        n_bars = _fetch_bars_chunked(cfg, sym, args.timeframe, start, end,
                                     out / f"{sym}_{args.timeframe}_bars.jsonl")
        grand["bars"] += n_bars
        print(f"  {sym}: {n_bars} bars")

        if args.quotes:
            quotes = _fetch_quotes_recent(cfg, sym, args.quote_days, args.quote_limit,
                                          out / f"{sym}_quotes.jsonl")
            grand["quotes"] += len(quotes)
            # qbars only over recent bars matching the quote window (cheap, fresh fetch)
            try:
                recent = fetch_alpaca_bars(cfg, sym, timeframe=args.timeframe, days=args.quote_days + 1)
            except Exception as e:
                print(f"  qbars: recent-bar fetch FAIL: {e}")
                recent = []
            if recent and quotes:
                # only attach quotes that are fresh vs the bar (avoid cross-day
                # stale quotes contaminating microprice/TCA when NBBO history is thin)
                qbars = attach_quotes_to_bars(recent, quotes, max_staleness_s=180.0)
                n_qb = _write_jsonl(out / f"{sym}_{args.timeframe}_qbars.jsonl", qbars)
                have, nonflat = _coverage(qbars)
                grand["qbars"] += n_qb
                print(f"  {sym}: qbars {have}/{n_qb} quoted, {nonflat} non-flat microprice tilt")

    print(f"\n[done] {out}/   bars={grand['bars']} quotes={grand['quotes']} qbars={grand['qbars']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
