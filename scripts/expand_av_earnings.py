"""
Daily Alpha Vantage earnings-surprise expander — incrementally cache the FULL universe a
quota-respecting batch at a time, so PEAD's cross-sectional deciles tighten over ~8 days.

Alpha Vantage free tier = 25 requests/day (soft-throttles ~1 req/sec). This pulls the next
`--batch` UNCACHED symbols each run, skips already-cached ones, and stops cleanly when the
daily quota is hit (RateLimited) — fully resumable. When every universe symbol is cached it
becomes a no-op and exits 0. Meant to be fired once/day by launchd (com.alpca.avearnings).

  .venv/bin/python scripts/expand_av_earnings.py --batch 23
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/Volumes/My Passport/AlpcaData/cache",
                    help="daily-bars dir (defines the universe)")
    ap.add_argument("--out", default="/Volumes/My Passport/AlpcaData/earnings_av")
    ap.add_argument("--batch", type=int, default=23, help="symbols to fetch this run (<=24, keep < 25/day)")
    ap.add_argument("--delay", type=float, default=2.5, help="seconds between calls (AV ~1 req/sec)")
    args = ap.parse_args()

    from alpca.config import load_config           # triggers .env load (ALPHAVANTAGE_API_KEY)
    load_config()
    import os
    if not os.environ.get("ALPHAVANTAGE_API_KEY"):
        print("[abort] ALPHAVANTAGE_API_KEY not set (.env)", file=sys.stderr)
        return 1
    from alpca.data.earnings import download_alphavantage_earnings

    cache, out = Path(args.cache), Path(args.out)
    universe = sorted(p.name.split("_1day_")[0] for p in cache.glob("*_1day_bars.jsonl"))
    cached = {p.name.replace("_earnings.json", "") for p in out.glob("*_earnings.json")}
    remaining = [s for s in universe if s not in cached]

    print(f"[av-earnings] universe {len(universe)}  cached {len(cached)}  remaining {len(remaining)}")
    if not remaining:
        print("[av-earnings] COMPLETE — every universe symbol cached. (no-op)")
        return 0

    batch = remaining[:max(1, min(args.batch, 24))]
    print(f"[av-earnings] fetching {len(batch)}: {', '.join(batch)}")
    counts = download_alphavantage_earnings(batch, out, delay_s=args.delay)
    got = sum(1 for s in batch if (out / f"{s}_earnings.json").exists() and s in counts)
    total_cached = len({p.name.replace('_earnings.json', '') for p in out.glob('*_earnings.json')})
    print(f"[av-earnings] fetched {got}/{len(batch)} this run, {sum(counts.values())} events; "
          f"now {total_cached}/{len(universe)} cached, {len(universe)-total_cached} to go")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
