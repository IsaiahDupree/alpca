"""
Download REAL short-interest history (Nasdaq, free) for the universe and cache it to the Passport
drive. Short interest / days-to-cover is the fundamental driver of borrow fees — crowded shorts
go expensive/hard-to-borrow — so this is the honest data behind a "borrow-fee tilt" (scout #1),
NOT a price proxy. Nasdaq gives ~1yr of bi-monthly settlement rows per symbol (24 points).

Run: .venv/bin/python scripts/download_short_interest.py --cache "/Volumes/My Passport/AlpcaData/cache"
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path

UA = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}


def fetch_short_interest(sym: str, timeout: float = 20.0):
    url = f"https://api.nasdaq.com/api/quote/{sym}/short-interest?assetClass=stocks&limit=200"
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout) as r:
        d = json.load(r)
    rows = (d.get("data") or {}).get("shortInterestTable", {}).get("rows") or []
    out = []
    for row in rows:
        try:
            out.append({
                "settlement": row["settlementDate"],                       # MM/DD/YYYY
                "interest": float(row["interest"].replace(",", "")),
                "avg_vol": float(row["avgDailyShareVolume"].replace(",", "")),
                "days_to_cover": float(row["daysToCover"]),
            })
        except (KeyError, ValueError, AttributeError):
            continue
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/Volumes/My Passport/AlpcaData/cache")
    ap.add_argument("--out", default="/Volumes/My Passport/AlpcaData/short_interest")
    ap.add_argument("--delay", type=float, default=0.6)
    ap.add_argument("--max-symbols", type=int, default=195)
    args = ap.parse_args()
    cache, out = Path(args.cache), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    syms = sorted(p.name.split("_1day_")[0] for p in cache.glob("*_1day_bars.jsonl"))[:args.max_symbols]
    have = {p.name.replace("_si.json", "") for p in out.glob("*_si.json")}
    todo = [s for s in syms if s not in have]
    print(f"[si] universe {len(syms)}, cached {len(have)}, fetching {len(todo)}")
    ok = fail = 0
    for i, s in enumerate(todo, 1):
        try:
            rows = fetch_short_interest(s)
            if rows:
                (out / f"{s}_si.json").write_text(json.dumps(rows))
                ok += 1
            else:
                fail += 1
            if i % 25 == 0:
                print(f"  {i}/{len(todo)} (ok {ok}, empty/fail {fail})")
        except Exception as e:
            fail += 1
            print(f"  {s}: {type(e).__name__} {e}")
        time.sleep(args.delay)
    total = len({p.name.replace('_si.json', '') for p in out.glob('*_si.json')})
    print(f"[si] done: +{ok} this run, {fail} empty/fail, {total}/{len(syms)} cached")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
