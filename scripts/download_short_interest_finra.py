"""
Download MULTI-YEAR short interest from FINRA (consolidatedShortInterest, public, no auth) for the
universe and cache to the Passport. FINRA gives ~9 years of bi-monthly settlement points per symbol
(vs Nasdaq's ~1 yr), which fully covers our 5-yr daily window INCLUDING the 2022 bear — the
cross-regime depth Case 21's short-interest tilt needs. Normalizes to the same schema the existing
`backtest_short_interest_tilt` reads (settlement MM/DD/YYYY, interest, avg_vol, days_to_cover), with
days-to-cover computed precisely as short / avg-volume (FINRA's own field is rounded to an integer).

Run: .venv/bin/python scripts/download_short_interest_finra.py --cache "/Volumes/My Passport/AlpcaData/cache"
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path

URL = "https://api.finra.org/data/group/otcMarket/name/consolidatedShortInterest"
HDR = {"Content-Type": "application/json", "Accept": "application/json"}


def fetch_finra_si(sym: str, timeout: float = 30.0):
    body = json.dumps({"limit": 1000, "compareFilters": [
        {"fieldName": "symbolCode", "fieldValue": sym, "compareType": "EQUAL"}]}).encode()
    with urllib.request.urlopen(urllib.request.Request(URL, data=body, headers=HDR), timeout=timeout) as r:
        rows = json.load(r)
    out = []
    for row in rows:
        sd = row.get("settlementDate")          # YYYY-MM-DD
        cur = row.get("currentShortPositionQuantity")
        vol = row.get("averageDailyVolumeQuantity")
        if not sd or cur is None or not vol:
            continue
        try:
            y, m, d = sd.split("-")
            dtc = float(cur) / float(vol) if float(vol) > 0 else 0.0
            out.append({"settlement": f"{m}/{d}/{y}", "interest": float(cur),
                        "avg_vol": float(vol), "days_to_cover": dtc})
        except (ValueError, ZeroDivisionError):
            continue
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/Volumes/My Passport/AlpcaData/cache")
    ap.add_argument("--out", default="/Volumes/My Passport/AlpcaData/short_interest_finra")
    ap.add_argument("--delay", type=float, default=0.3)
    ap.add_argument("--max-symbols", type=int, default=195)
    args = ap.parse_args()
    cache, out = Path(args.cache), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    syms = sorted(p.name.split("_1day_")[0] for p in cache.glob("*_1day_bars.jsonl"))[:args.max_symbols]
    have = {p.name.replace("_si.json", "") for p in out.glob("*_si.json")}
    todo = [s for s in syms if s not in have]
    print(f"[finra-si] universe {len(syms)}, cached {len(have)}, fetching {len(todo)}")
    ok = fail = 0
    for i, s in enumerate(todo, 1):
        try:
            rows = fetch_finra_si(s)
            if rows:
                (out / f"{s}_si.json").write_text(json.dumps(rows))
                ok += 1
            else:
                fail += 1
            if i % 25 == 0:
                print(f"  {i}/{len(todo)} (ok {ok}, empty {fail})")
        except Exception as e:
            fail += 1
            print(f"  {s}: {type(e).__name__} {str(e)[:80]}")
        time.sleep(args.delay)
    total = len({p.name.replace('_si.json', '') for p in out.glob('*_si.json')})
    print(f"[finra-si] done: +{ok} this run, {fail} empty/fail, {total}/{len(syms)} cached")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
