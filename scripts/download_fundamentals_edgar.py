"""
Download annual fundamentals from SEC EDGAR (companyfacts XBRL API — FREE, no quota, no auth, full
multi-year coverage) for the cached universe, and cache the three tags the accruals anomaly needs:
NetIncomeLoss, NetCashProvidedByUsedInOperatingActivities (operating CFO), Assets. Per symbol we
emit one row per fiscal year where all three are present:
    {fy_end (YYYY-MM-DD), filed (YYYY-MM-DD), net_income, cfo, total_assets}

NO LOOK-AHEAD: `filed` is the 10-K filing date — the accrual signal is only public then, ~2 months
after the fiscal year ends. The backtest keys off `filed`, never `fy_end`.

EDGAR requires a descriptive User-Agent and asks for <=10 req/sec. Run:
  .venv/bin/python scripts/download_fundamentals_edgar.py --cache "/Volumes/My Passport/AlpcaData/cache"
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from datetime import date
from pathlib import Path

UA = {"User-Agent": "AlpcaResearch isaiahdupree33@gmail.com"}
FLOW_TAGS = ("NetIncomeLoss", "NetCashProvidedByUsedInOperatingActivities")
STOCK_TAG = "Assets"


def _get(url: str, timeout: float = 30.0):
    return json.load(urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout))


def _days(a: str, b: str) -> int:
    ya, ma, da = map(int, a.split("-")); yb, mb, db = map(int, b.split("-"))
    return abs((date(yb, mb, db) - date(ya, ma, da)).days)


def _annual_flow(units):
    """Full-fiscal-year flow values keyed by period-end, earliest filing kept (original disclosure)."""
    out = {}
    for u in units:
        if u.get("fp") != "FY" or u.get("form") not in ("10-K", "10-K/A"):
            continue
        s, e, f = u.get("start"), u.get("end"), u.get("filed")
        if not (s and e and f) or not (350 <= _days(s, e) <= 380):    # ~full year only
            continue
        if e not in out or f < out[e][1]:
            out[e] = (float(u["val"]), f)
    return out


def _annual_stock(units):
    """Instant (balance-sheet) values keyed by period-end, earliest filing kept."""
    out = {}
    for u in units:
        if u.get("fp") != "FY" or u.get("form") not in ("10-K", "10-K/A"):
            continue
        e, f = u.get("end"), u.get("filed")
        if not (e and f):
            continue
        if e not in out or f < out[e][1]:
            out[e] = (float(u["val"]), f)
    return out


def extract_annual_fundamentals(facts: dict):
    g = facts.get("facts", {}).get("us-gaap", {})
    def units(tag):
        return g.get(tag, {}).get("units", {}).get("USD", [])
    ni = _annual_flow(units("NetIncomeLoss"))
    cfo = _annual_flow(units("NetCashProvidedByUsedInOperatingActivities"))
    assets = _annual_stock(units(STOCK_TAG))
    rows = []
    for e in sorted(set(ni) & set(cfo) & set(assets)):
        nival, nif = ni[e]; cfval, cff = cfo[e]; aval, af = assets[e]
        if aval <= 0:
            continue
        rows.append({"fy_end": e, "filed": max(nif, cff, af),    # known when the last of the three filed
                     "net_income": nival, "cfo": cfval, "total_assets": aval})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/Volumes/My Passport/AlpcaData/cache")
    ap.add_argument("--out", default="/Volumes/My Passport/AlpcaData/fundamentals_edgar")
    ap.add_argument("--delay", type=float, default=0.2)
    ap.add_argument("--max-symbols", type=int, default=195)
    args = ap.parse_args()
    cache, out = Path(args.cache), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    cikmap = {d["ticker"]: int(d["cik_str"])
              for d in _get("https://www.sec.gov/files/company_tickers.json").values()}
    syms = sorted(p.name.split("_1day_")[0] for p in cache.glob("*_1day_bars.jsonl"))[:args.max_symbols]
    have = {p.name.replace("_fund.json", "") for p in out.glob("*_fund.json")}
    todo = [s for s in syms if s not in have]
    print(f"[edgar] universe {len(syms)}, cached {len(have)}, fetching {len(todo)} (free, no quota)")
    ok = fail = 0
    for i, s in enumerate(todo, 1):
        cik = cikmap.get(s)
        if cik is None:
            fail += 1; continue
        try:
            facts = _get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json")
            rows = extract_annual_fundamentals(facts)
            if rows:
                (out / f"{s}_fund.json").write_text(json.dumps(rows)); ok += 1
            else:
                fail += 1
            if i % 25 == 0:
                print(f"  {i}/{len(todo)} (ok {ok}, empty/fail {fail})")
        except Exception as e:
            fail += 1
            print(f"  {s}: {type(e).__name__} {str(e)[:70]}")
        time.sleep(args.delay)
    total = len({p.name.replace('_fund.json', '') for p in out.glob('*_fund.json')})
    print(f"[edgar] done: +{ok} this run, {fail} empty/fail, {total}/{len(syms)} cached")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
