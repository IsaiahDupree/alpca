"""
Build a point-in-time insider-BUY signal from the SEC bulk Insider Transactions Data Sets (free, no
key, no AV quota). For each quarter 2016Q1..2026Q2: download <q>_form345.zip, join SUBMISSION (filing
date + issuer ticker), NONDERIV_TRANS (open-market purchases: TRANS_CODE='P' & acquired='A'), and
REPORTINGOWNER (insider role), and emit aggregated per-(ticker, FILING_DATE) insider buys.

FILING_DATE (when the Form 4 became public) is the tradeable signal date — NO lookahead (the
transaction itself is reported within ~2 business days, but we key off the public filing date).

Output JSONL rows: {ticker, filing_date, trans_date, buy_value, n_owners, roles}.

Run: .venv/bin/python scripts/build_insider_signal.py [--out PATH] [--start 2016Q1] [--end 2026Q2]
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
import time
import urllib.request
import zipfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path

BASE = "https://www.sec.gov/files/structureddata/data/insider-transactions-data-sets"
UA = "Alpca Research isaiahdupree33@gmail.com"
csv.field_size_limit(10_000_000)


def quarters(start: str, end: str):
    sy, sq = int(start[:4]), int(start[5])
    ey, eq = int(end[:4]), int(end[5])
    y, q = sy, sq
    while (y, q) <= (ey, eq):
        yield f"{y}q{q}"
        q += 1
        if q > 4:
            q = 1; y += 1


def norm_date(s: str) -> str:
    s = (s or "").strip()
    for fmt in ("%d-%b-%Y", "%Y-%m-%d", "%m/%d/%Y", "%d-%b-%y"):
        try:
            return datetime.strptime(s, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return ""


def fetch_zip(q: str) -> zipfile.ZipFile | None:
    url = f"{BASE}/{q}_form345.zip"
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            data = r.read()
        return zipfile.ZipFile(io.BytesIO(data))
    except Exception as e:
        print(f"  [{q}] download ERR {e}")
        return None


def _reader(zf, name):
    with zf.open(name) as f:
        yield from csv.DictReader(io.TextIOWrapper(f, "latin-1"), delimiter="\t")


def parse_quarter(zf) -> list[dict]:
    # accession -> (filing_date, ticker)
    sub = {}
    for r in _reader(zf, "SUBMISSION.tsv"):
        tkr = (r.get("ISSUERTRADINGSYMBOL") or "").strip().upper()
        if tkr and tkr not in ("NONE", "N/A", "NA"):
            sub[r["ACCESSION_NUMBER"]] = (norm_date(r.get("FILING_DATE", "")), tkr)
    # accession -> roles
    roles = defaultdict(set)
    for r in _reader(zf, "REPORTINGOWNER.tsv"):
        rel = (r.get("RPTOWNER_RELATIONSHIP") or "")
        roles[r["ACCESSION_NUMBER"]].add(rel)
    # aggregate open-market purchases per (ticker, filing_date)
    agg = defaultdict(lambda: {"buy_value": 0.0, "owners": set(), "roles": set(), "trans_date": ""})
    for r in _reader(zf, "NONDERIV_TRANS.tsv"):
        if (r.get("TRANS_CODE") or "").strip() != "P":          # P = open-market purchase
            continue
        if (r.get("TRANS_ACQUIRED_DISP_CD") or "").strip() != "A":  # A = acquired
            continue
        acc = r["ACCESSION_NUMBER"]
        if acc not in sub:
            continue
        fdate, tkr = sub[acc]
        if not fdate:
            continue
        try:
            sh = float(r.get("TRANS_SHARES") or 0)
            px = float(r.get("TRANS_PRICEPERSHARE") or 0)
        except ValueError:
            continue
        if sh <= 0 or px <= 0:                                   # skip $0 grants/options noise
            continue
        key = (tkr, fdate)
        a = agg[key]
        a["buy_value"] += sh * px
        a["owners"].add(acc)
        a["roles"] |= roles.get(acc, set())
        a["trans_date"] = norm_date(r.get("TRANS_DATE", "")) or a["trans_date"]
    out = []
    for (tkr, fdate), a in agg.items():
        out.append({"ticker": tkr, "filing_date": fdate, "trans_date": a["trans_date"],
                    "buy_value": round(a["buy_value"], 2), "n_owners": len(a["owners"]),
                    "roles": sorted(x for x in a["roles"] if x)[:4]})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/Volumes/My Passport/AlpcaData/insider/insider_buys.jsonl")
    ap.add_argument("--start", default="2016Q1")
    ap.add_argument("--end", default="2026Q2")
    args = ap.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    with out.open("w") as f:
        for q in quarters(args.start, args.end):
            zf = fetch_zip(q)
            if zf is None:
                continue
            rows = parse_quarter(zf)
            for r in rows:
                f.write(json.dumps(r) + "\n")
            total += len(rows)
            print(f"  [{q}] {len(rows)} ticker-day insider-buy aggregates (total {total})")
            time.sleep(0.5)   # SEC politeness
    print(f"\n[done] {total} insider-buy rows -> {out}")


if __name__ == "__main__":
    main()
