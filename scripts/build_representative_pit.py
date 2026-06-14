"""
Build a REPRESENTATIVE point-in-time delisted set (Case 45 input).

Case 44's +50 delistings were hand-picked FAMOUS failures, which over-feeds a short leg. This instead
takes ALL 1,707 US-listed delistings from Alpaca's inactive-assets API and keeps the ones that were
genuinely MID-CAP-CALIBER at the START of the window (2021) — selecting on 2021 price + liquidity,
NEVER on how the name ended, so bankruptcies AND acquisitions (which HURT a short) both qualify with
no outcome bias. The survivors of that filter, combined with the survivor mid-cap universe, form a
representative point-in-time universe to re-run momentum on.

Filter (all on the FIRST ~120 bars / 2021, outcome-blind):
  - first bar on/before 2021-10-01 and ≥150 total bars (existed near window start, real history)
  - last bar before 2026-03-01 (actually DELISTED inside the window, not a data gap on a live name)
  - median price over first 120 bars in [$5, $500] (mid-cap range — excludes pennies & mega-caps)
  - average dollar volume over first 120 bars > $3M/day (liquid enough to be a real mid-cap)

Run: .venv/bin/python scripts/build_representative_pit.py
"""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
import time
from pathlib import Path


def _epoch(s):
    return time.mktime(time.strptime(s, "%Y-%m-%d"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/Volumes/My Passport/AlpcaData/cache_delisted_sip")
    ap.add_argument("--out", default="/Volumes/My Passport/AlpcaData/cache_midcap_pit_delisted")
    ap.add_argument("--min-price", type=float, default=5.0)
    ap.add_argument("--max-price", type=float, default=500.0)
    ap.add_argument("--min-dollar-vol", type=float, default=3e6)
    args = ap.parse_args()
    src, out = Path(args.src), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*_1day_bars.jsonl"):
        old.unlink()

    start_cap = _epoch("2021-10-01")      # must have started by here
    end_floor = _epoch("2026-03-01")      # must have ended (delisted) before here
    kept, examined = [], 0
    fail_like = acq_like = 0
    for p in sorted(src.glob("*_1day_bars.jsonl")):
        rows = [json.loads(l) for l in p.open() if l.strip()]
        if len(rows) < 150:
            continue
        examined += 1
        rows.sort(key=lambda b: int(b["timestamp"]))
        ts = [int(b["timestamp"]) for b in rows]
        cl = [float(b["close"]) for b in rows if b.get("close")]
        vol = [float(b.get("volume", 0) or 0) for b in rows]
        if not cl:
            continue
        first120 = cl[:120]
        med_px = statistics.median(first120)
        dvol = statistics.median([cl[i] * vol[i] for i in range(min(120, len(rows)))])
        if not (ts[0] <= start_cap and ts[-1] < end_floor):
            continue
        if not (args.min_price <= med_px <= args.max_price and dvol >= args.min_dollar_vol):
            continue
        sym = p.name.split("_1day_")[0]
        # outcome proxy (NOT used for selection): last-30-bar return — failures crater, acquisitions flat/up
        tail = cl[-1] / cl[-30] - 1.0 if len(cl) > 30 else 0.0
        if tail < -0.30:
            fail_like += 1
        else:
            acq_like += 1
        kept.append((sym, len(rows), round(med_px, 1), round(tail, 2)))
        shutil.copy(p, out / p.name)

    print(f"[representative PIT] examined {examined} delisted-with-history · KEPT {len(kept)} mid-cap-caliber")
    print(f"  outcome mix (proxy, not a filter): ~{fail_like} failure-like (cratered) · ~{acq_like} flat/up "
          f"(acquisition/orderly) — a real universe has BOTH")
    print(f"  sample: {sorted(kept, key=lambda x: -x[1])[:12]}")
    Path("data").mkdir(exist_ok=True)
    Path("data/representative_pit_delisted.json").write_text(json.dumps(
        {"kept": [k[0] for k in kept], "n_kept": len(kept), "fail_like": fail_like,
         "acq_like": acq_like, "detail": kept}, indent=2))
    print(f"  -> copied to {out}")
    print(f"  -> data/representative_pit_delisted.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
