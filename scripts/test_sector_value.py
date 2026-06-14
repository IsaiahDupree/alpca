"""
Case 36 — SECTOR-NEUTRAL VALUE vs raw value.

Hypothesis: a raw value composite secretly loads on cheap SECTORS (energy/financials cheap, tech
expensive) — a regime-timed sector bet, not pure value. Neutralizing the composite WITHIN sector
(demean by sector peers, then long/short on the residual) isolates the within-sector value premium,
which the literature says is the more persistent, more robust slice. Value already GENERALIZES
(positive fresh-symbol holdout, the only fundamental that did) but is too thin (~0.14) to deploy; this
is the highest-EV attempt to lift a known-good leg into the deployable + uncorrelated-second-leg range.

The bar is the same as everything else: main universe + DISJOINT fresh-symbol holdout + per-year
regime stability + cost + DSR + the falsification gate. A win must (a) beat raw value AND (b) keep a
POSITIVE fresh-symbol holdout — neutralizing must not destroy the generalization that made value
interesting in the first place.

Run: .venv/bin/python scripts/test_sector_value.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from alpca.backtest.value import backtest_value_composite  # noqa: E402
from alpca.backtest.evaluation import deflated_sharpe_ratio, sharpe_of  # noqa: E402
from alpca.ai.strategy_gate import falsification_gate  # noqa: E402

PPY = 252.0


def sic_to_sector(sic: str) -> str:
    """Map a 4-digit SIC to one of ~11 coarse sectors (coarse enough that buckets aren't singletons,
    fine enough that 'sector' means something). Manufacturing (2000-3999) is split by sub-industry."""
    try:
        n = int(str(sic)[:4])
    except (ValueError, TypeError):
        return "other"
    d2 = n // 100
    if d2 in (28,) or n in range(2833, 2837) or d2 in (80,) or n in range(3840, 3852):
        return "health"            # pharma + biotech + health services + medical devices
    if n in range(7370, 7380) or n in range(3570, 3580) or n in range(3660, 3680) or d2 == 36:
        return "tech"              # software + computers + semis + electronics
    if d2 in (13, 29) or n == 2911:
        return "energy"
    if d2 in (60, 61, 62, 63, 64):
        return "financials"        # banks, securities, insurance
    if d2 in (65, 66, 67):
        return "realestate"
    if d2 in (48, 49):
        return "utilities_telecom"
    if d2 in (40, 41, 42, 43, 44, 45, 46, 47):
        return "transport"
    if d2 in (52, 53, 54, 55, 56, 57, 58, 59):
        return "retail"
    if d2 in (20, 21):
        return "staples"           # food, beverage, tobacco
    if d2 in (10, 12, 14, 26, 28, 32, 33):
        return "materials"
    if d2 in range(15, 40):
        return "industrials"       # remaining manufacturing + construction
    return "other"


def _load(c):
    return {p.name.split("_1day_")[0]: [json.loads(l) for l in p.open() if l.strip()]
            for p in Path(c).glob("*_1day_bars.jsonl")}


def _load_funds(d):
    d = Path(d)
    return {p.name.replace("_fund.json", ""): json.loads(p.read_text())
            for p in d.glob("*_fund.json")} if d.exists() else {}


def _eq(daily, s=1.0):
    e = [s]
    for x in daily:
        e.append(e[-1] * (1 + x))
    return e


def _run(bars, funds, sector_map, *, label):
    r = backtest_value_composite(bars, funds, top_frac=0.2, rebalance_days=21, cost_bps=2.0,
                                 sector_by_sym=sector_map, periods_per_year=PPY)
    eq = r.equity_curve
    if len(eq) < 60:
        return None
    sp = int(len(eq) * 0.7)
    oos = sharpe_of(eq[sp:], PPY)
    by = {}
    for ep, x in zip(r.dates, r.daily_returns):
        by.setdefault(time.gmtime(ep).tm_year, []).append(x)
    yr = {y: round(sharpe_of(_eq(v), PPY), 2) for y, v in by.items() if len(v) >= 30}
    return {"label": label, "sharpe": r.sharpe, "oos": oos, "per_year": yr,
            "turnover": r.avg_turnover, "n_days": r.n_days, "avg_active": r.avg_active,
            "equity": eq, "daily": r.daily_returns, "dates": r.dates}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/Volumes/My Passport/AlpcaData/cache")
    ap.add_argument("--cache-fresh", default="/Volumes/My Passport/AlpcaData/cache_fresh")
    ap.add_argument("--fundamentals", default="/Volumes/My Passport/AlpcaData/fundamentals_edgar")
    ap.add_argument("--fundamentals-fresh", default="/Volumes/My Passport/AlpcaData/fundamentals_edgar_fresh")
    ap.add_argument("--sic", default="/Volumes/My Passport/AlpcaData/sic_codes.json")
    ap.add_argument("--n-trials", type=int, default=47)
    ap.add_argument("--out", default="data/sector_value_results.json")
    args = ap.parse_args()

    main_b, fresh_b = _load(args.cache), _load(args.cache_fresh)
    fm, ff = _load_funds(args.fundamentals), _load_funds(args.fundamentals_fresh)
    sic = json.loads(Path(args.sic).read_text())
    secmap = {s: sic_to_sector(v.get("sic", "")) for s, v in sic.items()}
    from collections import Counter
    bucket_n = Counter(secmap[s] for s in main_b if s in secmap)
    print(f"[ok] main {len(main_b)} (+{len(fresh_b)} fresh) · funds {len(fm)} (+{len(ff)})")
    print(f"[sectors] {dict(bucket_n.most_common())}\n")

    out = {}
    print(f"{'variant':>22}{'universe':>10}{'sharpe':>8}{'OOS':>7}{'+yrs':>7}{'turn':>7}")
    print("-" * 61)
    for vlabel, smap in (("raw value", None), ("sector-neutral value", secmap)):
        for ulabel, bars, funds in (("main", main_b, fm), ("fresh-holdout", fresh_b, ff)):
            res = _run(bars, funds, smap, label=f"{vlabel}|{ulabel}")
            if res is None:
                print(f"{vlabel:>22}{ulabel:>10}   (insufficient)"); continue
            pos = sum(1 for s in res["per_year"].values() if s > 0)
            out[res["label"]] = {k: v for k, v in res.items() if k not in ("equity", "daily", "dates")}
            print(f"{vlabel:>22}{ulabel:>10}{res['sharpe']:>8.2f}{res['oos']:>7.2f}"
                  f"{pos:>4}/{len(res['per_year']):<2}{res['turnover']:>7.3f}")

    # Gate the sector-neutral variant on the honest fields (main sharpe/oos/per-year + fresh holdout)
    mn = out.get("sector-neutral value|main"); fr = out.get("sector-neutral value|fresh-holdout")
    raw_mn = out.get("raw value|main")
    if mn and fr:
        eqm = _eq(_run(main_b, fm, secmap, label="g")["daily"])
        dsr = deflated_sharpe_ratio(eqm, n_trials=args.n_trials, sharpe_variance=1e-4)
        result = {"name": "sector_neutral_value", "sharpe": round(mn["sharpe"], 3),
                  "oos_sharpe": round(mn["oos"], 3), "fresh_holdout_sharpe": round(fr["sharpe"], 3),
                  "per_year": mn["per_year"], "dsr": round(dsr, 3),
                  "cost_2bps_sharpe": round(mn["sharpe"], 3), "turnover": round(mn["turnover"], 3)}
        g = falsification_gate(result)
        print(f"\n[gate] sector-neutral value: DSR {dsr:.2f} · rail {'PASS' if g.passed else 'FAIL'}")
        for r in g.reasons:
            print(f"        - {r}")
        beats_raw = mn["sharpe"] > (raw_mn["sharpe"] if raw_mn else -9)
        generalizes = fr["sharpe"] > 0
        verdict = ("KEEP (beats raw + generalizes + rail-pass)"
                   if (beats_raw and generalizes and g.passed)
                   else f"REJECT (beats_raw={beats_raw}, generalizes={generalizes}, rail={g.passed})")
        print(f"[verdict] {verdict}")
        out["_gate"] = {**result, "rail_pass": g.passed, "reasons": g.reasons,
                        "beats_raw": beats_raw, "generalizes": generalizes, "verdict": verdict}

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\n[saved] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
