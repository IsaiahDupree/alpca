"""
Case 37 — VALUE + MOMENTUM combined (Asness-Moskowitz-Pedersen "Value and Momentum Everywhere").

Value and (cross-sectional) momentum are the two most-documented market-neutral premia, and they are
NEGATIVELY correlated — cheap stocks have usually been falling, winners have usually gotten expensive.
So a 50/50 combined rank (long cheap-AND-rising, short expensive-AND-falling) is historically MORE
regime-stable and higher-Sharpe than either leg alone. This is the single strongest zero-new-data
candidate for the elusive SECOND uncorrelated leg: value already generalizes (positive fresh holdout)
but is too thin; momentum alone on a dollar-neutral book is a legit cross-sectional signal (NOT beta,
because it's long-short). The combo could lift the blend into deployable range AND diversify the pairs
basket.

Sweep momentum_weight 0.0 (pure value) -> 1.0 (pure xsec momentum). Same bar for every blend: main +
DISJOINT fresh-symbol holdout + per-year regime + cost + DSR + gate. A win must beat BOTH legs on the
honest fields AND keep a POSITIVE fresh holdout.

Run: .venv/bin/python scripts/test_value_momentum.py
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
WEIGHTS = [0.0, 0.25, 0.5, 0.75, 1.0]


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


def _run(bars, funds, w):
    r = backtest_value_composite(bars, funds, top_frac=0.2, rebalance_days=21, cost_bps=2.0,
                                 momentum_weight=w, momentum_lookback=252, momentum_skip=21,
                                 periods_per_year=PPY)
    eq = r.equity_curve
    if len(eq) < 60:
        return None
    sp = int(len(eq) * 0.7)
    oos = sharpe_of(eq[sp:], PPY)
    by = {}
    for ep, x in zip(r.dates, r.daily_returns):
        by.setdefault(time.gmtime(ep).tm_year, []).append(x)
    yr = {y: round(sharpe_of(_eq(v), PPY), 2) for y, v in by.items() if len(v) >= 30}
    return {"sharpe": r.sharpe, "oos": oos, "per_year": yr, "turnover": r.avg_turnover,
            "n_days": r.n_days, "daily": r.daily_returns}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/Volumes/My Passport/AlpcaData/cache")
    ap.add_argument("--cache-fresh", default="/Volumes/My Passport/AlpcaData/cache_fresh")
    ap.add_argument("--fundamentals", default="/Volumes/My Passport/AlpcaData/fundamentals_edgar")
    ap.add_argument("--fundamentals-fresh", default="/Volumes/My Passport/AlpcaData/fundamentals_edgar_fresh")
    ap.add_argument("--n-trials", type=int, default=52)
    ap.add_argument("--out", default="data/value_momentum_results.json")
    args = ap.parse_args()

    main_b, fresh_b = _load(args.cache), _load(args.cache_fresh)
    fm, ff = _load_funds(args.fundamentals), _load_funds(args.fundamentals_fresh)
    print(f"[ok] main {len(main_b)} (+{len(fresh_b)} fresh) · funds {len(fm)} (+{len(ff)})\n")
    print(f"{'mom_w':>7}{'main':>8}{'OOS':>7}{'+yrs':>7}{'fresh':>8}{'f+yrs':>7}{'turn':>7}{'DSR':>7}")
    print("-" * 58)

    out = {}
    best = None
    for w in WEIGHTS:
        m = _run(main_b, fm, w)
        f = _run(fresh_b, ff, w)
        if not m or not f:
            print(f"{w:>7.2f}  (insufficient)"); continue
        dsr = deflated_sharpe_ratio(_eq(m["daily"]), n_trials=args.n_trials, sharpe_variance=1e-4)
        mp = sum(1 for s in m["per_year"].values() if s > 0)
        fp = sum(1 for s in f["per_year"].values() if s > 0)
        rec = {"momentum_weight": w, "main_sharpe": round(m["sharpe"], 3), "oos": round(m["oos"], 3),
               "main_pos_years": f"{mp}/{len(m['per_year'])}", "fresh_sharpe": round(f["sharpe"], 3),
               "fresh_pos_years": f"{fp}/{len(f['per_year'])}", "turnover": round(m["turnover"], 3),
               "dsr": round(dsr, 3), "per_year": m["per_year"], "fresh_per_year": f["per_year"]}
        out[f"w={w}"] = rec
        print(f"{w:>7.2f}{m['sharpe']:>8.2f}{m['oos']:>7.2f}{mp:>4}/{len(m['per_year']):<2}"
              f"{f['sharpe']:>8.2f}{fp:>4}/{len(f['per_year']):<2}{m['turnover']:>7.3f}{dsr:>7.2f}")
        # candidate "best" = generalizes (fresh>0) and highest main sharpe among those
        if f["sharpe"] > 0 and (best is None or m["sharpe"] > best[1]):
            best = (w, m["sharpe"], rec, m, f)

    if best is not None:
        w, _, rec, m, f = best
        result = {"name": f"value_momentum_w{w}", "sharpe": rec["main_sharpe"],
                  "oos_sharpe": rec["oos"], "fresh_holdout_sharpe": rec["fresh_sharpe"],
                  "per_year": m["per_year"], "dsr": rec["dsr"],
                  "cost_2bps_sharpe": rec["main_sharpe"], "turnover": rec["turnover"]}
        g = falsification_gate(result)
        pure_v = out.get("w=0.0", {}).get("main_sharpe", -9)
        pure_m = out.get("w=1.0", {}).get("main_sharpe", -9)
        beats_legs = rec["main_sharpe"] > pure_v and rec["main_sharpe"] > pure_m
        print(f"\n[best generalizing blend] momentum_weight={w}  main {rec['main_sharpe']:.2f} "
              f"(pure value {pure_v:.2f}, pure mom {pure_m:.2f}) · fresh {rec['fresh_sharpe']:.2f}")
        print(f"[gate] DSR {rec['dsr']:.2f} · rail {'PASS' if g.passed else 'FAIL'}")
        for r in g.reasons:
            print(f"        - {r}")
        verdict = ("KEEP (blend beats both legs + generalizes + rail-pass)"
                   if (beats_legs and f["sharpe"] > 0 and g.passed)
                   else f"REJECT (beats_legs={beats_legs}, generalizes={f['sharpe']>0}, rail={g.passed})")
        print(f"[verdict] {verdict}")
        out["_gate"] = {**result, "rail_pass": g.passed, "beats_legs": beats_legs,
                        "reasons": g.reasons, "verdict": verdict}
    else:
        print("\n[verdict] REJECT — no blend produced a positive fresh-symbol holdout")
        out["_gate"] = {"verdict": "REJECT — no generalizing blend"}

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\n[saved] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
