"""
Case 38 — VALUE on a MID-CAP universe (where the premium is supposed to live).

The factor zoo was absent on our 195 liquid large-caps (Cases 26–37); the literature says value (and
most premia) are STRONGER in smaller, less-efficient names. So: a fresh ~127-name S&P-MidCap-400
universe (zero overlap with the large-caps we developed on — itself an out-of-universe test), with
EDGAR fundamentals. We run the raw value composite three ways:
  (1) FULL mid-cap universe          — is value stronger here than the 0.11 it scored on large-caps?
  (2) internal TRAIN half (frozen)   — develop nothing; just split
  (3) internal HOLDOUT half          — fresh-symbol generalization WITHIN mid-caps

Same honest bar: per-year regime, cost, DSR, gate. A real mid-cap value premium should (a) beat the
large-cap 0.11 on the full universe AND (b) keep a positive holdout-half Sharpe.

Run: .venv/bin/python scripts/test_midcap_value.py
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


def _run(bars, funds, *, momentum_weight=0.0):
    r = backtest_value_composite(bars, funds, top_frac=0.2, rebalance_days=21, cost_bps=2.0,
                                 momentum_weight=momentum_weight, periods_per_year=PPY)
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
            "n_days": r.n_days, "avg_active": r.avg_active, "daily": r.daily_returns, "n_syms": len(r.symbols)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/Volumes/My Passport/AlpcaData/cache_midcap")
    ap.add_argument("--fundamentals", default="/Volumes/My Passport/AlpcaData/fundamentals_edgar_midcap")
    ap.add_argument("--n-trials", type=int, default=54)
    ap.add_argument("--label", default="mid-cap")
    ap.add_argument("--momentum-weight", type=float, default=0.0, help="light AMP tilt (0.25 = sweet spot on mid-caps)")
    ap.add_argument("--baseline", type=float, default=0.11, help="Sharpe to beat (large-cap value=0.11)")
    ap.add_argument("--out", default="data/midcap_value_results.json")
    args = ap.parse_args()

    bars, funds = _load(args.cache), _load_funds(args.fundamentals)
    # symbols with BOTH bars and usable fundamentals
    usable = sorted(s for s in bars if s in funds and funds[s])
    print(f"[ok] mid-cap bars {len(bars)} · funds {len(funds)} · usable (both) {len(usable)}")
    if len(usable) < 20:
        print("[wait] too few usable symbols yet — fundamentals still downloading?"); return 1
    # deterministic train/holdout split (sorted, alternating — disjoint symbol sets)
    train = {s: bars[s] for i, s in enumerate(usable) if i % 3 != 0}
    hold = {s: bars[s] for i, s in enumerate(usable) if i % 3 == 0}
    ftrain = {s: funds[s] for s in train}
    fhold = {s: funds[s] for s in hold}
    print(f"[split] train {len(train)} · holdout {len(hold)} (disjoint)\n")

    print(f"{'slice':>16}{'syms':>6}{'sharpe':>8}{'OOS':>7}{'+yrs':>7}{'turn':>7}")
    print("-" * 51)
    out = {}
    for label, b, f in (("full midcap", {s: bars[s] for s in usable}, {s: funds[s] for s in usable}),
                        ("train-half", train, ftrain), ("holdout-half", hold, fhold)):
        r = _run(b, f, momentum_weight=args.momentum_weight)
        if r is None:
            print(f"{label:>16}  (insufficient)"); continue
        pos = sum(1 for s in r["per_year"].values() if s > 0)
        out[label] = {k: v for k, v in r.items() if k != "daily"}
        print(f"{label:>16}{r['n_syms']:>6}{r['sharpe']:>8.2f}{r['oos']:>7.2f}"
              f"{pos:>4}/{len(r['per_year']):<2}{r['turnover']:>7.3f}")

    full = out.get("full midcap"); hold_r = out.get("holdout-half")
    if full and hold_r:
        eqf = _eq(_run({s: bars[s] for s in usable}, {s: funds[s] for s in usable}, momentum_weight=args.momentum_weight)["daily"])
        dsr = deflated_sharpe_ratio(eqf, n_trials=args.n_trials, sharpe_variance=1e-4)
        result = {"name": "midcap_value", "sharpe": round(full["sharpe"], 3),
                  "oos_sharpe": round(full["oos"], 3), "fresh_holdout_sharpe": round(hold_r["sharpe"], 3),
                  "per_year": full["per_year"], "dsr": round(dsr, 3),
                  "cost_2bps_sharpe": round(full["sharpe"], 3), "turnover": round(full["turnover"], 3)}
        g = falsification_gate(result)
        beats_largecap = full["sharpe"] > args.baseline
        generalizes = hold_r["sharpe"] > 0
        print(f"\n[gate] mid-cap value: DSR {dsr:.2f} · rail {'PASS' if g.passed else 'FAIL'}")
        for r in g.reasons:
            print(f"        - {r}")
        verdict = ("KEEP (beats large-cap 0.11 + generalizes + rail-pass)"
                   if (beats_largecap and generalizes and g.passed)
                   else f"REJECT/WEAK (beats_largecap={beats_largecap}, generalizes={generalizes}, rail={g.passed})")
        print(f"[verdict] {verdict}  ({args.label} baseline to beat = {args.baseline})")
        out["_gate"] = {**result, "rail_pass": g.passed, "beats_largecap": beats_largecap,
                        "generalizes": generalizes, "verdict": verdict}

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\n[saved] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
