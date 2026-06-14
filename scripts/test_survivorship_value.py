"""
Case 43 — SURVIVORSHIP-BIAS POINT-IN-TIME re-test of mid-cap value.

The mid-cap value edge (Cases 38/40/42) was measured on a universe of names that EXIST TODAY. That
omits the value-traps — cheap, distressed mid-caps that went bankrupt or were delisted (BBBY, RAD,
ENDP, AVYA, BIG, CANO, …). Value's LONG leg buys the cheap names, so excluding the cheapest-that-died
inflates it. The honest fix: put the dead names back. Alpaca serves delisted-ticker history up to the
delisting date (verified), so we CAN.

This loads the survivor-only mid-cap universe vs the same universe PLUS ~50 delisted/bankrupt names,
and re-runs value + value+light-momentum on each. The dead names are ranked while still listed (cheap,
falling) so value buys them and rides them DOWN — that loss is captured in the bars up to delisting.

Caveat (conservative): after a name delists its bars stop, so the backtest books ~0 thereafter, not
the final gap-to-zero — so the TRUE survivorship hit is somewhat WORSE than measured here. If the edge
degrades materially even so, the bias is real; if it survives, the residual terminal-gap risk remains.

Run: .venv/bin/python scripts/test_survivorship_value.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from alpca.backtest.value import backtest_value_composite  # noqa: E402
from alpca.backtest.evaluation import sharpe_of  # noqa: E402

PPY = 252.0


def _load_multi(dirs):
    out = {}
    for c in dirs:
        for p in Path(c).glob("*_1day_bars.jsonl"):
            rows = [json.loads(l) for l in p.open() if l.strip()]
            if rows:
                out[p.name.split("_1day_")[0]] = rows
    return out


def _load_funds_multi(dirs):
    out = {}
    for d in dirs:
        d = Path(d)
        if d.exists():
            for p in d.glob("*_fund.json"):
                out[p.name.replace("_fund.json", "")] = json.loads(p.read_text())
    return out


def _eq(daily, s=1.0):
    e = [s]
    for x in daily:
        e.append(e[-1] * (1 + x))
    return e


def _run(bars, funds, mw):
    r = backtest_value_composite(bars, funds, top_frac=0.2, rebalance_days=21, cost_bps=2.0,
                                 momentum_weight=mw, periods_per_year=PPY)
    by = {}
    for ep, x in zip(r.dates, r.daily_returns):
        by.setdefault(time.gmtime(ep).tm_year, []).append(x)
    yr = {y: round(sharpe_of(_eq(v), PPY), 2) for y, v in by.items() if len(v) >= 30}
    pos = sum(1 for s in yr.values() if s > 0)
    return {"sharpe": round(r.sharpe, 3), "n_syms": len(r.symbols),
            "pos_years": f"{pos}/{len(yr)}", "per_year": yr}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--midcap", default="/Volumes/My Passport/AlpcaData/cache_midcap")
    ap.add_argument("--delisted", default="/Volumes/My Passport/AlpcaData/cache_midcap_delisted")
    ap.add_argument("--mid-funds", default="/Volumes/My Passport/AlpcaData/fundamentals_edgar_midcap")
    ap.add_argument("--del-funds", default="/Volumes/My Passport/AlpcaData/fundamentals_edgar_midcap_delisted")
    ap.add_argument("--out", default="data/survivorship_value.json")
    args = ap.parse_args()

    surv_b = _load_multi([args.midcap])
    surv_f = _load_funds_multi([args.mid_funds])
    aug_b = _load_multi([args.midcap, args.delisted])
    aug_f = _load_funds_multi([args.mid_funds, args.del_funds])
    add_b = len(aug_b) - len(surv_b)
    add_f = len(aug_f) - len(surv_f)
    print(f"[ok] survivor-only: {len(surv_b)} bars / {len(surv_f)} funds")
    print(f"[ok] + delisted:    {len(aug_b)} bars (+{add_b}) / {len(aug_f)} funds (+{add_f})\n")

    out = {}
    print(f"{'variant':>16}{'universe':>14}{'syms':>6}{'sharpe':>8}{'+yrs':>7}")
    print("-" * 51)
    for label, mw in (("value", 0.0), ("value+mom.25", 0.25)):
        s = _run(surv_b, {k: v for k, v in surv_f.items() if k in surv_b}, mw)
        a = _run(aug_b, {k: v for k, v in aug_f.items() if k in aug_b}, mw)
        out[label] = {"survivor_only": s, "with_delisted": a,
                      "sharpe_delta": round(a["sharpe"] - s["sharpe"], 3)}
        print(f"{label:>16}{'survivor-only':>14}{s['n_syms']:>6}{s['sharpe']:>8.2f}{s['pos_years']:>7}")
        print(f"{label:>16}{'+ delisted':>14}{a['n_syms']:>6}{a['sharpe']:>8.2f}{a['pos_years']:>7}"
              f"   (delta {a['sharpe']-s['sharpe']:+.2f})")

    worst = min(out[k]["sharpe_delta"] for k in out)
    print(f"\n[survivorship hit] worst Sharpe delta {worst:+.2f}")
    verdict = ("EDGE SURVIVES — value holds up with the dead names back in (degradation small)"
               if worst > -0.15 else
               "SURVIVORSHIP-INFLATED — value degrades materially once value-traps are included")
    print(f"[verdict] {verdict}")
    out["_verdict"] = {"worst_sharpe_delta": worst, "verdict": verdict}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"[saved] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
