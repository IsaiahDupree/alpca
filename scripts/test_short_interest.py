"""
Case study: short-interest (borrow-fee) tilt on REAL Nasdaq short interest. Long low days-to-cover
/ short high (the short-interest anomaly). The honest crux is the BORROW STRESS: the high-DTC names
the anomaly says to short are the expensive-to-borrow ones, so the fee may eat the short leg (the
same wall that sank surprise-PEAD). Judged with anomaly-vs-control, a borrow sweep (flat + DTC-
scaled), OOS, and DSR.

POWER CAVEAT: Nasdaq free SI is ~1yr (24 bi-monthly points) — single regime, weak significance.
A positive here is a LEAD, not a validation; deepen via FINRA history before trusting it.

Run: .venv/bin/python scripts/test_short_interest.py
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from alpca.backtest.evaluation import (  # noqa: E402
    deflated_sharpe_ratio, probabilistic_sharpe_ratio, sharpe_of, sharpe_pvalue, sharpe_tstat)
from alpca.backtest.short_interest import backtest_short_interest_tilt  # noqa: E402

PPY = 252.0


def _eq_from(daily, start=100_000.0):
    eq = [start]
    for x in daily:
        eq.append(eq[-1] * (1 + x))
    return eq


def _active_equity(r):
    """Trim the leading flat (pre-SI-data) period: SI covers only the last ~1yr of the 5yr daily
    window, so the strategy is flat until the data starts. Judging the whole 5yr would fake an
    IS/OOS split (all positions land in the 'OOS' half). Return equity over the ACTIVE span only."""
    dr = r.daily_returns
    first = next((i for i, x in enumerate(dr) if abs(x) > 1e-12), None)
    if first is None:
        return [r.equity_curve[0]], 0
    eq = [100_000.0]
    for x in dr[first:]:
        eq.append(eq[-1] * (1 + x))
    return eq, len(dr) - first


def oos(eq, frac=0.3):
    sp = int(len(eq) * (1 - frac))
    return sharpe_of(eq[:sp], PPY), sharpe_of(eq[sp:], PPY)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/Volumes/My Passport/AlpcaData/cache")
    ap.add_argument("--si", default="/Volumes/My Passport/AlpcaData/short_interest")
    ap.add_argument("--n-trials", type=int, default=40)
    ap.add_argument("--out", default="data/short_interest_results.json")
    args = ap.parse_args()
    cache, sidir = Path(args.cache), Path(args.si)

    bars_by, si_by = {}, {}
    for sf in sidir.glob("*_si.json"):
        sym = sf.name.replace("_si.json", "")
        bf = cache / f"{sym}_1day_bars.jsonl"
        rows = json.loads(sf.read_text())
        if rows and bf.exists():
            bars_by[sym] = [json.loads(l) for l in bf.open() if l.strip()]
            si_by[sym] = rows
    n_obs = sum(len(v) for v in si_by.values())
    print(f"[ok] {len(si_by)} symbols, {n_obs} SI observations (~{n_obs/max(len(si_by),1):.0f}/symbol)\n")
    if len(si_by) < 10:
        print("[fail] too few symbols with SI"); return 1

    # ---- anomaly vs control (no borrow), then borrow stress ----
    print(f"{'variant':>22}{'sharpe':>8}{'IS':>7}{'OOS':>7}{'ret':>8}{'maxDD':>8}{'turn/d':>8}{'sig':>5}")
    print("-" * 73)
    grid = []
    scen = [
        ("anomaly, no borrow",     True,  None),
        ("control (chase), no brw", False, None),
        ("anomaly, flat 3% borrow", True,  0.03),
        ("anomaly, DTC-scaled brw", True,  {"base": 0.01, "per_dtc": 0.03, "cap": 0.50}),
    ]
    for name, rev, brw in scen:
        r = backtest_short_interest_tilt(bars_by, si_by, top_frac=0.2, reverse=rev, borrow=brw,
                                         cost_bps=2.0, periods_per_year=PPY)
        aeq, active_days = _active_equity(r)              # judge the ACTIVE ~1yr, not the flat 5yr
        if active_days < 60:
            continue
        sh = sharpe_of(aeq, PPY)
        is_sh, oos_sh = oos(aeq)
        t, pv = sharpe_tstat(aeq), sharpe_pvalue(aeq)
        sig = pv < 0.05 and abs(t) > 2.0
        grid.append({"variant": name, "reverse": rev, "sharpe": sh, "is": is_sh, "oos": oos_sh,
                     "ret": (aeq[-1] - aeq[0]) / aeq[0], "maxdd": r.max_drawdown,
                     "turnover": r.avg_turnover, "rebals": r.n_rebalances, "active_days": active_days, "sig": sig})
        print(f"{name:>22}{sh:>8.2f}{is_sh:>7.2f}{oos_sh:>7.2f}{(aeq[-1]/aeq[0]-1)*100:>7.0f}%"
              f"{r.max_drawdown*100:>7.1f}%{r.avg_turnover:>8.3f}{('Y' if sig else ''):>5}")

    anomaly = next((x for x in grid if x["variant"] == "anomaly, no borrow"), None)
    control = next((x for x in grid if x["reverse"] is False), None)
    dtc_brw = next((x for x in grid if "DTC-scaled" in x["variant"]), None)

    # DSR on the realistic (DTC-scaled borrow) sleeve. Deflate by a CLEAN same-direction top_frac
    # sweep (NOT the anomaly-vs-control pair — the control is the negation, not a trial).
    brw_cfg = {"base": 0.01, "per_dtc": 0.03, "cap": 0.50}
    sweep_sh = []
    for tf in (0.1, 0.15, 0.2, 0.3):
        rr = backtest_short_interest_tilt(bars_by, si_by, top_frac=tf, reverse=True, borrow=brw_cfg,
                                          cost_bps=2.0, periods_per_year=PPY)
        aeqq, adq = _active_equity(rr)
        if adq >= 60:
            sweep_sh.append(sharpe_of(aeqq, PPY) / (PPY ** 0.5))
    var_trials = (statistics.pvariance(sweep_sh) if len(sweep_sh) > 1 else 1e-5) or 1e-5
    dsr = psr = active_days = None
    if dtc_brw is not None:
        r = backtest_short_interest_tilt(bars_by, si_by, top_frac=0.2, reverse=True,
                                         borrow=brw_cfg, cost_bps=2.0, periods_per_year=PPY)
        aeq, active_days = _active_equity(r)
        psr = probabilistic_sharpe_ratio(aeq)
        dsr = deflated_sharpe_ratio(aeq, n_trials=args.n_trials, sharpe_variance=var_trials)

    # per-calendar-year regime breakdown on the realistic (DTC-borrow) sleeve — the cross-regime
    # test the 1yr Nasdaq feed couldn't take; FINRA's multi-year depth makes it meaningful.
    by_year = {}
    if dtc_brw is not None:
        rr = backtest_short_interest_tilt(bars_by, si_by, top_frac=0.2, reverse=True, borrow=brw_cfg,
                                          cost_bps=2.0, periods_per_year=PPY)
        first = next((i for i, x in enumerate(rr.daily_returns) if abs(x) > 1e-12), 0)
        for ep, x in zip(rr.dates[first:], rr.daily_returns[first:]):
            by_year.setdefault(time.gmtime(ep).tm_year, []).append(x)
    yr_sh = {y: sharpe_of(_eq_from(v), PPY) for y, v in by_year.items() if len(v) >= 30}
    n_years = len(yr_sh)
    pos_years = sum(1 for s in yr_sh.values() if s > 0)

    print("\n" + "=" * 73)
    if anomaly and control:
        multi = n_years >= 3
        if multi:
            print(f"VERDICT (MULTI-REGIME — FINRA gives {n_years} active calendar years):")
        else:
            ad = anomaly.get("active_days", 0)
            print(f"VERDICT (POWER-LIMITED: only {ad} active days / {n_years} yr — not multi-regime):")
        print(f"  anomaly (short high-DTC) Sharpe {anomaly['sharpe']:.2f}; control {control['sharpe']:.2f} "
              f"(sign-confirms the anomaly).")
        if dtc_brw is not None:
            print(f"  under DTC-scaled borrow (the crux): Sharpe {dtc_brw['sharpe']:.2f}, PSR {psr:.2f}, "
                  f"DSR {dsr:.2f}, turnover {dtc_brw['turnover']:.3f}/day (bi-monthly = low).")
            if yr_sh:
                print("  per-calendar-year Sharpe (DTC-borrow sleeve): "
                      + ", ".join(f"{y}:{yr_sh[y]:+.2f}" for y in sorted(yr_sh)))
                print(f"  -> positive in {pos_years}/{n_years} years")
            if multi:
                verdict = (f"survives ACROSS {n_years} regimes (DSR {dsr:.2f}, {pos_years}/{n_years} years +) "
                           "-> a VALIDATED 3rd-leg candidate" if dsr > 0.9 and pos_years >= n_years - 1 and dtc_brw["sharpe"] > 0.3
                           else f"positive but NOT robust across regimes ({pos_years}/{n_years} yrs) -> still a lead")
            else:
                verdict = ("right sign + survives borrow but <3yr -> a LEAD; needs more regimes"
                           if dtc_brw["sharpe"] > 0.3 else "borrow eats the short leg -> not a net edge")
            print(f"  -> {verdict}")

    Path(args.out).write_text(json.dumps({"n_symbols": len(si_by), "n_obs": n_obs, "grid": grid,
                                          "dtc_borrow_dsr": dsr, "dtc_borrow_psr": psr,
                                          "per_year": yr_sh}, indent=2))
    print(f"\n[done] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
