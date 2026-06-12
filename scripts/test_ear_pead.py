"""
Case study: EAR-PEAD (earnings-announcement-return drift). Judges three modes honestly —
  long      : long high-EAR names only  -> MUST beat buy-and-hold or it's just beta
  neutral   : long high-EAR / short low-EAR (single-name short, the borrow-fragile one)
  beta_hedged: long high-EAR, short the index by beta (cheap GC short — the PEAD short-leg fix)
plus OOS split, cost stress, DSR, and a PROFIT-PER-DAY translation at half-Kelly sizing (the
honest answer to "highest profit/day": edge x risk budget, not trade frequency).

Run: .venv/bin/python scripts/test_ear_pead.py
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from alpca.backtest.evaluation import (  # noqa: E402
    deflated_sharpe_ratio, probabilistic_sharpe_ratio, sharpe_of, sharpe_pvalue, sharpe_tstat,
    buy_and_hold)
from alpca.backtest.ear_pead import backtest_ear_pead  # noqa: E402

PPY = 252.0


def oos(eq, frac=0.3):
    sp = int(len(eq) * (1 - frac))
    return sharpe_of(eq[:sp], PPY), sharpe_of(eq[sp:], PPY)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/Volumes/My Passport/AlpcaData/cache")
    ap.add_argument("--earnings", default="/Volumes/My Passport/AlpcaData/earnings_av")
    ap.add_argument("--bench", default="SPY")
    ap.add_argument("--hold", type=int, default=40)
    ap.add_argument("--entry-thr", type=float, default=2.0, help="min |EAR| in percent (2.0 = +2% reaction)")
    ap.add_argument("--cost-bps", type=float, default=2.0)
    ap.add_argument("--n-trials", type=int, default=37)
    ap.add_argument("--out", default="data/ear_pead_results.json")
    args = ap.parse_args()
    cache, edir = Path(args.cache), Path(args.earnings)

    bars_by, events_by = {}, {}
    for ef in edir.glob("*_earnings.json"):
        sym = ef.name.replace("_earnings.json", "")
        ev = json.loads(ef.read_text())
        bf = cache / f"{sym}_1day_bars.jsonl"
        if ev and bf.exists():
            bars_by[sym] = [json.loads(l) for l in bf.open() if l.strip()]
            events_by[sym] = ev
    bench_bars = None
    bbf = cache / f"{args.bench}_1day_bars.jsonl"
    if bbf.exists():
        bench_bars = [json.loads(l) for l in bbf.open() if l.strip()]
    print(f"[ok] {len(events_by)} symbols, bench={args.bench} {'loaded' if bench_bars else 'MISSING'}\n")
    if len(events_by) < 5:
        print("[fail] too few symbols"); return 1

    # ---- 3 modes at the chosen threshold ----
    print(f"{'mode':>12}{'sharpe':>8}{'IS':>7}{'OOS':>7}{'ret':>8}{'maxDD':>8}{'beta':>7}"
          f"{'events':>8}{'sig':>5}")
    print("-" * 72)
    rows = {}
    for mode in ("long", "neutral", "beta_hedged"):
        r = backtest_ear_pead(bars_by, events_by, hold=args.hold, entry_thr=args.entry_thr,
                              mode=mode, bench_bars=bench_bars, cost_bps=args.cost_bps, periods_per_year=PPY)
        is_sh, oos_sh = oos(r.equity_curve)
        t, pv = sharpe_tstat(r.equity_curve), sharpe_pvalue(r.equity_curve)
        sig = pv < 0.05 and abs(t) > 2.0
        rows[mode] = {"mode": mode, "sharpe": r.sharpe, "is": is_sh, "oos": oos_sh,
                      "ret": r.total_return, "maxdd": r.max_drawdown, "beta": r.beta,
                      "events": r.n_events_used, "sig": sig, "equity": r.equity_curve}
        print(f"{mode:>12}{r.sharpe:>8.2f}{is_sh:>7.2f}{oos_sh:>7.2f}{r.total_return*100:>7.0f}%"
              f"{r.max_drawdown*100:>7.1f}%{r.beta:>7.2f}{r.n_events_used:>8}{('Y' if sig else ''):>5}")

    # ---- long-only MUST beat buy-and-hold (else it's beta) ----
    print("\n" + "=" * 72)
    # B&H of an equal-weight basket of the traded names ~ approximate with the bench
    if bench_bars:
        bh = buy_and_hold(bench_bars, PPY)
        print(f"BENCHMARK {args.bench} buy-and-hold over the window: Sharpe {bh.sharpe:.2f}, "
              f"return {bh.total_return*100:.0f}%, maxDD {bh.maxdd*100:.1f}%")
        lo = rows["long"]
        print(f"  long-only EAR-PEAD: Sharpe {lo['sharpe']:.2f} vs B&H {bh.sharpe:.2f}, "
              f"return {lo['ret']*100:.0f}% vs {bh.total_return*100:.0f}% -> "
              f"{'BEATS B&H (alpha candidate)' if lo['sharpe'] > bh.sharpe and lo['ret'] > bh.total_return else 'does NOT beat B&H on both -> long leg is BETA'}")

    # ---- DSR on the beta-hedged leg, deflated by a clean threshold sweep of the SAME sleeve ----
    bh_mode = rows["beta_hedged"]
    print(f"\nBETA-HEDGED sleeve threshold sweep (long high-EAR, short {args.bench} by beta — a CHEAP "
          f"GC index short, NOT the borrow-fragile single-name short):")
    print(f"{'entry_thr':>9}{'sharpe':>8}{'IS':>7}{'OOS':>7}{'ret':>8}{'maxDD':>8}{'beta':>7}")
    print("-" * 54)
    sweep = []
    for thr in (1.0, 1.5, 2.0, 3.0, 4.0):
        r = backtest_ear_pead(bars_by, events_by, hold=args.hold, entry_thr=thr, mode="beta_hedged",
                              bench_bars=bench_bars, cost_bps=args.cost_bps, periods_per_year=PPY)
        is_sh, oos_sh = oos(r.equity_curve)
        sweep.append({"thr": thr, "sharpe": r.sharpe, "is": is_sh, "oos": oos_sh,
                      "ret": r.total_return, "maxdd": r.max_drawdown, "beta": r.beta,
                      "equity": r.equity_curve})
        print(f"{thr:>9.1f}{r.sharpe:>8.2f}{is_sh:>7.2f}{oos_sh:>7.2f}{r.total_return*100:>7.0f}%"
              f"{r.max_drawdown*100:>7.1f}%{r.beta:>7.2f}")
    pp = [x["sharpe"] / (PPY ** 0.5) for x in sweep]
    var_trials = (statistics.pvariance(pp) if len(pp) > 1 else 1e-5) or 1e-5
    best = max(sweep, key=lambda x: x["sharpe"])
    eqh = best["equity"]
    psr = probabilistic_sharpe_ratio(eqh)
    dsr = deflated_sharpe_ratio(eqh, n_trials=args.n_trials, sharpe_variance=var_trials)
    print(f"  best thr {best['thr']}: Sharpe {best['sharpe']:.2f}, IS {best['is']:.2f}/OOS "
          f"{best['oos']:.2f}, PSR {psr:.2f}, DSR {dsr:.2f} (deflated {args.n_trials} trials)")
    bh_mode = best

    # ---- PROFIT-PER-DAY at half-Kelly: the honest answer to "max profit/day" ----
    print("\n" + "=" * 72)
    print("MAX PROFIT-PER-DAY (honest) = surviving edge x risk budget, NOT trade frequency.")
    print(f"{'sleeve':>14}{'Sharpe':>8}{'fullKelly g/yr':>16}{'halfKelly g/yr':>16}{'~bps/day':>10}{'noise x':>9}")
    print("-" * 73)
    for label, m in (("long-only", rows["long"]), ("beta_hedged", bh_mode)):
        S = m["sharpe"]
        # geometric growth ceiling: full Kelly ~ S^2/2 per year; half Kelly ~ (3/8)S^2
        g_full = max(S, 0.0) ** 2 / 2.0
        g_half = 0.375 * max(S, 0.0) ** 2
        bps_day = g_half / 252.0 * 1e4
        # daily noise at the half-Kelly vol target sigma_target = S (so growth=S^2*...); express noise/edge
        # edge/day = S^2*.../252; noise/day = sigma_target/sqrt(252) with sigma_target~S(annual vol units)
        sigma_t = max(S, 1e-9)               # half-Kelly targets ~Sharpe-proportional vol
        noise_day = sigma_t / math.sqrt(252) * 1e4
        edge_day = max(bps_day, 1e-9)
        print(f"{label:>14}{S:>8.2f}{g_full*100:>14.0f}% {g_half*100:>14.0f}% {bps_day:>9.1f}{noise_day/edge_day:>8.0f}x")
    print("  Read: even a real Sharpe ~0.7 sleeve maxes near ~7 bps/day geometric at half-Kelly, under\n"
          "  ~20x that in daily noise. 'Highest profit/day' = push DSR-surviving Sharpe up, then size to\n"
          "  Kelly. Frequency does the opposite (Case 17: 2x/day turnover turned Sharpe 0.93 -> -0.41).")

    Path(args.out).write_text(json.dumps(
        {"n_symbols": len(events_by),
         "modes": {m: {k: v for k, v in rows[m].items() if k != "equity"} for m in rows},
         "beta_hedged_dsr": dsr, "beta_hedged_psr": psr}, indent=2))
    print(f"\n[done] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
