"""
Harness-test PEAD (post-earnings-announcement drift), judging the LONG, SHORT, and dollar-
neutral COMBINED legs separately. Earnings surprise from the cache built by
alpca.data.earnings.download_universe_earnings (Nasdaq free ~1yr, or Finnhub if a key is set).

Market-neutral combined leg -> the return itself is the alpha (no buy-and-hold). Honest null:
PEAD has decayed since the 2000s; the long leg is often just beta; any real edge lives in the
short leg / dollar-neutral combo. CAVEAT: free Nasdaq surprise is only ~4 quarters/ticker ->
a ~1yr window = weak statistical power, single regime. A multi-year test needs a FINNHUB_API_KEY.

Run: .venv/bin/python scripts/test_pead.py --cache "/Volumes/My Passport/AlpcaData/cache" \
       --earnings "/Volumes/My Passport/AlpcaData/earnings"
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from alpca.backtest.evaluation import (  # noqa: E402
    deflated_sharpe_ratio, probabilistic_sharpe_ratio, sharpe_of, sharpe_pvalue, sharpe_tstat)
from alpca.backtest.pead import backtest_pead  # noqa: E402

PPY = 252.0


def oos(eq, frac=0.3):
    sp = int(len(eq) * (1 - frac))
    return sharpe_of(eq[:sp], PPY), sharpe_of(eq[sp:], PPY)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/Volumes/My Passport/AlpcaData/cache")
    ap.add_argument("--earnings", default="/Volumes/My Passport/AlpcaData/earnings_av")
    ap.add_argument("--hold", type=int, default=30)
    ap.add_argument("--entry-thr", type=float, default=2.0)
    ap.add_argument("--cost-bps", type=float, default=2.0)
    ap.add_argument("--n-trials", type=int, default=34, help="DSR deflation: project search breadth")
    ap.add_argument("--borrow-aprs", default="0.0,0.01,0.03,0.10",
                    help="short-borrow APRs to stress (large-cap GC ~0.3-1%%, HTB much higher)")
    ap.add_argument("--out", default="data/pead_results.json")
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
    n_events = sum(len(v) for v in events_by.values())
    spans = [e["date"] for v in events_by.values() for e in v]
    print(f"[ok] {len(events_by)} symbols, {n_events} earnings events "
          f"(window ~{(max(spans)-min(spans))/86400/365:.1f}yr)\n" if spans else "[fail] no events")
    if not spans:
        return 1

    grid = [(0.0, "long"), (0.0, "short"), (0.0, "both"),
            (1.0, "both"), (3.0, "both")]
    rows = []
    print(f"{'entry_thr':>9}{'leg':>7}{'sharpe':>8}{'IS':>7}{'OOS':>7}{'ret':>8}{'maxDD':>8}"
          f"{'events':>8}{'active':>8}{'sig':>5}")
    print("-" * 80)
    for thr, leg in [(args.entry_thr, "long"), (args.entry_thr, "short"), (args.entry_thr, "both"),
                     (1.0, "both"), (3.0, "both")]:
        r = backtest_pead(bars_by, events_by, hold=args.hold, entry_thr=thr, leg=leg,
                          cost_bps=args.cost_bps, periods_per_year=PPY)
        if r.n_days < 60:
            continue
        is_sh, oos_sh = oos(r.equity_curve)
        t, pv = sharpe_tstat(r.equity_curve), sharpe_pvalue(r.equity_curve)
        sig = pv < 0.05 and abs(t) > 2.0
        rows.append({"entry_thr": thr, "leg": leg, "sharpe": r.sharpe, "is_sharpe": is_sh,
                     "oos_sharpe": oos_sh, "ret": r.total_return, "maxdd": r.max_drawdown,
                     "events": r.n_events_used, "avg_active": r.avg_active, "sig": sig})
        print(f"{thr:>9.1f}{leg:>7}{r.sharpe:>8.2f}{is_sh:>7.2f}{oos_sh:>7.2f}"
              f"{r.total_return*100:>7.0f}%{r.max_drawdown*100:>7.1f}%{r.n_events_used:>8}"
              f"{r.avg_active:>8.1f}{('Y' if sig else ''):>5}")

    both = next((x for x in rows if x["leg"] == "both" and x["entry_thr"] == args.entry_thr), None)
    longl = next((x for x in rows if x["leg"] == "long"), None)
    shortl = next((x for x in rows if x["leg"] == "short"), None)
    print("\n" + "=" * 80)
    if both and longl and shortl:
        # DSR: deflate the best dollar-neutral config for the project's search breadth.
        # Units must be PER-PERIOD (PSR uses per-period Sharpe) -> divide annual Sharpes by sqrt(ppy).
        dn_pp = [x["sharpe"] / (PPY ** 0.5) for x in rows if x["leg"] == "both"]
        var_trials = (statistics.pvariance(dn_pp) if len(dn_pp) > 1 else 1e-5) or 1e-5
        best_dn = max((x for x in rows if x["leg"] == "both"), key=lambda x: x["sharpe"])
        eq = backtest_pead(bars_by, events_by, hold=args.hold, entry_thr=best_dn["entry_thr"],
                           leg="both", cost_bps=args.cost_bps, periods_per_year=PPY).equity_curve
        psr = probabilistic_sharpe_ratio(eq)
        # n_trials reflects the project's broad search (~34 strategies + edge families), not just
        # this 5-config sweep — the honest, conservative deflation count.
        dsr = deflated_sharpe_ratio(eq, n_trials=args.n_trials, sharpe_variance=var_trials)
        print(f"LEG DECOMPOSITION (entry_thr {args.entry_thr}):")
        print(f"  long-only  Sharpe {longl['sharpe']:.2f} (likely beta)")
        print(f"  short-only Sharpe {shortl['sharpe']:.2f} (where neutral alpha would live)")
        print(f"  dollar-neutral Sharpe {both['sharpe']:.2f}, IS {both['is_sharpe']:.2f}, OOS {both['oos_sharpe']:.2f}")
        print(f"  best dollar-neutral (thr {best_dn['entry_thr']}): Sharpe {best_dn['sharpe']:.2f}  "
              f"PSR(>0) {psr:.2f}  DSR(deflated) {dsr:.2f}")
        real_oos = both["is_sharpe"] > 0.2 and both["oos_sharpe"] > 0.3
        verdict = ("dollar-neutral PEAD holds across a REAL walk-forward (IS & OOS both positive) "
                   "AND survives Deflated-Sharpe — a genuine market-neutral edge candidate"
                   if real_oos and dsr > 0.90 else
                   "positive but does NOT clear Deflated-Sharpe / walk-forward bar — not validated"
                   if both["oos_sharpe"] > 0.3 else
                   "no convincing edge across the walk-forward")
        print(f"  VERDICT: {verdict}")
    # ---- SHORTING REALISM: borrow-fee stress on the dollar-neutral leg ----
    print("\n" + "=" * 80)
    print("SHORTING REALISM — borrow fee charged daily on the short notional (dollar-neutral, "
          f"thr {args.entry_thr}):")
    print(f"{'borrow_apr':>11}{'sharpe':>8}{'OOS':>7}{'ret':>8}{'PSR':>7}{'DSR':>7}")
    print("-" * 50)
    borrow_rows = []
    for apr in [float(x) for x in args.borrow_aprs.split(",")]:
        r = backtest_pead(bars_by, events_by, hold=args.hold, entry_thr=args.entry_thr, leg="both",
                          cost_bps=args.cost_bps, borrow_apr=apr, periods_per_year=PPY)
        _, oos_sh = oos(r.equity_curve)
        psr_a = probabilistic_sharpe_ratio(r.equity_curve)
        dsr_a = deflated_sharpe_ratio(r.equity_curve, n_trials=args.n_trials,
                                      sharpe_variance=(var_trials if both else 1e-5))
        borrow_rows.append({"borrow_apr": apr, "sharpe": r.sharpe, "oos": oos_sh,
                            "ret": r.total_return, "psr": psr_a, "dsr": dsr_a})
        print(f"{apr*100:>10.1f}%{r.sharpe:>8.2f}{oos_sh:>7.2f}{r.total_return*100:>7.0f}%"
              f"{psr_a:>7.2f}{dsr_a:>7.2f}")
    gc = next((x for x in borrow_rows if abs(x["borrow_apr"] - 0.01) < 1e-9), borrow_rows[0])
    print(f"\n  At a realistic large-cap GC borrow (~1%/yr): Sharpe {gc['sharpe']:.2f}, DSR {gc['dsr']:.2f}. "
          f"Borrow drag on a dollar-neutral book (0.5 short notional) is ~apr/2 per year — modest for "
          f"GC names, material only if shorts become hard-to-borrow.")

    Path(args.out).write_text(json.dumps({"n_symbols": len(events_by), "n_events": n_events,
                                          "grid": rows, "borrow_stress": borrow_rows}, indent=2))
    print(f"\n[done] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
