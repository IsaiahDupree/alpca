"""
OVERFITTING AUDIT of the EAR-PEAD beta-hedged leg — the one sleeve we actually put in the
combiner. Subjects it to the harsher tests the case study skipped:

  1. FIXED a-priori params (thr 2.0 default, NOT the cherry-picked 1.5) — no selection on the test.
  2. TRAILING beta hedge (126d) vs FULL-SAMPLE — removes the hedge-ratio lookahead.
  3. PER-CALENDAR-YEAR Sharpe — regime stability is the real anti-overfit signal (does it survive
     2022's bear, not just the bull years?).
  4. DSR at HONEST, escalating trial counts (37 -> 100 -> 200) — shows how much of the "significance"
     is just under-counting the project's true search breadth.

If the leg is real it stays positive across most years, barely moves under a trailing hedge, and
keeps a non-trivial DSR even at 200 trials. If it's overfit, one or more of these collapses.

Run: .venv/bin/python scripts/audit_overfit.py
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
    deflated_sharpe_ratio, probabilistic_sharpe_ratio, sharpe_of)
from alpca.backtest.ear_pead import backtest_ear_pead  # noqa: E402

PPY = 252.0


def _eq_from(daily, start=100_000.0):
    eq = [start]
    for x in daily:
        eq.append(eq[-1] * (1 + x))
    return eq


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/Volumes/My Passport/AlpcaData/cache")
    ap.add_argument("--earnings", default="/Volumes/My Passport/AlpcaData/earnings_av")
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
    bench = [json.loads(l) for l in (cache / "SPY_1day_bars.jsonl").open() if l.strip()]
    print(f"[ok] {len(events_by)} symbols, fixed a-priori params (thr 2.0, hold 40)\n")

    def run(hw):
        return backtest_ear_pead(bars_by, events_by, hold=40, entry_thr=2.0, mode="beta_hedged",
                                 bench_bars=bench, hedge_window=hw, cost_bps=2.0, periods_per_year=PPY)

    full = run(0)        # full-sample beta (lookahead)
    trail = run(126)     # trailing 6-month beta (no lookahead)

    # ---- 1+2: full vs trailing hedge ----
    print("=== HEDGE LOOKAHEAD TEST (does removing the full-sample hedge ratio hurt?) ===")
    print(f"  full-sample beta hedge : Sharpe {full.sharpe:.2f}  (beta {full.beta:.2f})")
    print(f"  TRAILING 126d beta hedge: Sharpe {trail.sharpe:.2f}  (avg beta {trail.beta:.2f})")
    drop = full.sharpe - trail.sharpe
    print(f"  -> {'OK: trailing hedge barely changes it' if abs(drop) < 0.25 else 'WARNING: leans on the full-sample hedge'} (Δ {drop:+.2f})\n")

    # ---- 3: per-calendar-year regime stability (on the honest trailing-hedge sleeve) ----
    print("=== REGIME STABILITY — per-calendar-year Sharpe (trailing hedge) ===")
    by_year = {}
    for ep, r in zip(trail.dates, trail.daily_returns):
        by_year.setdefault(time.gmtime(ep).tm_year, []).append(r)
    pos_years = 0
    yrs = sorted(by_year)
    for y in yrs:
        rr = by_year[y]
        if len(rr) < 30:
            print(f"  {y}: (only {len(rr)}d, skipped)")
            continue
        sh = sharpe_of(_eq_from(rr), PPY)
        pos_years += 1 if sh > 0 else 0
        bar = "+" * max(0, int(sh * 5)) or ("-" * max(0, int(-sh * 5)))
        print(f"  {y}: Sharpe {sh:>5.2f}  ({len(rr)}d)  {bar}")
    scored = [y for y in yrs if len(by_year[y]) >= 30]
    print(f"  -> positive in {pos_years}/{len(scored)} scored years "
          f"({'robust across regimes' if pos_years >= max(1, len(scored) - 1) else 'CONCENTRATED in a few years -> overfit risk'})\n")

    # ---- 4: DSR honesty — escalate the trial count to the project's TRUE search breadth ----
    print("=== DEFLATION HONESTY — DSR vs trial count (trailing-hedge sleeve) ===")
    # a small same-direction threshold sweep gives the per-trial Sharpe variance
    sweep = []
    for thr in (1.0, 1.5, 2.0, 3.0, 4.0):
        r = backtest_ear_pead(bars_by, events_by, hold=40, entry_thr=thr, mode="beta_hedged",
                              bench_bars=bench, hedge_window=126, cost_bps=2.0, periods_per_year=PPY)
        sweep.append(r.sharpe / (PPY ** 0.5))
    var_trials = (statistics.pvariance(sweep) if len(sweep) > 1 else 1e-5) or 1e-5
    psr = probabilistic_sharpe_ratio(trail.equity_curve)
    print(f"  PSR(>0): {psr:.2f}")
    for nt in (37, 100, 200, 400):
        dsr = deflated_sharpe_ratio(trail.equity_curve, n_trials=nt, sharpe_variance=var_trials)
        print(f"  DSR @ {nt:>3} trials: {dsr:.2f}")
    print("  -> the project's TRUE config count is in the hundreds; read the 200-400 column as the\n"
          "     honest one. If DSR holds there, the edge is robust to our own search breadth.\n")

    print("=" * 60)
    print("VERDICT: real iff (trailing≈full) AND (positive most years) AND (DSR survives ~200 trials).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
