"""
Case 53 — Cross-sectional SHORT-HORIZON REVERSAL as a new stock-based leg, with the FULL post-audit bar.

A documented equity anomaly (long last-week's losers / short last-week's winners), market-neutral, a
different mechanism + horizon than the pairs basket (pairwise cointegration), so plausibly uncorrelated
and cadence-orthogonal. The audit (Case 52) flagged it as a candidate. The honest bar, post-audit:
  - the COST WALL (reversal is turnover-heavy) — Sharpe at 2 / 5 / 10 bps
  - fresh-symbol holdout (generalizes out-of-universe)
  - per-year AND the 2024+ slice (the test that caught short-vol's pre-2024-only lift)
  - the SECOND-LEG GATE vs the cached pairs walk-forward book (positive + uncorrelated + LIFTS + robust)

Run: .venv/bin/python scripts/test_reversal.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from alpca.backtest.factor import backtest_factor, short_horizon_return_signal  # noqa: E402
from alpca.backtest.leg_gate import evaluate_leg_candidate  # noqa: E402
from alpca.backtest.evaluation import sharpe_of  # noqa: E402

PPY = 252.0


def _load(c):
    return {p.name.split("_1day_")[0]: [json.loads(l) for l in p.open() if l.strip()]
            for p in Path(c).glob("*_1day_bars.jsonl")}


def _eq(daily, s=1.0):
    e = [s]
    for x in daily:
        e.append(e[-1] * (1 + x))
    return e


def _run(bars, *, window, rebal, cost_bps):
    return backtest_factor(bars, short_horizon_return_signal(window), name="reversal", top_frac=0.2,
                           rebalance_days=rebal, cost_bps=cost_bps, long_high=False, periods_per_year=PPY)


def _slice_sharpe(dates, daily, year_from):
    sub = [x for ep, x in zip(dates, daily) if time.gmtime(ep).tm_year >= year_from]
    return sharpe_of(_eq(sub), PPY) if len(sub) > 30 else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--largecap", default="/Volumes/My Passport/AlpcaData/cache_largecap_sip")
    ap.add_argument("--midcap", default="/Volumes/My Passport/AlpcaData/cache_midcap_sip")
    ap.add_argument("--pairs", default="data/pairs_wf_returns.json")
    ap.add_argument("--out", default="data/reversal_results.json")
    args = ap.parse_args()

    universes = {"large-cap": _load(args.largecap), "mid-cap": _load(args.midcap)}
    pj = json.loads(Path(args.pairs).read_text())
    book = {int(r["asof"]): r["ret"] for r in pj["returns"]}
    out = {}

    # 1) COST WALL + horizon/rebalance sweep on each universe (does any config survive cost?)
    print(f"{'universe':>10}{'win':>5}{'rebal':>7}{'2bps':>7}{'5bps':>7}{'10bps':>7}{'turn':>7}")
    print("-" * 50)
    best = {}
    for uname, bars in universes.items():
        if len(bars) < 20:
            continue
        for window in (3, 5, 10):
            for rebal in (5, 10):
                r2 = _run(bars, window=window, rebal=rebal, cost_bps=2.0)
                r5 = _run(bars, window=window, rebal=rebal, cost_bps=5.0)
                r10 = _run(bars, window=window, rebal=rebal, cost_bps=10.0)
                print(f"{uname:>10}{window:>5}{rebal:>7}{r2.sharpe:>7.2f}{r5.sharpe:>7.2f}"
                      f"{r10.sharpe:>7.2f}{r2.avg_turnover:>7.2f}")
                # track the best 5bps config per universe (5bps ~ honest reversal cost)
                if uname not in best or r5.sharpe > best[uname][0]:
                    best[uname] = (r5.sharpe, window, rebal)
        print()

    # 2) full bar on the best config per universe: fresh-holdout + per-year + 2024+ + leg gate
    for uname, bars in universes.items():
        if uname not in best:
            continue
        _, window, rebal = best[uname]
        syms = sorted(bars)
        hold = {s: bars[s] for i, s in enumerate(syms) if i % 3 == 0}
        full = _run(bars, window=window, rebal=rebal, cost_bps=5.0)
        h = _run(hold, window=window, rebal=rebal, cost_bps=5.0)
        by = {}
        for ep, x in zip(full.dates, full.daily_returns):
            by.setdefault(time.gmtime(ep).tm_year, []).append(x)
        yr = {y: round(sharpe_of(_eq(v), PPY), 2) for y, v in sorted(by.items()) if len(v) >= 30}
        s24 = _slice_sharpe(full.dates, full.daily_returns, 2024)
        cand = {int(t): x for t, x in zip(full.dates, full.daily_returns)}
        v = evaluate_leg_candidate(cand, book, book_label="pairs")
        print(f"=== {uname} reversal (win{window}/rebal{rebal}, 5bps) ===")
        print(f"  full {full.sharpe:.2f} · fresh-holdout {h.sharpe:.2f} · 2024+ {s24:.2f} · per-year {yr}")
        print(f"  LEG GATE vs pairs: candidate {v.candidate_sharpe} · ρ {v.rho} · combined {v.combined_sharpe} "
              f"(lift {v.lift:+}) · LOO {v.loo_positive_frac:.0%} · ex-recent {v.ex_recent_lift:+}")
        print(f"  checks: { {k: ('PASS' if ok else 'fail') for k,ok in v.checks.items()} }")
        verdict = ("CANDIDATE" if (full.sharpe > 0.15 and h.sharpe > 0 and (s24 == s24 and s24 > 0) and v.passed)
                   else "reject")
        print(f"  -> {verdict}\n")
        out[uname] = {"window": window, "rebal": rebal, "full_5bps": round(full.sharpe, 3),
                      "holdout": round(h.sharpe, 3), "y2024plus": round(s24, 3) if s24 == s24 else None,
                      "per_year": yr, "leg_gate_pass": v.passed, "lift": v.lift, "rho": v.rho,
                      "verdict": verdict}

    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"[saved] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
