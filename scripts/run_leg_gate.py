"""
Run the SECOND-LEG GATE on real candidates against the deployed pairs book — the reusable tool that
turns "is this a real diversifying leg?" into one call (and validates the gate reproduces the Cases
47-51 hand verdicts: short-vol GO, momentum / seasonality NO-GO).

Run: .venv/bin/python scripts/run_leg_gate.py [shortvol|momentum|seasonality]
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from alpca.backtest.pairs import delisting_aware_walkforward  # noqa: E402
from alpca.backtest.leg_gate import evaluate_leg_candidate  # noqa: E402
from scripts.test_two_sleeve_combiner import (  # noqa: E402
    _load, momentum_long_hedge_returns, seasonality_returns)

LC = "/Volumes/My Passport/AlpcaData/cache_largecap_sip"
MC = "/Volumes/My Passport/AlpcaData/cache_midcap_sip"
VOL = "/Volumes/My Passport/AlpcaData/cache_vol"
SPYC = "/Volumes/My Passport/AlpcaData/cache"


def _shortvol():
    b = sorted(_load(VOL)["SVXY"], key=lambda x: int(x["timestamp"]))
    return {int(b[i]["timestamp"]): float(b[i]["close"]) / float(b[i - 1]["close"]) - 1.0
            for i in range(1, len(b)) if float(b[i - 1]["close"]) > 0}


def main() -> int:
    which = sys.argv[1] if len(sys.argv) > 1 else "shortvol"
    pr = delisting_aware_walkforward(_load(LC), train=252, test=63, top_n=10, max_adf=-2.86)
    book = {int(t): r for t, r in zip(pr.dates, pr.daily_returns)}
    print(f"[book] pairs WF {pr.sharpe:.2f} · {len(book)} OOS days\n")

    if which == "shortvol":
        cand = _shortvol()
    elif which == "momentum":
        mc = _load(MC); spy_bars = _load(SPYC).get("SPY", [])
        spy = {int(b["timestamp"]): None for b in spy_bars}
        sb = sorted(spy_bars, key=lambda x: int(x["timestamp"]))
        spy = {int(sb[i]["timestamp"]): float(sb[i]["close"]) / float(sb[i - 1]["close"]) - 1.0
               for i in range(1, len(sb)) if float(sb[i - 1]["close"]) > 0}
        cand = momentum_long_hedge_returns(mc, spy)
    elif which == "seasonality":
        cand = seasonality_returns(_load(MC))
    else:
        print(f"unknown candidate {which}"); return 1

    v = evaluate_leg_candidate(cand, book, book_label="pairs")
    print(f"=== SECOND-LEG GATE: {which} vs pairs book ===")
    print(f"  candidate Sharpe {v.candidate_sharpe} · book {v.book_sharpe} · ρ {v.rho} · "
          f"combined {v.combined_sharpe} (lift {v.lift:+})")
    print(f"  leave-one-year-out lift positive: {v.loo_positive_frac:.0%} · ex-recent-year lift {v.ex_recent_lift:+}")
    for k, ok in v.checks.items():
        print(f"    [{'PASS' if ok else 'FAIL'}] {k}")
    print(f"\n  VERDICT: {'GO — real diversifying leg' if v.passed else 'NO-GO'}")
    for r in v.reasons:
        print(f"    - {r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
