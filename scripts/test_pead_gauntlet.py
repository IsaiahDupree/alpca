"""
Case 55 — PEAD 3rd-leg hunt, the DEFINITIVE re-test through the full post-audit gauntlet.

PEAD was rejected twice (surprise-PEAD died to adverse-selection borrow, Case 14; EAR-PEAD failed the
fresh-symbol holdout, Case 18). This re-runs the strongest variants on the available data (40 mega-cap
+ 19 disjoint holdout names — earnings are HARD data-gated at 40/195, AV free tier now rate-limits) with
every modern bar: adverse borrow · fresh-symbol holdout · per-year + 2024+ slice · the SECOND-LEG GATE vs
the cadence-fixed pairs book. EAR-PEAD (price-only announcement-return + cheap index short) is the
borrow-free form most likely to survive and exposes dated returns for the leg gate.

Run: .venv/bin/python scripts/test_pead_gauntlet.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from alpca.backtest.ear_pead import backtest_ear_pead  # noqa: E402
from alpca.backtest.pead import backtest_pead  # noqa: E402
from alpca.backtest.leg_gate import evaluate_leg_candidate  # noqa: E402
from alpca.backtest.evaluation import sharpe_of, deflated_sharpe_ratio  # noqa: E402

PPY = 252.0


def _events(d):
    out = {}
    for p in Path(d).glob("*.json"):
        sym = p.name.replace("_earnings.json", "").replace(".json", "")
        rows = json.loads(p.read_text())
        ev = [{"date": int(r["date"]), "surprise_pct": r.get("surprise_pct")}
              for r in rows if r.get("date") and r.get("surprise_pct") is not None]
        if ev:
            out[sym] = ev
    return out


def _bars(syms, *dirs):
    out = {}
    for s in syms:
        for d in dirs:
            f = Path(d) / f"{s}_1day_bars.jsonl"
            if f.exists():
                out[s] = [json.loads(l) for l in f.open() if l.strip()]
                break
    return out


def _eq(daily, s=1.0):
    e = [s]
    for x in daily:
        e.append(e[-1] * (1 + x))
    return e


def _slice(dates, daily, yr_from):
    sub = [x for ep, x in zip(dates, daily) if time.gmtime(ep).tm_year >= yr_from]
    return sharpe_of(_eq(sub), PPY) if len(sub) > 30 else float("nan")


def _per_year(dates, daily):
    by = {}
    for ep, x in zip(dates, daily):
        by.setdefault(time.gmtime(ep).tm_year, []).append(x)
    return {y: round(sharpe_of(_eq(v), PPY), 2) for y, v in sorted(by.items()) if len(v) >= 20}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/Volumes/My Passport/AlpcaData/cache")
    ap.add_argument("--cache2", default="/Volumes/My Passport/AlpcaData/cache_largecap_sip")
    ap.add_argument("--ev-main", default="/Volumes/My Passport/AlpcaData/earnings_av")
    ap.add_argument("--ev-hold", default="/Volumes/My Passport/AlpcaData/earnings_av_holdout")
    ap.add_argument("--pairs", default="data/pairs_wf_returns.json")
    args = ap.parse_args()

    ev_m, ev_h = _events(args.ev_main), _events(args.ev_hold)
    bars_m = _bars(list(ev_m), args.cache, args.cache2)
    bars_h = _bars(list(ev_h), args.cache, args.cache2)
    spy = _bars(["SPY"], args.cache, args.cache2).get("SPY")
    book = {int(r["asof"]): r["ret"] for r in json.loads(Path(args.pairs).read_text())["returns"]}
    print(f"[ok] main {len(bars_m)} sym (earnings+bars) · holdout {len(bars_h)} · SPY {'yes' if spy else 'NO'}\n")

    # ---- EAR-PEAD beta-hedged (borrow-free index short, trailing beta — no lookahead) ----
    print("=== EAR-PEAD (beta-hedged, trailing beta) ===")
    for thr in (1.0, 2.0):
        r = backtest_ear_pead(bars_m, ev_m, hold=40, ear_window=3, entry_thr=thr, mode="beta_hedged",
                              bench_bars=spy, hedge_window=60, cost_bps=2.0, periods_per_year=PPY)
        rh = backtest_ear_pead(bars_h, ev_h, hold=40, ear_window=3, entry_thr=thr, mode="beta_hedged",
                               bench_bars=spy, hedge_window=60, cost_bps=2.0, periods_per_year=PPY)
        s24 = _slice(r.dates, r.daily_returns, 2024)
        cand = {int(t): x for t, x in zip(r.dates, r.daily_returns)}
        v = evaluate_leg_candidate(cand, book, book_label="pairs")
        dsr = deflated_sharpe_ratio(_eq(r.daily_returns), n_trials=110, sharpe_variance=1e-4)
        print(f" thr{thr}: main {r.sharpe:+.2f} · FRESH-HOLDOUT {rh.sharpe:+.2f} · 2024+ {s24:+.2f} · DSR {dsr:.2f}")
        print(f"        per-year {_per_year(r.dates, r.daily_returns)}")
        print(f"        leg-gate: ρ {v.rho} · lift {v.lift:+} · {'PASS' if v.passed else 'NO-GO'} "
              f"{ {k:('Y' if x else 'n') for k,x in v.checks.items()} }")
        gen = rh.sharpe > 0
        print(f"        -> {'CANDIDATE' if (r.sharpe>0.2 and gen and s24==s24 and s24>0 and v.passed) else 'reject'}"
              f" (fresh-holdout {'+' if gen else 'NEG'})\n")

    # ---- surprise-PEAD L/S under ADVERSE borrow (the Case-14 killer) ----
    print("=== surprise-PEAD L/S (adverse-selection borrow, the honest stress) ===")
    rf = backtest_pead(bars_m, ev_m, hold=30, entry_thr=2.0, leg="both", cost_bps=2.0)
    ra = backtest_pead(bars_m, ev_m, hold=30, entry_thr=2.0, leg="both", cost_bps=2.0,
                       adverse_borrow={"base": 0.01, "special": 0.30, "sat": 50.0, "no_locate": 200.0})
    rs = backtest_pead(bars_m, ev_m, hold=30, entry_thr=2.0, leg="short", cost_bps=2.0,
                       adverse_borrow={"base": 0.01, "special": 0.30, "sat": 50.0, "no_locate": 200.0})
    print(f"  flat-borrow {rf.sharpe:+.2f} · ADVERSE-borrow {ra.sharpe:+.2f} · short-leg-alone {rs.sharpe:+.2f}")
    print(f"  -> {'survives adverse borrow' if ra.sharpe > 0.3 else 'dies to adverse borrow (short leg the problem)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
