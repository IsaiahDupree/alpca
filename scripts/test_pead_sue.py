"""
Case 59 — PEAD with REAL SUE (standardized unexpected earnings) vs raw surprise_pct,
at the now-broader 63-symbol breadth, under the adverse-selection borrow stress, with a
fresh-symbol holdout (the 19 disjoint names that killed EAR-PEAD in Case 18).

The standing bar (from memory / Case 14): "survive adverse_borrow at FULL breadth + try real
SUE vs raw surprise_pct." Raw surprise_pct = (eps-consensus)/|consensus| is wildly fat-tailed
(division by a near-zero consensus explodes it); the academically-correct PEAD signal is SUE =
(eps - consensus) / std(firm's own past unexpected-earnings) — a clean, comparable z-score.

What this tests:
  - SUE vs raw surprise as the ranking signal (entry threshold in SUE std-units for SUE).
  - dollar-neutral ("both") leg under ADVERSE borrow (base 1% -> special 30%, no-locate >200%).
  - long / short / both decomposition (long is usually beta; alpha must live in short/neutral).
  - IS/OOS split AND the decisive FRESH-SYMBOL HOLDOUT: same fixed rule on the disjoint 19.

Verdict rule: SUE-PEAD is a CANDIDATE 3rd leg only if the dollar-neutral leg is positive OOS
*and* survives adverse borrow *and* stays positive on the fresh-19 holdout. Else REJECT.

Run: .venv/bin/python scripts/test_pead_sue.py
Writes: data/pead_sue_results.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from alpca.backtest.evaluation import (  # noqa: E402
    deflated_sharpe_ratio, probabilistic_sharpe_ratio, sharpe_of)
from alpca.backtest.pead import backtest_pead  # noqa: E402

PPY = 252.0
ADVERSE = {"base": 0.01, "special": 0.30, "sat": 50.0, "no_locate": 200.0}


def oos(eq, frac=0.3):
    sp = int(len(eq) * (1 - frac))
    return round(sharpe_of(eq[:sp], PPY), 3), round(sharpe_of(eq[sp:], PPY), 3)


def add_sue(events, min_priors=4):
    """No-lookahead SUE per event: UE = eps - consensus; SUE_t = UE_t / std(UE_{<t})
    over an expanding window (>= min_priors past events). Events before enough history,
    or missing eps/consensus, simply get no 'sue' field (and are skipped by the backtest)."""
    evs = sorted([e for e in events if e.get("eps") is not None and e.get("consensus") is not None],
                 key=lambda e: e["date"])
    ue_hist = []
    out = []
    for e in evs:
        ue = float(e["eps"]) - float(e["consensus"])
        if len(ue_hist) >= min_priors:
            sd = statistics.pstdev(ue_hist)
            if sd > 0:
                e = {**e, "sue": ue / sd}
        ue_hist.append(ue)
        out.append(e)
    return out


def load(edir: Path, cache: Path):
    bars_by, ev_raw, ev_sue = {}, {}, {}
    for ef in edir.glob("*_earnings.json"):
        sym = ef.name.replace("_earnings.json", "")
        ev = json.loads(ef.read_text())
        bf = cache / f"{sym}_1day_bars.jsonl"
        if ev and bf.exists():
            bars_by[sym] = [json.loads(l) for l in bf.open() if l.strip()]
            ev_raw[sym] = ev
            ev_sue[sym] = add_sue(ev)
    return bars_by, ev_raw, ev_sue


def run(bars_by, events_by, *, signal_field, entry_thr, borrow_field="surprise_pct",
        adverse=None):
    out = {}
    for leg in ("long", "short", "both"):
        r = backtest_pead(bars_by, events_by, hold=30, entry_thr=entry_thr, leg=leg,
                          cost_bps=2.0, adverse_borrow=adverse, signal_field=signal_field,
                          borrow_field=borrow_field, periods_per_year=PPY)
        is_sh, oos_sh = oos(r.equity_curve)
        out[leg] = {"sharpe": round(r.sharpe, 3), "is": is_sh, "oos": oos_sh,
                    "ret": round(r.total_return, 3), "maxdd": round(r.max_drawdown, 3),
                    "events": r.n_events_used, "no_locate": r.n_no_locate,
                    "equity": r.equity_curve, "daily": r.daily_returns,
                    "psr": round(probabilistic_sharpe_ratio(r.equity_curve), 3)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/Volumes/My Passport/AlpcaData/cache")
    ap.add_argument("--train", default="/Volumes/My Passport/AlpcaData/earnings_av")
    ap.add_argument("--holdout", default="/Volumes/My Passport/AlpcaData/earnings_av_holdout")
    ap.add_argument("--out", default="data/pead_sue_results.json")
    args = ap.parse_args()
    cache = Path(args.cache)

    tr_bars, tr_raw, tr_sue = load(Path(args.train), cache)
    ho_bars, ho_raw, ho_sue = load(Path(args.holdout), cache)
    n_sue_ev = sum(1 for s in tr_sue for e in tr_sue[s] if "sue" in e)
    print(f"[ok] train {len(tr_bars)} syms / holdout {len(ho_bars)} syms · "
          f"{n_sue_ev} train events have SUE\n")

    results = {}
    configs = [
        ("RAW surprise_pct  thr2.0  flat-borrow",  tr_raw, "surprise_pct", 2.0, None),
        ("RAW surprise_pct  thr2.0  ADVERSE",      tr_raw, "surprise_pct", 2.0, ADVERSE),
        ("SUE               thr1.0  flat-borrow",  tr_sue, "sue", 1.0, None),
        ("SUE               thr1.0  ADVERSE",      tr_sue, "sue", 1.0, ADVERSE),
        ("SUE               thr1.5  ADVERSE",      tr_sue, "sue", 1.5, ADVERSE),
    ]
    print(f"{'config':<42}{'long':>7}{'short':>7}{'both':>7}{'both-OOS':>10}{'PSR':>6}{'noLoc':>7}")
    print("-" * 86)
    both_sharpes = []
    for name, ev, fld, thr, adv in configs:
        r = run(tr_bars, ev, signal_field=fld, entry_thr=thr, adverse=adv)
        results[name] = r
        both_sharpes.append(r['both']['sharpe'])
        print(f"{name:<42}{r['long']['sharpe']:>7.2f}{r['short']['sharpe']:>7.2f}"
              f"{r['both']['sharpe']:>7.2f}{r['both']['oos']:>10.2f}{r['both']['psr']:>6.2f}"
              f"{r['both']['no_locate']:>7}")
    # DSR for the headline SUE-adverse config, deflated by the variance across the configs searched
    sv = statistics.pvariance(both_sharpes) if len(both_sharpes) > 1 else 1e-5
    sue_eq = results["SUE               thr1.0  ADVERSE"]["both"]["equity"]
    headline_dsr = round(deflated_sharpe_ratio(sue_eq, n_trials=len(configs), sharpe_variance=sv or 1e-5), 3)
    print(f"\nSUE-adverse headline DSR (deflated over {len(configs)} configs): {headline_dsr}")

    # ---- decisive: fresh-symbol holdout with the SAME fixed SUE rule, adverse borrow ----
    print("\n" + "=" * 86)
    print("FRESH-SYMBOL HOLDOUT (19 disjoint names) — SUE thr1.0, ADVERSE borrow")
    print("-" * 86)
    ho = run(ho_bars, ho_sue, signal_field="sue", entry_thr=1.0, adverse=ADVERSE)
    results["HOLDOUT_sue_adverse"] = ho
    tr_best = run(tr_bars, tr_sue, signal_field="sue", entry_thr=1.0, adverse=ADVERSE)
    print(f"{'TRAIN-63  both':<28}{tr_best['both']['sharpe']:>7.2f}  OOS {tr_best['both']['oos']:>6.2f}"
          f"  short {tr_best['short']['sharpe']:>6.2f}")
    print(f"{'HOLDOUT-19 both':<28}{ho['both']['sharpe']:>7.2f}  OOS {ho['both']['oos']:>6.2f}"
          f"  short {ho['short']['sharpe']:>6.2f}")

    # verdict
    sue_adv = results["SUE               thr1.0  ADVERSE"]["both"]
    raw_adv = results["RAW surprise_pct  thr2.0  ADVERSE"]["both"]
    sue_beats_raw = sue_adv["sharpe"] > raw_adv["sharpe"] + 0.05
    survives_adverse = sue_adv["sharpe"] > 0.20 and sue_adv["oos"] > 0.0
    generalizes = ho["both"]["sharpe"] > 0.0 and ho["both"]["oos"] > 0.0
    candidate = sue_beats_raw and survives_adverse and generalizes
    verdict = ("CANDIDATE — SUE-PEAD survives adverse borrow AND generalizes to fresh symbols"
               if candidate else
               "REJECT — does not clear survive-adverse-borrow + fresh-symbol-holdout bar")
    print("\n" + "=" * 86)
    print(f"SUE beats raw (adverse): {sue_beats_raw} | survives adverse OOS+: {survives_adverse} "
          f"| generalizes (fresh-19): {generalizes}")
    print(f"VERDICT: {verdict}")

    results["_verdict"] = {"sue_beats_raw": sue_beats_raw, "survives_adverse": survives_adverse,
                           "generalizes": generalizes, "candidate": bool(candidate),
                           "verdict": verdict, "headline_dsr": headline_dsr,
                           "n_train": len(tr_bars), "n_holdout": len(ho_bars),
                           "n_sue_events": n_sue_ev}
    # strip bulky equity/daily arrays before persisting
    for cfg in results.values():
        if isinstance(cfg, dict):
            for leg in cfg.values():
                if isinstance(leg, dict):
                    leg.pop("equity", None)
                    leg.pop("daily", None)
    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\n[meta] wrote {args.out}")


if __name__ == "__main__":
    main()
