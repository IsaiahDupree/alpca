"""
Case 47 — three ORTHOGONAL entry/sizing/risk refinements to the ONE deployed edge (cointegrated-pairs
basket). These are NOT pair-selection changes (Johansen/Hurst/Kalman/ADF-tight = the Case-5 graveyard) —
they are additive, default-OFF flags on top of the validated top-10 + 5% ADF walk-forward:

  (a) cost_cal_entry  — per pair, act_entry_z = max(entry_z, 4*cost_frac/ou_std), ou_std = the OU/AR(1)
                        equilibrium std of the TRAIN spread. Raises/skips the entry threshold on pairs too
                        tight to cover a round-trip cost. (lookahead-free: TRAIN-only OU fit)
  (b) ou_sizing       — leg size = leg_notional_pct * min(|z|/act_entry_z, 1.0). A RESHAPE (max fraction =
                        leg_notional_pct), NOT a leverage bump: biggest at the threshold, smaller deeper in.
  (c) regime_monitor  — rolling ADF p-value on the trailing hedged spread gates each pair: ACTIVE (trade) /
                        WARNING (hold, no new opens) / HALTED (flatten). Risk overlay only (warn 0.10/halt 0.20).

Each refinement is judged as a DELTA vs the reproduced baseline on the IDENTICAL config/universe:
  WF Sharpe · OOS (last-third split) · per-calendar-year stability · maxDD · Sortino ·
  FRESH-SYMBOL HOLDOUT (frozen params on the disjoint mid-cap universe — the gate that killed Cases 18/23) ·
  Deflated Sharpe (multiple-testing penalty).

ACCEPTANCE BAR (strict, anti-overfit): ADOPT only if it beats baseline OOS net-of-cost AND clears the
fresh-symbol holdout AND doesn't degrade per-year stability AND survives DSR. (c) is risk-only -> judged
on maxDD/Sortino at ~equal return, not Sharpe alpha. Default expectation is REJECT (the pairs-improvement
line is a graveyard); confirming neutral-or-worse is a valid result.

Run: .venv/bin/python scripts/test_pairs_refinements.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from alpca.backtest.pairs import walkforward_pairs  # noqa: E402
from alpca.backtest.evaluation import (  # noqa: E402
    sharpe_of, sortino_of, max_drawdown_of, deflated_sharpe_ratio, probabilistic_sharpe_ratio)

PPY = 252.0
DSR_TRIALS = 8          # baseline + 3 single + best combo + headroom for the variants tried this session
DSR_SR_VAR = 1e-4       # project convention (matches test_midcap_*.py)


def _load(c, min_bars=1000):
    """Load cached bars, dropping names with < min_bars history. The drop is NOT cherry-picking:
    walkforward_pairs uses a GLOBAL timestamp intersection, so a single empty/short-history file
    collapses the common calendar to ~0 and silently kills the whole walk-forward (the all-zeros
    artifact). A name with too little history can't be screened on a 252-bar train window anyway."""
    out = {}
    for p in Path(c).glob("*_1day_bars.jsonl"):
        rows = [json.loads(l) for l in p.open() if l.strip()]
        if len(rows) >= min_bars:
            out[p.name.split("_1day_")[0]] = rows
    return out


def _eq_from_returns(rets, start=1.0):
    eq = [start]
    for r in rets:
        eq.append(eq[-1] * (1 + r))
    return eq


def _per_year(dates, rets):
    """Per-calendar-year Sharpe from dated OOS daily returns (regime-stability check)."""
    by = {}
    for t, r in zip(dates, rets):
        by.setdefault(time.gmtime(int(t)).tm_year, []).append(r)
    out = {}
    for y, rs in sorted(by.items()):
        if len(rs) >= 5:
            out[y] = round(sharpe_of(_eq_from_returns(rs), PPY), 2)
    return out


def _oos_split(rets, frac=1 / 3):
    """OOS = Sharpe of the last `frac` of the concatenated OOS curve (held-out tail)."""
    if len(rets) < 30:
        return 0.0
    k = int(len(rets) * (1 - frac))
    return round(sharpe_of(_eq_from_returns(rets[k:]), PPY), 3)


def _run(bars, *, top_n, max_adf, **flags):
    r = walkforward_pairs(bars, train=252, test=63, top_n=top_n, max_half_life=30.0,
                          min_half_life=3.0, entry_z=2.0, exit_z=0.5, cost_bps=2.0,
                          max_adf=max_adf, periods_per_year=PPY, **flags)
    rets, dates = r.daily_returns, r.dates
    eq = r.equity_curve
    py = _per_year(dates, rets)
    return {
        "wf_sharpe": round(r.sharpe, 3),
        "oos_sharpe": _oos_split(rets),
        "sortino": round(sortino_of(eq, PPY), 3),
        "max_drawdown": round(r.max_drawdown, 4),
        "total_return": round(r.total_return, 4),
        "n_windows": r.n_windows,
        "n_days": r.n_oos_bars,
        "per_year": py,
        "per_year_pos": f"{sum(1 for s in py.values() if s > 0)}/{len(py)}",
        "psr": round(probabilistic_sharpe_ratio(eq), 3),
        "dsr": round(deflated_sharpe_ratio(eq, n_trials=DSR_TRIALS, sharpe_variance=DSR_SR_VAR), 3),
        "_rets": rets, "_dates": dates,
    }


VARIANTS = {
    "baseline":            {},
    "(a) cost_cal_entry":  {"cost_cal_entry": True},
    "(b) ou_sizing":       {"ou_sizing": True, "cost_cal_entry": True},   # (b) needs act_entry_z from (a)
    "(c) regime_monitor":  {"regime_monitor": True},
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--largecap", default="/Volumes/My Passport/AlpcaData/cache_largecap_sip")
    ap.add_argument("--fresh", default="/Volumes/My Passport/AlpcaData/cache_midcap_sip",
                    help="disjoint universe for the fresh-symbol holdout (zero overlap with large-caps)")
    ap.add_argument("--top-n", type=int, default=10)
    ap.add_argument("--max-adf", type=float, default=-2.86)
    ap.add_argument("--out", default="data/pairs_refinements.json")
    args = ap.parse_args()

    lc = _load(args.largecap)
    fresh = _load(args.fresh)
    print(f"[load] large-cap (main) {len(lc)} syms · fresh-holdout (mid-cap, disjoint) {len(fresh)} syms\n")

    results = {}
    print(f"{'variant':>22}{'WF':>7}{'OOS':>7}{'maxDD':>8}{'Sortino':>9}{'per-year':>11}{'DSR':>7}")
    print("-" * 71)
    for name, flags in VARIANTS.items():
        r = _run(lc, top_n=args.top_n, max_adf=args.max_adf, **flags)
        results[name] = r
        print(f"{name:>22}{r['wf_sharpe']:>7.2f}{r['oos_sharpe']:>7.2f}{r['max_drawdown']*100:>7.1f}%"
              f"{r['sortino']:>9.2f}{r['per_year_pos']:>11}{r['dsr']:>7.2f}")

    # best surviving single-refinement combo: only combine the ones that did not degrade WF & maxDD vs baseline
    base = results["baseline"]
    survivors = []
    for nm in ["(a) cost_cal_entry", "(b) ou_sizing", "(c) regime_monitor"]:
        r = results[nm]
        # a refinement "survives to combine" if it is non-degrading on its own primary metric
        if nm == "(c) regime_monitor":
            ok = r["max_drawdown"] >= base["max_drawdown"] - 1e-6  # risk-only: must not worsen maxDD
        else:
            ok = r["wf_sharpe"] >= base["wf_sharpe"] - 0.02 and r["oos_sharpe"] >= base["oos_sharpe"] - 0.02
        if ok:
            survivors.append(nm)
    combo_flags = {}
    if "(a) cost_cal_entry" in survivors:
        combo_flags["cost_cal_entry"] = True
    if "(b) ou_sizing" in survivors:
        combo_flags["ou_sizing"] = True
        combo_flags["cost_cal_entry"] = True
    if "(c) regime_monitor" in survivors:
        combo_flags["regime_monitor"] = True
    combo = None
    if combo_flags and len(survivors) >= 1:
        combo = _run(lc, top_n=args.top_n, max_adf=args.max_adf, **combo_flags)
        results["best_combo"] = combo
        print(f"{'best combo '+str(survivors):>22}"[:22].rjust(22), end="")
        print(f"{combo['wf_sharpe']:>7.2f}{combo['oos_sharpe']:>7.2f}{combo['max_drawdown']*100:>7.1f}%"
              f"{combo['sortino']:>9.2f}{combo['per_year_pos']:>11}{combo['dsr']:>7.2f}")
    else:
        print("\n[combo] no single refinement survived its own non-degradation check -> no combo run")

    # ---- FRESH-SYMBOL HOLDOUT (frozen params, disjoint mid-cap universe) ----
    print(f"\n=== FRESH-SYMBOL HOLDOUT (disjoint mid-cap universe, {len(fresh)} syms, frozen params) ===")
    print(f"{'variant':>22}{'WF':>7}{'OOS':>7}{'maxDD':>8}{'per-year':>11}")
    print("-" * 55)
    holdout = {}
    for name, flags in VARIANTS.items():
        r = _run(fresh, top_n=args.top_n, max_adf=args.max_adf, **flags)
        holdout[name] = {k: v for k, v in r.items() if not k.startswith("_")}
        print(f"{name:>22}{r['wf_sharpe']:>7.2f}{r['oos_sharpe']:>7.2f}{r['max_drawdown']*100:>7.1f}%"
              f"{r['per_year_pos']:>11}")
    if combo is not None:
        rc = _run(fresh, top_n=args.top_n, max_adf=args.max_adf, **combo_flags)
        holdout["best_combo"] = {k: v for k, v in rc.items() if not k.startswith("_")}
        print(f"{'best_combo':>22}{rc['wf_sharpe']:>7.2f}{rc['oos_sharpe']:>7.2f}{rc['max_drawdown']*100:>7.1f}%"
              f"{rc['per_year_pos']:>11}")

    # ---- VERDICTS ----
    print(f"\n=== VERDICTS (vs baseline WF {base['wf_sharpe']:.2f} / OOS {base['oos_sharpe']:.2f} / "
          f"maxDD {base['max_drawdown']*100:.1f}% / DSR {base['dsr']:.2f}) ===")
    verdicts = {}
    for name in ["(a) cost_cal_entry", "(b) ou_sizing", "(c) regime_monitor"]:
        r = results[name]
        h = holdout[name]
        if name == "(c) regime_monitor":
            # risk-only: judge on maxDD reduction / Sortino at ~equal return, NOT Sharpe alpha
            dd_better = r["max_drawdown"] > base["max_drawdown"] + 1e-6   # less negative = smaller DD
            sortino_better = r["sortino"] >= base["sortino"] - 0.02
            ret_ok = r["total_return"] >= base["total_return"] - 0.02
            adopt = dd_better and sortino_better and ret_ok and h["max_drawdown"] >= -0.5
            why = (f"maxDD {base['max_drawdown']*100:.1f}%->{r['max_drawdown']*100:.1f}% "
                   f"(better={dd_better}), Sortino {base['sortino']:.2f}->{r['sortino']:.2f}, "
                   f"ret {base['total_return']*100:.1f}%->{r['total_return']*100:.1f}%, "
                   f"holdout maxDD {h['max_drawdown']*100:.1f}%")
        else:
            beats_oos = r["oos_sharpe"] > base["oos_sharpe"] + 1e-6
            clears_holdout = h["wf_sharpe"] > 0 and h["oos_sharpe"] > 0
            stable = sum(1 for s in r["per_year"].values() if s > 0) >= sum(
                1 for s in base["per_year"].values() if s > 0) - 1
            survives_dsr = r["dsr"] >= base["dsr"] - 0.02
            adopt = beats_oos and clears_holdout and stable and survives_dsr
            why = (f"beats_OOS={beats_oos} ({base['oos_sharpe']:.2f}->{r['oos_sharpe']:.2f}), "
                   f"fresh_holdout WF {h['wf_sharpe']:.2f}/OOS {h['oos_sharpe']:.2f} (clears={clears_holdout}), "
                   f"per-year {r['per_year_pos']} (stable={stable}), DSR {r['dsr']:.2f} (survives={survives_dsr})")
        verdicts[name] = {"adopt": adopt, "why": why}
        print(f"  {name}: {'ADOPT' if adopt else 'REJECT'} — {why}")

    out = {
        "config": {"top_n": args.top_n, "max_adf": args.max_adf, "train": 252, "test": 63,
                   "cost_bps": 2.0, "entry_z": 2.0, "exit_z": 0.5, "dsr_trials": DSR_TRIALS},
        "n_largecap": len(lc), "n_fresh": len(fresh),
        "main": {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")} for k, v in results.items()},
        "fresh_holdout": holdout,
        "verdicts": verdicts,
        "combo_survivors": survivors,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\n[saved] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
