"""
THE combination test — assemble our real, genuinely-different strategy return streams and
blend them with the inverse-vol + half-Kelly combiner, to answer honestly: does combining
uncorrelated legs lift the risk-adjusted return, and what daily ROI is actually achievable?

Legs (all daily, same equity calendar -> tail-alignment is valid):
  - beta sleeve        : rsi-mr on SPY (risk-reduced long beta, deployed live)
  - market-neutral     : cross-sectional momentum L/S over the universe (~0 market beta)
  - seasonality (TOM)  : turn-of-month overlay on SPY (event-clock, ~0 correlation)
  - seasonality (FOMC) : pre-FOMC drift overlay on SPY (event-clock)

Prints the cross-leg CORRELATION MATRIX (the metric the whole thing lives or dies on), the
inverse-vol combined Sharpe vs the equal-weight NULL, and the honest Sharpe->daily-return
translation (why 'X% per day' is noise).

Run: .venv/bin/python scripts/test_combine.py --cache "/Volumes/My Passport/AlpcaData/cache"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from alpca.backtest.combine import combined_sharpe_formula, evaluate_combo  # noqa: E402
from alpca.backtest.cross_sectional import backtest_cross_sectional_momentum  # noqa: E402
from alpca.backtest.pairs import backtest_pairs, screen_pairs  # noqa: E402
from alpca.backtest.runner_backtest import backtest_resting  # noqa: E402
from alpca.backtest.seasonality import (  # noqa: E402
    backtest_seasonal, pre_fomc_position, turn_of_month_position)
from alpca.strategies.registry import make  # noqa: E402

PPY = 252.0


def rets_from_equity(eq):
    return [(eq[i] - eq[i - 1]) / eq[i - 1] for i in range(1, len(eq)) if eq[i - 1] > 0]


def pairs_basket_returns(universe):
    """The real surviving edge: equal-weight basket of the top cointegrated pairs' daily
    returns (the market-neutral leg we actually trust)."""
    screened = screen_pairs(list(universe), universe, min_overlap=250, max_half_life=40, min_half_life=3)
    curves = []
    for r in screened[:8]:
        res = backtest_pairs(universe[r["a"]], universe[r["b"]], sym_a=r["a"], sym_b=r["b"],
                             lookback=int(max(20, min(120, r["half_life"] * 3))),
                             entry_z=2.0, exit_z=0.5, cost_bps=2.0)
        rr = rets_from_equity(res.equity_curve)
        if rr:
            curves.append(rr)
    if not curves:
        return []
    m = min(len(c) for c in curves)
    return [sum(c[-m:][t] for c in curves) / len(curves) for t in range(m)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/Volumes/My Passport/AlpcaData/cache")
    ap.add_argument("--target-vol", type=float, default=0.08)
    ap.add_argument("--out", default="data/combine_results.json")
    args = ap.parse_args()
    cache = Path(args.cache)

    def load(sym):
        p = cache / f"{sym}_1day_bars.jsonl"
        return [json.loads(l) for l in p.open() if l.strip()] if p.exists() else []

    spy = load("SPY")
    # market-neutral leg over a sector-diverse universe (broad enough to screen pairs)
    uni_syms = sorted(p.name.split("_1day_")[0] for p in cache.glob("*_1day_bars.jsonl"))[:80]
    universe = {s: load(s) for s in uni_syms if load(s)}

    print("Building real strategy return streams...")
    streams = {}
    rsi = backtest_resting(make("rsi-mr"), spy)
    streams["rsi-mr (beta)"] = rets_from_equity(rsi.equity_curve)
    basket = pairs_basket_returns(universe)
    if basket:
        streams["pairs-basket (MN)"] = basket      # the real OOS-surviving edge
    cs = backtest_cross_sectional_momentum(universe, lookback=250, hold=20, top_k=5, bottom_k=5,
                                           cost_bps=2.0, periods_per_year=PPY, market_neutral=True)
    streams["x-sectional (MN)"] = rets_from_equity(cs.equity_curve)
    tom = backtest_seasonal(spy, turn_of_month_position(spy), name="tom", periods_per_year=PPY)
    streams["turn-of-month"] = rets_from_equity(tom.equity_curve)
    fomc = backtest_seasonal(spy, pre_fomc_position(spy), name="fomc", periods_per_year=PPY)
    streams["pre-FOMC"] = rets_from_equity(fomc.equity_curve)

    rep = evaluate_combo(streams, ppy=PPY, target_vol=args.target_vol)

    print("\n===== PER-LEG (annualized) =====")
    for k, v in rep.legs.items():
        print(f"  {k:<20} Sharpe {v['ann_sharpe']:>5.2f}   daily-vol {v['vol']*100:.3f}%")

    print("\n===== CROSS-LEG CORRELATION MATRIX (lives/dies on this) =====")
    print("  " + "".join(f"{n[:10]:>12}" for n in rep.corr_names))
    for i, n in enumerate(rep.corr_names):
        print(f"  {n[:10]:<10}" + "".join(f"{rep.corr_matrix[i][j]:>12.2f}" for j in range(len(rep.corr_names))))
    print(f"  avg |off-diagonal correlation| = {rep.avg_abs_corr:.2f}  "
          f"({'GOOD: low, diversification works' if rep.avg_abs_corr < 0.3 else 'HIGH: legs overlap, little benefit'})")

    print("\n===== COMBINED (inverse-vol + half-Kelly) vs NULL =====")
    print(f"  equal-weight blend Sharpe (NULL): {rep.equalweight_sharpe:.2f}")
    print(f"  inverse-vol blend  Sharpe       : {rep.invvol_sharpe:.2f}  "
          f"({'beats null' if rep.invvol_sharpe > rep.equalweight_sharpe else 'ties/loses null'})")
    print(f"  weights: " + ", ".join(f"{k}={w:.2f}" for k, w in rep.invvol_weights.items()))
    best_leg = max(rep.legs.values(), key=lambda v: v["ann_sharpe"])["ann_sharpe"]
    print(f"  best single leg Sharpe: {best_leg:.2f}  ->  combined {rep.invvol_sharpe:.2f} "
          f"({'diversification LIFT' if rep.invvol_sharpe > best_leg else 'no lift over best leg'})")
    k = len(streams)
    print(f"  theory check: {k} legs @ Sharpe {best_leg:.2f}, rho {rep.avg_abs_corr:.2f} -> "
          f"formula predicts {combined_sharpe_formula(best_leg, k, rep.avg_abs_corr):.2f}")

    t = rep.translation
    print("\n===== HONEST DAILY-ROI TRANSLATION (target vol "
          f"{t['ann_vol']*100:.0f}%) =====")
    print(f"  combined annual Sharpe {t['sharpe_annual']:.2f}  ->  expected EXCESS return "
          f"{t['expected_excess_annual']*100:.1f}%/yr (~{t['expected_total_annual']*100:.1f}% total)")
    print(f"  expected DAILY return  {t['expected_daily_excess']*100:.3f}%  "
          f"(~{t['expected_daily_excess']*10000:.1f} bps/day)")
    print(f"  daily VOL (noise)      {t['daily_vol']*100:.2f}%  -> noise is {t['noise_to_edge_ratio']:.0f}x the daily edge")
    print(f"  => the edge is INVISIBLE day-to-day; it only exists over hundreds of days. "
          f"'X% per day' targets are noise-mining.")

    Path(args.out).write_text(json.dumps({
        "legs": rep.legs, "corr_names": rep.corr_names, "corr_matrix": rep.corr_matrix,
        "avg_abs_corr": rep.avg_abs_corr, "equalweight_sharpe": rep.equalweight_sharpe,
        "invvol_sharpe": rep.invvol_sharpe, "invvol_weights": rep.invvol_weights,
        "translation": rep.translation, "n_days": rep.n_days}, indent=2))
    print(f"\n[done] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
