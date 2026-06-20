"""
PRIOR-CONFIRMATION RUN — do the published formulaic-alpha LIBRARIES (Kakushadze Alpha101, with the
overlapping Alpha158/Qlib158 and GTJA191 families) produce ANY surviving cross-sectional equity edge on
our venue, or do they collapse like value / momentum / seasonality did (base rate ~1/51 survive)?

Tests a DE-COLLINEARIZED representative subset of Alpha101 (~22 formulas, one per distinct family — see
alpca/backtest/alpha101.py header for the Alpha158/GTJA191 duplicates each subsumes) through the SAME
honest bar the rest of the zoo faced: main universe + a DISJOINT fresh-symbol holdout (cache_fresh, a
universe never seen in selection) + 70/30 OOS split + per-year regime stability + realistic 2bps cost +
Deflated Sharpe + the deterministic falsification gate.

MULTIPLE-TESTING: testing 20+ factors inflates false positives. We apply it two ways:
  (1) DSR with n_trials = (#factors tested)  -> the Bailey/López-de-Prado deflation IS the Bonferroni
      analog for Sharpe; the gate already requires DSR >= 0.9 against that inflated benchmark.
  (2) An explicit Bonferroni flag: PSR(vs SR0=0) >= 1 - 0.05/K (familywise alpha 5% across K factors).
A survivor must clear the falsification gate (which includes fresh-symbol + DSR) AND the Bonferroni flag.

SELECT-IS, REPORT-OOS. No look-ahead (factor known as of day t, returns t->t+1). Gross AND net (2bps)
reported — short-horizon reversal factors look great gross and die to the cost wall.

Run: .venv/bin/python scripts/test_alpha101_libs.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from alpca.backtest.factor import backtest_factor  # noqa: E402
from alpca.backtest import alpha101 as a  # noqa: E402
from alpca.backtest.evaluation import (  # noqa: E402
    deflated_sharpe_ratio, probabilistic_sharpe_ratio, sharpe_of)
from alpca.ai.strategy_gate import falsification_gate  # noqa: E402

PPY = 252.0

# name -> (builder(bars), long_high, family). long_high encodes the published alpha SIGN.
FACTORS = {
    # A: rank-reversal
    "A_reversal_ret5":    (lambda b: a.alpha101_reversal_returns(b, 5), False, "A_reversal"),
    "A_a3_corr_open_vol": (lambda b: a.alpha101_a3_neg_corr_open_vol(b, 10), True, "A_reversal"),
    "A_a4_tsrank_low":    (lambda b: a.alpha101_a4(b, 9),               True,  "A_reversal"),
    "A_a9_cond_mom":      (lambda b: a.alpha101_a9(b),                  True,  "A_reversal"),
    # B: volume-price correlation/covariance
    "B_a2_dvol_dclose":   (lambda b: a.alpha101_a2(b, 6),               True,  "B_volprice"),
    "B_a6_corr_open_vol": (lambda b: a.alpha101_a6(b, 10),              True,  "B_volprice"),
    "B_a12_signdvol":     (lambda b: a.alpha101_a12(b),                 True,  "B_volprice"),
    "B_a13_cov_close_vol":(lambda b: a.alpha101_a13(b, 5),              True,  "B_volprice"),
    "B_a16_cov_high_vol": (lambda b: a.alpha101_a16(b, 5),              True,  "B_volprice"),
    "B_vwap_close_dev":   (lambda b: a.alpha101_vwap_close_dev(b, 5),   False, "B_volprice"),
    # C: decay-linear / weighted momentum
    "C_decay_mom10":      (lambda b: a.alpha101_decay_mom(b, 10),       True,  "C_decay"),
    "C_a8_open_ret_lag":  (lambda b: a.alpha101_a8(b, 5),               True,  "C_decay"),
    # D: high/low/close microstructure
    "D_a53_range_pos":    (lambda b: a.alpha101_a53(b, 9),              True,  "D_micro"),
    "D_a54_lowclose":     (lambda b: a.alpha101_a54(b),                 True,  "D_micro"),
    "D_kbar_close_pos":   (lambda b: a.alpha101_kbar_close_pos(b),      False, "D_micro"),
    "D_a101_intraday":    (lambda b: a.alpha101_a101(b),                True,  "D_micro"),
    # E: returns-based vol/skew
    "E_a20_gap_reversal": (lambda b: a.alpha101_a20(b),                 True,  "E_volskew"),
    "E_a22_corr_std":     (lambda b: a.alpha101_a22(b, 5, 20),          True,  "E_volskew"),
    "E_a40_std_corr":     (lambda b: a.alpha101_a40(b, 10, 10),         True,  "E_volskew"),
    "E_ret_std20":        (lambda b: a.alpha101_ret_std(b, 20),         False, "E_volskew"),
    "E_signedpow_ret5":   (lambda b: a.alpha101_signedpower_ret(b, 5, 2.0), True, "E_volskew"),
}


def _load(c):
    return {p.name.split("_1day_")[0]: [json.loads(l) for l in p.open() if l.strip()]
            for p in Path(c).glob("*_1day_bars.jsonl")}


def _eq(daily, s=1.0):
    e = [s]
    for x in daily:
        e.append(e[-1] * (1 + x))
    return e


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/Volumes/My Passport/AlpcaData/cache")
    ap.add_argument("--cache-fresh", default="/Volumes/My Passport/AlpcaData/cache_fresh")
    ap.add_argument("--top-frac", type=float, default=0.2)
    ap.add_argument("--rebalance-days", type=int, default=21)
    ap.add_argument("--out", default="data/alpha101_libs_results.json")
    args = ap.parse_args()

    main_b = _load(args.cache)
    fresh_b = _load(args.cache_fresh)
    main_b.pop("SPY", None); fresh_b.pop("SPY", None)
    K = len(FACTORS)
    bonf_thresh = 1.0 - 0.05 / K          # familywise alpha 5% across K factors
    print(f"[ok] main universe {len(main_b)} syms · fresh holdout {len(fresh_b)} syms (disjoint)")
    print(f"[multiple-testing] K={K} factors · DSR n_trials={K} · Bonferroni PSR>= {bonf_thresh:.4f}\n")

    hdr = f"{'factor':>22}{'fam':>12}{'gross':>7}{'net':>7}{'OOS':>7}{'fresh':>7}{'+yrs':>7}{'DSR':>6}{'bonf':>6}{'GATE':>6}"
    print(hdr); print("-" * len(hdr))
    rows = []
    survivors = []
    for name, (builder, long_high, fam) in FACTORS.items():
        # NET (2bps) on main universe — the deployable number
        r = backtest_factor(main_b, builder(main_b), name=name, top_frac=args.top_frac,
                            rebalance_days=args.rebalance_days, cost_bps=2.0, long_high=long_high,
                            periods_per_year=PPY)
        eq = r.equity_curve
        if len(eq) < 60:
            print(f"{name:>22}  (insufficient)"); continue
        # GROSS (0bps) — to expose cost-wall deaths
        rg = backtest_factor(main_b, builder(main_b), name=name, top_frac=args.top_frac,
                            rebalance_days=args.rebalance_days, cost_bps=0.0, long_high=long_high,
                            periods_per_year=PPY)
        gross = rg.sharpe
        sp = int(len(eq) * 0.7)
        oos = sharpe_of(eq[sp:], PPY)
        by = {}
        for ep, x in zip(r.dates, r.daily_returns):
            by.setdefault(time.gmtime(ep).tm_year, []).append(x)
        yr = {y: round(sharpe_of(_eq(v), PPY), 2) for y, v in by.items() if len(v) >= 30}
        pos = sum(1 for s in yr.values() if s > 0)
        # FRESH-SYMBOL HOLDOUT — frozen params, disjoint universe (the real gate)
        rf = backtest_factor(fresh_b, builder(fresh_b), name=name, top_frac=args.top_frac,
                            rebalance_days=args.rebalance_days, cost_bps=2.0, long_high=long_high,
                            periods_per_year=PPY)
        # DSR deflated by K trials (Bonferroni analog for Sharpe)
        dsr = deflated_sharpe_ratio(eq, n_trials=K, sharpe_variance=1e-4)
        # explicit Bonferroni: PSR of net equity vs SR0=0, must clear 1 - 0.05/K
        psr0 = probabilistic_sharpe_ratio(eq, sr_benchmark=0.0)
        bonf_pass = psr0 >= bonf_thresh

        result = {"name": name, "family": fam, "long_high": long_high,
                  "gross_sharpe": round(gross, 3), "net_sharpe": round(r.sharpe, 3),
                  "oos_sharpe": round(oos, 3), "fresh_holdout_sharpe": round(rf.sharpe, 3),
                  "per_year": yr, "pos_years": f"{pos}/{len(yr)}",
                  "dsr": round(dsr, 3), "psr_vs_zero": round(psr0, 4),
                  "bonferroni_pass": bool(bonf_pass), "turnover": round(r.avg_turnover, 3),
                  "cost_2bps_sharpe": round(r.sharpe, 3)}
        g = falsification_gate(result)
        gate_and_bonf = g.passed and bonf_pass
        result["rail_pass"] = g.passed
        result["rail_reasons"] = g.reasons
        result["validated"] = bool(gate_and_bonf)   # gate already requires fresh_holdout > min_fresh
        rows.append(result)
        if result["validated"]:
            survivors.append(name)
        print(f"{name:>22}{fam:>12}{gross:>7.2f}{r.sharpe:>7.2f}{oos:>7.2f}{rf.sharpe:>7.2f}"
              f"{pos:>4}/{len(yr):<2}{dsr:>6.2f}{('Y' if bonf_pass else '.'):>6}"
              f"{('PASS' if gate_and_bonf else '.'):>6}")

    print("\n" + "=" * 60)
    print(f"[base rate context] Alpca zoo survival ~1/51")
    print(f"[tested] {len(rows)} factors across {len({r['family'] for r in rows})} families")
    print(f"[fresh-symbol + DSR + Bonferroni survivors] {survivors or 'NONE'}")
    verdict = "WEAK/SURVIVOR" if survivors else "DEBUNKED — collapses like value/momentum/seasonality"
    print(f"[VERDICT] {verdict}")

    out = {"k_factors": K, "bonferroni_threshold": round(bonf_thresh, 4),
           "universe_main": len(main_b), "universe_fresh": len(fresh_b),
           "results": rows, "survivors": survivors, "verdict": verdict}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"[saved] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
