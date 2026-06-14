"""
Case 48 — CROSS-SECTIONAL calendar seasonality (Heston-Sadka same-month) as a second-leg candidate.

The binding constraint (Case 47): find a market-neutral leg that is POSITIVE over the forward-relevant
window (2022→) AND uncorrelated with the pairs basket. Cross-sectional seasonality fits structurally —
rank each name by its OWN return in the SAME calendar month in PRIOR years (strict no-lookahead), long
the historically-strong-this-month names, short the weak. The P&L is on a calendar clock, orthogonal to
trend (momentum) and pairwise mean-reversion (pairs). It uses zero new data, and because it needs
prior-year history it only starts trading in year 2 — i.e. it is naturally evaluated on 2022–2026.

Same bar as every candidate: full universe + disjoint fresh-symbol holdout + per-year regime + 2bps cost
+ DSR. Run on large-cap and mid-cap universes. A pass must (a) be positive on the full universe, (b) keep
a positive fresh-symbol holdout, AND (c) be positive over 2022→ (its whole live window).

Run: .venv/bin/python scripts/test_seasonality_xs.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from alpca.backtest.factor import backtest_factor, cross_sectional_seasonality_signal  # noqa: E402
from alpca.backtest.evaluation import deflated_sharpe_ratio, sharpe_of  # noqa: E402

PPY = 252.0


def _load(c):
    return {p.name.split("_1day_")[0]: [json.loads(l) for l in p.open() if l.strip()]
            for p in Path(c).glob("*_1day_bars.jsonl")}


def _eq(daily, s=1.0):
    e = [s]
    for x in daily:
        e.append(e[-1] * (1 + x))
    return e


def _run(bars, label):
    r = backtest_factor(bars, cross_sectional_seasonality_signal(min_prior=15), name="seasonality",
                        top_frac=0.2, rebalance_days=21, cost_bps=2.0, long_high=True, periods_per_year=PPY)
    eq = r.equity_curve
    if len(eq) < 60:
        return None
    by = {}
    for ep, x in zip(r.dates, r.daily_returns):
        by.setdefault(time.gmtime(ep).tm_year, []).append(x)
    yr = {y: round(sharpe_of(_eq(v), PPY), 2) for y, v in by.items() if len(v) >= 30}
    # 2022-> window only (its live period)
    fwd = [x for ep, x in zip(r.dates, r.daily_returns) if time.gmtime(ep).tm_year >= 2022]
    fwd_sh = sharpe_of(_eq(fwd), PPY) if len(fwd) > 30 else 0.0
    return {"label": label, "sharpe": r.sharpe, "per_year": yr, "fwd2022_sharpe": fwd_sh,
            "turnover": r.avg_turnover, "n_days": r.n_days, "daily": r.daily_returns, "dates": r.dates}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--universes", nargs="+", default=[
        "/Volumes/My Passport/AlpcaData/cache_largecap_sip",
        "/Volumes/My Passport/AlpcaData/cache_midcap_sip"])
    ap.add_argument("--n-trials", type=int, default=92)
    ap.add_argument("--out", default="data/seasonality_xs_results.json")
    args = ap.parse_args()

    out = {}
    print(f"{'universe':>22}{'slice':>10}{'sharpe':>8}{'2022+':>8}{'+yrs':>7}{'turn':>7}{'DSR':>6}")
    print("-" * 68)
    for cache in args.universes:
        bars = _load(cache)
        if len(bars) < 20:
            print(f"{Path(cache).name:>22}  (too few symbols)"); continue
        syms = sorted(bars)
        train = {s: bars[s] for i, s in enumerate(syms) if i % 3 != 0}
        hold = {s: bars[s] for i, s in enumerate(syms) if i % 3 == 0}
        uname = Path(cache).name.replace("cache_", "").replace("_sip", "")
        full = _run(bars, f"{uname}|full")
        h = _run(hold, f"{uname}|holdout")
        if not full:
            continue
        dsr = deflated_sharpe_ratio(_eq(full["daily"]), n_trials=args.n_trials, sharpe_variance=1e-4)
        pos = sum(1 for s in full["per_year"].values() if s > 0)
        hold_sh = h["sharpe"] if h else float("nan")
        out[uname] = {"full_sharpe": round(full["sharpe"], 3), "fwd2022_sharpe": round(full["fwd2022_sharpe"], 3),
                      "holdout_sharpe": round(hold_sh, 3), "per_year": full["per_year"],
                      "turnover": round(full["turnover"], 3), "dsr": round(dsr, 3)}
        print(f"{uname:>22}{'full':>10}{full['sharpe']:>8.2f}{full['fwd2022_sharpe']:>8.2f}"
              f"{pos:>4}/{len(full['per_year']):<2}{full['turnover']:>7.3f}{dsr:>6.2f}")
        if h:
            hp = sum(1 for s in h["per_year"].values() if s > 0)
            print(f"{uname:>22}{'holdout':>10}{h['sharpe']:>8.2f}{h['fwd2022_sharpe']:>8.2f}"
                  f"{hp:>4}/{len(h['per_year']):<2}")
        generalizes = (not (hold_sh != hold_sh)) and hold_sh > 0
        fwd_pos = full["fwd2022_sharpe"] > 0
        verdict = ("CANDIDATE — positive full + holdout + 2022→" if (full["sharpe"] > 0.15 and generalizes and fwd_pos)
                   else f"reject (full>{0.15}={full['sharpe']>0.15}, generalizes={generalizes}, 2022+={fwd_pos})")
        out[uname]["verdict"] = verdict
        print(f"{'':>22}-> {verdict}\n")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"[saved] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
