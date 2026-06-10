"""
Harness-test the Avellaneda-Lee PCA residual stat-arb (alpca/backtest/stat_arb_pca.py)
against the HONEST NULL: our existing cointegration-pairs basket (OOS Sharpe ~0.54).

Market-neutral -> no buy-and-hold to beat; the return IS the alpha. The bar to clear:
beat 0.54 OUT-OF-SAMPLE, net of cost, statistically significant and regime-stable.
To avoid grid-overfit, configs are SELECTED on the in-sample 70% and the winner is
reported on the held-out last 30% (the number that counts).

Run: .venv/bin/python scripts/test_pca_statarb.py --cache "/Volumes/My Passport/AlpcaData/cache"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from alpca.backtest.evaluation import (  # noqa: E402
    max_drawdown_of, segment_sharpes, sharpe_of, sharpe_pvalue, sharpe_tstat)
from alpca.backtest.stat_arb_pca import backtest_pca_statarb  # noqa: E402

PPY = 252.0
PAIRS_BASKET_OOS = 0.54  # the null: our one proven market-neutral edge


def split_sharpes(eq, frac=0.3):
    n = len(eq)
    split = int(n * (1 - frac))
    return sharpe_of(eq[:split], PPY), sharpe_of(eq[split:], PPY)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/Volumes/My Passport/AlpcaData/cache")
    ap.add_argument("--cost-bps", type=float, default=2.0)
    ap.add_argument("--out", default="data/pca_statarb_results.json")
    args = ap.parse_args()
    cache = Path(args.cache)

    bars_by = {}
    for p in sorted(cache.glob("*_1day_bars.jsonl")):
        sym = p.name.split("_1day_")[0]
        rows = [json.loads(l) for l in p.open() if l.strip()]
        if len(rows) > 400:
            bars_by[sym] = rows
    print(f"[ok] universe = {len(bars_by)} symbols, cost {args.cost_bps:g}bps/leg\n")

    # honest grid; SELECT on in-sample, REPORT on OOS
    grid = []
    for lb in (60, 90):
        for nf in (10, 15):
            for so, sc in ((1.25, 0.50), (1.5, 0.75), (2.0, 0.5)):
                grid.append(dict(lookback=lb, n_factors=nf, s_open=so, s_close=sc))

    print(f"{'lookback':>8}{'factors':>8}{'s_open':>7}{'s_close':>8}"
          f"{'fullSh':>8}{'IS_Sh':>7}{'OOS_Sh':>7}{'maxDD':>8}{'gross':>7}{'turn/d':>8}{'sig':>5}")
    print("-" * 92)
    rows = []
    for g in grid:
        r = backtest_pca_statarb(bars_by, cost_bps=args.cost_bps, max_half_life=30.0, **g)
        if r.n_days < 100:
            continue
        is_sh, oos_sh = split_sharpes(r.equity_curve)
        tstat = sharpe_tstat(r.equity_curve)
        pval = sharpe_pvalue(r.equity_curve)
        segs = segment_sharpes(r.equity_curve, PPY, 4)
        sig = pval < 0.05 and abs(tstat) > 2.0
        stable = sum(1 for s in segs if s > 0) * 2 >= len(segs)
        rows.append({**g, "full_sharpe": r.sharpe, "is_sharpe": is_sh, "oos_sharpe": oos_sh,
                     "maxdd": r.max_drawdown, "ret": r.total_return, "tstat": tstat, "pval": pval,
                     "segs": [round(x, 2) for x in segs], "sig": sig, "stable": stable,
                     "avg_gross": r.avg_gross_names, "turnover": r.avg_daily_turnover})
        print(f"{g['lookback']:>8}{g['n_factors']:>8}{g['s_open']:>7.2f}{g['s_close']:>8.2f}"
              f"{r.sharpe:>8.2f}{is_sh:>7.2f}{oos_sh:>7.2f}{r.max_drawdown*100:>7.1f}%"
              f"{r.avg_gross_names:>7.0f}{r.avg_daily_turnover:>8.3f}{('Y' if sig else ''):>5}")

    if not rows:
        print("[fail] no valid configs")
        return 1

    # SELECT on in-sample Sharpe, REPORT the winner's OOS (honest anti-overfit)
    winner = max(rows, key=lambda x: x["is_sharpe"])
    print("\n" + "=" * 92)
    print(f"SELECTED on in-sample: lookback={winner['lookback']} factors={winner['n_factors']} "
          f"s_open={winner['s_open']} s_close={winner['s_close']}")
    print(f"  IN-SAMPLE Sharpe {winner['is_sharpe']:.2f}  ->  OUT-OF-SAMPLE Sharpe {winner['oos_sharpe']:.2f}  "
          f"(full {winner['full_sharpe']:.2f}, maxDD {winner['maxdd']*100:.1f}%)")
    print(f"  significant={winner['sig']}  stable={winner['stable']}  segs={winner['segs']}  "
          f"avg_gross_names={winner['avg_gross']:.0f}  turnover/day={winner['turnover']:.3f}")
    print("-" * 92)
    print(f"NULL (cointegration-pairs basket) OOS Sharpe: {PAIRS_BASKET_OOS}")
    verdict = ("BEATS the pairs-basket null OOS — a genuine extension of our edge"
               if winner["oos_sharpe"] > PAIRS_BASKET_OOS + 0.05
               else "TIES the null (no improvement) — not worth the added complexity"
               if winner["oos_sharpe"] > PAIRS_BASKET_OOS - 0.15
               else "UNDERPERFORMS the null OOS — rejected")
    print(f"VERDICT: PCA residual stat-arb {verdict}.")

    Path(args.out).write_text(json.dumps({
        "universe": len(bars_by), "cost_bps": args.cost_bps, "null_pairs_oos": PAIRS_BASKET_OOS,
        "grid": rows, "selected": winner, "verdict": verdict}, indent=2))
    print(f"\n[done] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
