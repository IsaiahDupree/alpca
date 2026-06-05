"""
HOURLY-crypto edge discovery — the "more bars / more walk-forward windows" angle.

Crypto trades 24/7, so 3 years of HOURLY bars is ~26k bars per coin (vs ~750 daily).
If crypto has an intraday market-neutral edge, the extra resolution + many more
walk-forward windows are where it would show. Everything is judged by the SAME honest
harness as equities, with CRYPTO-REALISTIC settings:
  - periods_per_year = 365*24 = 8760 (hourly, 24/7)
  - cost 10 bps per leg (crypto spreads/fees are wider than equities' ~2 bps)
  - walk-forward windows measured in HOURS (train ~90d, test ~30d, re-screen each step)

Market-neutral pairs and cross-sectional L/S have NO buy-and-hold to beat — the return
itself is the alpha. Walk-forward Sharpe is THE honest number (every trade on unseen data).

Run:
  .venv/bin/python scripts/discover_crypto_hourly.py --cache "/Volumes/My Passport/AlpcaData/crypto_hourly"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from alpca.backtest.cross_sectional import backtest_cross_sectional_momentum  # noqa: E402
from alpca.backtest.evaluation import max_drawdown_of, sharpe_of  # noqa: E402
from alpca.backtest.pairs import backtest_pairs, screen_pairs, walkforward_pairs  # noqa: E402

PPY = 365 * 24  # hourly, 24/7
H = 24          # hours per day, for readable window sizing


def _basket(results, starting=100_000.0, ppy=PPY):
    rets = []
    for r in results:
        eq = r.equity_curve
        rr = [(eq[i] - eq[i - 1]) / eq[i - 1] for i in range(1, len(eq)) if eq[i - 1] > 0]
        if rr:
            rets.append(rr)
    if not rets:
        return 0.0, 0.0, 0.0
    m = min(len(x) for x in rets)
    basket = [sum(x[-m:][t] for x in rets) / len(rets) for t in range(m)]
    eq = [starting]
    for r in basket:
        eq.append(eq[-1] * (1 + r))
    return (eq[-1] - starting) / starting, sharpe_of(eq, ppy), max_drawdown_of(eq)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/Volumes/My Passport/AlpcaData/crypto_hourly")
    ap.add_argument("--timeframe", default="1hour")
    ap.add_argument("--cost-bps", type=float, default=10.0, help="per-leg cost (crypto realistic)")
    ap.add_argument("--max-half-life", type=float, default=72.0, help="hours (3 days)")
    ap.add_argument("--min-half-life", type=float, default=6.0, help="hours")
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--train-days", type=int, default=90)
    ap.add_argument("--test-days", type=int, default=30)
    ap.add_argument("--out", default="data/crypto_hourly_discovery.json")
    args = ap.parse_args()

    cache = Path(args.cache)
    bars = {}
    for p in sorted(cache.glob(f"*_{args.timeframe}_bars.jsonl")):
        sym = p.name.split(f"_{args.timeframe}_")[0]
        rows = [json.loads(l) for l in p.open() if l.strip()]
        if len(rows) > 2000:
            bars[sym] = rows
    syms = sorted(bars)
    if len(syms) < 5:
        print(f"[fail] only {len(syms)} symbols under {cache}", file=sys.stderr)
        return 1
    nbars = {s: len(b) for s, b in bars.items()}
    print(f"[ok] HOURLY crypto universe = {len(syms)} coins, "
          f"{min(nbars.values())}-{max(nbars.values())} bars each "
          f"(~{min(nbars.values())//H}-{max(nbars.values())//H}d), cost {args.cost_bps:g}bps/leg\n")

    # ---- cointegration screen (half-life in HOURS) ----
    screened = screen_pairs(syms, bars, min_overlap=2000,
                            max_half_life=args.max_half_life, min_half_life=args.min_half_life)
    print(f"===== COINTEGRATION-SCREENED PAIRS (half-life {args.min_half_life:g}-{args.max_half_life:g}h): "
          f"{len(screened)} found =====")
    print(f"{'pair':<16}{'half-life':>11}{'hedge':>7}{'ret':>8}{'sharpe':>8}{'maxDD':>8}")
    print("-" * 60)
    pair_results, rows_out = [], []
    for r in screened[:args.top]:
        lb = int(max(24, min(240, r["half_life"] * 3)))
        res = backtest_pairs(bars[r["a"]], bars[r["b"]], sym_a=r["a"], sym_b=r["b"],
                             lookback=lb, entry_z=2.0, exit_z=0.5,
                             cost_bps=args.cost_bps, periods_per_year=PPY)
        pair_results.append(res)
        print(f"{r['a']+'/'+r['b']:<16}{r['half_life']:>9.1f}h{r['hedge']:>7.2f}"
              f"{res.total_return*100:>7.1f}%{res.sharpe:>8.2f}{res.max_drawdown*100:>7.1f}%")
        rows_out.append({**r, "ret": res.total_return, "sharpe": res.sharpe, "maxdd": res.max_drawdown})

    bret = bsh = bdd = 0.0
    if pair_results:
        bret, bsh, bdd = _basket(pair_results)
        print(f"\n  IN-SAMPLE BASKET of top {len(pair_results)} pairs:  ret {bret*100:+.1f}%  "
              f"Sharpe {bsh:.2f}  maxDD {bdd*100:.1f}%  (OVERFITS — see walk-forward)")

    # ---- WALK-FORWARD: re-screen + re-hedge each month, trade the next (THE honest number) ----
    train, test = args.train_days * H, args.test_days * H
    wf = walkforward_pairs(bars, train=train, test=test, top_n=args.top * 2,
                           max_half_life=args.max_half_life, min_half_life=args.min_half_life,
                           cost_bps=args.cost_bps, periods_per_year=PPY)
    print(f"\n  WALK-FORWARD (re-screen+re-hedge each {args.test_days}d, train {args.train_days}d, "
          f"{wf.n_windows} windows):")
    print(f"    ret {wf.total_return*100:+.1f}%  Sharpe {wf.sharpe:.2f}  maxDD {wf.max_drawdown*100:.1f}%  "
          f"({wf.n_oos_bars} fully-OOS hourly bars)")
    print(f"    ^^ THE MOST HONEST NUMBER — every trade on data unseen at selection time.")

    # ---- cross-sectional momentum (lookback/hold in HOURS) ----
    print(f"\n===== CROSS-SECTIONAL MOMENTUM (market-neutral L/S, {len(syms)}-coin universe) =====")
    best = None
    grid = [(24, 6, 3), (72, 24, 3), (168, 24, 4), (168, 72, 4), (336, 72, 5)]  # 1d/3d/7d/7d/14d lookbacks
    for lb, hd, k in grid:
        r = backtest_cross_sectional_momentum(bars, lookback=lb, hold=hd, top_k=k, bottom_k=k,
                                              cost_bps=args.cost_bps, periods_per_year=PPY, market_neutral=True)
        # also test the reversal (long losers / short winners) — crypto often mean-reverts intraday
        rr = backtest_cross_sectional_momentum(bars, lookback=lb, hold=hd, top_k=k, bottom_k=k,
                                               cost_bps=args.cost_bps, periods_per_year=PPY,
                                               market_neutral=True, reverse=True)
        print(f"  lb {lb:>3}h hold {hd:>2}h k {k}:  MOM ret {r.total_return*100:+6.1f}% Sh {r.sharpe:5.2f} "
              f"DD {r.max_drawdown*100:5.1f}%  |  REV ret {rr.total_return*100:+6.1f}% Sh {rr.sharpe:5.2f}")
        for tag, x in ((f"mom lb{lb}/hd{hd}/k{k}", r), (f"rev lb{lb}/hd{hd}/k{k}", rr)):
            if best is None or x.sharpe > best[1]:
                best = (tag, x.sharpe, x.total_return)

    Path(args.out).write_text(json.dumps({
        "n_coins": len(syms), "timeframe": args.timeframe, "ppy": PPY, "cost_bps": args.cost_bps,
        "n_screened_pairs": len(screened), "top_pairs": rows_out,
        "basket_insample": {"return": bret, "sharpe": bsh, "maxdd": bdd} if pair_results else None,
        "walkforward": {"return": wf.total_return, "sharpe": wf.sharpe, "maxdd": wf.max_drawdown,
                        "windows": wf.n_windows, "oos_bars": wf.n_oos_bars},
        "best_cross_sectional": {"config": best[0], "sharpe": best[1], "return": best[2]} if best else None,
    }, indent=2))

    print(f"\n[verdict] walk-forward pairs Sharpe {wf.sharpe:.2f}, best cross-sectional {best[1]:.2f} ({best[0]}).")
    print(f"          {'EDGE survives OOS' if (wf.sharpe > 0.7 or best[1] > 0.7) else 'modest' if (wf.sharpe > 0.3 or best[1] > 0.3) else 'NO honest edge — rejected'}.")
    print(f"[done] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
