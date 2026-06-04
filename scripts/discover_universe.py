"""
Universe-scale market-neutral discovery — cointegration-screen EVERY pair in a large
cached universe, backtest the top stable pairs individually AND as a diversified basket
(diversification across uncorrelated spreads is what lifts a market-neutral Sharpe), and
run cross-sectional momentum across the whole universe. Everything is judged honestly
(Sharpe, maxDD) — these strategies are market-NEUTRAL so there is no buy-and-hold to beat;
the return itself is the alpha (it does not depend on market direction).

Reusable by the scheduled discovery job. Run:
  .venv/bin/python scripts/discover_universe.py --cache "/Volumes/My Passport/AlpcaData/cache"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from alpca.backtest.cross_sectional import backtest_cross_sectional_momentum  # noqa: E402
from alpca.backtest.evaluation import max_drawdown_of, sharpe_of  # noqa: E402
from alpca.backtest.pairs import backtest_pairs, screen_pairs  # noqa: E402

# light sector map (for labeling whether screened pairs are same-sector = sane)
SECTOR = {}
for grp, names in {
    "tech": "AAPL MSFT GOOGL AMZN META NVDA TSLA AVGO ORCL ADBE CRM AMD INTC CSCO QCOM TXN MU AMAT XLK QQQ",
    "fin": "JPM BAC WFC GS MS C V MA AXP BLK SCHW PNC USB COF XLF",
    "health": "UNH JNJ LLY PFE MRK ABBV TMO ABT DHR BMY AMGN GILD XLV",
    "consumer": "WMT PG KO PEP COST MCD NKE HD LOW SBUX TGT CL XLP XLY",
    "energy": "XOM CVX COP SLB EOG XLE",
    "industrial": "CAT BA GE HON UPS DE LMT RTX XLI",
    "broad": "SPY IWM DIA TLT GLD XLU XLB",
}.items():
    for n in names.split():
        SECTOR[n] = grp


def _basket(results, starting=100_000.0, ppy=252.0):
    """Equal-weight portfolio of pair equity curves -> (total_return, sharpe, maxdd)."""
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
    ap.add_argument("--cache", default="data/cache")
    ap.add_argument("--timeframe", default="1day")
    ap.add_argument("--max-half-life", type=float, default=30.0)
    ap.add_argument("--top", type=int, default=12, help="how many screened pairs to backtest/basket")
    ap.add_argument("--out", default="data/universe_discovery.json")
    args = ap.parse_args()

    cache = Path(args.cache)
    bars = {}
    for p in sorted(cache.glob(f"*_{args.timeframe}_bars.jsonl")):
        sym = p.name.split(f"_{args.timeframe}_")[0]
        rows = [json.loads(l) for l in p.open() if l.strip()]
        if len(rows) > 200:
            bars[sym] = rows
    syms = sorted(bars)
    if len(syms) < 5:
        print(f"[fail] only {len(syms)} symbols cached under {cache}", file=sys.stderr)
        return 1
    print(f"[ok] universe = {len(syms)} symbols on {args.timeframe}\n")

    # ---- cointegration screen ----
    screened = screen_pairs(syms, bars, min_overlap=200, max_half_life=args.max_half_life, min_half_life=3)
    print(f"===== COINTEGRATION-SCREENED PAIRS (half-life <= {args.max_half_life:g}d): {len(screened)} found =====")
    print(f"{'pair':<14}{'half-life':>10}{'hedge':>7}{'ret':>8}{'sharpe':>8}{'maxDD':>8}  sector")
    print("-" * 70)
    pair_results = []
    rows_out = []
    for r in screened[:args.top]:
        lb = int(max(20, min(120, r["half_life"] * 3)))
        res = backtest_pairs(bars[r["a"]], bars[r["b"]], sym_a=r["a"], sym_b=r["b"],
                             lookback=lb, entry_z=2.0, exit_z=0.5, cost_bps=2.0)
        pair_results.append(res)
        same = SECTOR.get(r["a"], "?") == SECTOR.get(r["b"], "??")
        tag = f"{SECTOR.get(r['a'],'?')}{'=' if same else '/'}{SECTOR.get(r['b'],'?')}"
        print(f"{r['a']+'/'+r['b']:<14}{r['half_life']:>9.1f}d{r['hedge']:>7.2f}"
              f"{res.total_return*100:>7.1f}%{res.sharpe:>8.2f}{res.max_drawdown*100:>7.1f}%  {tag}")
        rows_out.append({**r, "ret": res.total_return, "sharpe": res.sharpe, "maxdd": res.max_drawdown})

    bret = bsh = bdd = oos_sh = 0.0
    if pair_results:
        bret, bsh, bdd = _basket(pair_results)
        print(f"\n  IN-SAMPLE BASKET of top {len(pair_results)} pairs (equal-weight):"
              f"  ret {bret*100:+.1f}%  Sharpe {bsh:.2f}  maxDD {bdd*100:.1f}%  (this OVERFITS — see OOS)")

    # ---- OUT-OF-SAMPLE validation: screen on first 60%, trade the SAME basket on last 40% ----
    isamp = {s: b[:int(len(b) * 0.6)] for s, b in bars.items()}
    osamp = {s: b[int(len(b) * 0.6):] for s, b in bars.items()}
    is_screen = screen_pairs(syms, isamp, min_overlap=120, max_half_life=args.max_half_life, min_half_life=3)
    oos_top = is_screen[:args.top]
    oos_res = [backtest_pairs(osamp[r["a"]], osamp[r["b"]],
                              lookback=int(max(20, min(120, r["half_life"] * 3))),
                              entry_z=2.0, exit_z=0.5, cost_bps=2.0) for r in oos_top]
    oret, oos_sh, odd = _basket(oos_res)
    print(f"  OUT-OF-SAMPLE BASKET (screened on first 60%, traded on held-out 40%):"
          f"  ret {oret*100:+.1f}%  Sharpe {oos_sh:.2f}  maxDD {odd*100:.1f}%")
    print(f"  ^ THE HONEST NUMBER. {'holds up — real market-neutral edge' if oos_sh > 0.7 else 'modest but positive OOS edge' if oos_sh > 0.2 else 'collapses OOS — overfit'}.")

    # ---- cross-sectional momentum across the whole universe ----
    print(f"\n===== CROSS-SECTIONAL MOMENTUM (market-neutral L/S, {len(syms)}-name universe) =====")
    best = None
    for lb, hd, k in [(60, 20, 5), (120, 20, 5), (120, 60, 10), (250, 20, 10)]:
        r = backtest_cross_sectional_momentum(bars, lookback=lb, hold=hd, top_k=k, bottom_k=k,
                                              cost_bps=2.0, market_neutral=True)
        print(f"  lookback {lb:>3} hold {hd:>2} k {k:>2}:  ret {r.total_return*100:+6.1f}%  "
              f"Sharpe {r.sharpe:5.2f}  maxDD {r.max_drawdown*100:5.1f}%  rebals {r.n_rebalances}")
        if best is None or r.sharpe > best[1]:
            best = (f"lb{lb}/hd{hd}/k{k}", r.sharpe, r.total_return)

    Path(args.out).write_text(json.dumps({
        "n_symbols": len(syms), "timeframe": args.timeframe,
        "n_screened_pairs": len(screened), "top_pairs": rows_out,
        "basket_insample": {"return": bret, "sharpe": bsh, "maxdd": bdd} if pair_results else None,
        "basket_oos": {"return": oret, "sharpe": oos_sh, "maxdd": odd},
        "best_cross_sectional": {"config": best[0], "sharpe": best[1], "return": best[2]} if best else None,
    }, indent=2))
    print(f"\n[done] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
