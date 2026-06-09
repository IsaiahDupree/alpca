"""
Harness-test the Avellaneda-Stoikov INVENTORY-SKEW sizing (alpca/backtest/inventory_skew.py).

Two honest questions, judged by the same primitives as the rest of the platform
(sharpe_of / sharpe_tstat / sharpe_pvalue / segment_sharpes / buy_and_hold):

  CONTEXT 1 (single asset, directional): does A-S continuous vol-scaled mean-reversion
  sizing beat (a) a naive BINARY z-score entry and (b) BUY-AND-HOLD — out-of-sample,
  statistically significant, stable? (Honest null: it's a risk overlay = beta, not alpha.)

  CONTEXT 2 (pairs spread, market-neutral): on a cointegrated spread, does A-S continuous
  sizing beat the classic BINARY z-entry pairs rule? Here the spread return IS the alpha
  (no B&H). This is the most legitimate test of the inventory-aware idea, because the
  spread genuinely mean-reverts.

Run: .venv/bin/python scripts/test_inventory_skew.py --cache "/Volumes/My Passport/AlpcaData/cache"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from alpca.backtest.evaluation import (  # noqa: E402
    buy_and_hold, max_drawdown_of, sharpe_of, sharpe_pvalue, sharpe_tstat, segment_sharpes)
from alpca.backtest.inventory_skew import (  # noqa: E402
    as_target, as_target_spread, backtest_spread_targets, backtest_targets,
    binary_target, binary_target_spread, spread_series, _closes)
from alpca.backtest.pairs import screen_pairs  # noqa: E402

PPY = 252.0


def judge(eq, ppy=PPY):
    sh = sharpe_of(eq, ppy)
    return dict(ret=(eq[-1] - eq[0]) / eq[0], sharpe=sh, dd=max_drawdown_of(eq),
                tstat=sharpe_tstat(eq), pval=sharpe_pvalue(eq),
                segs=[round(s, 2) for s in segment_sharpes(eq, ppy, 4)],
                sig=(sharpe_pvalue(eq) < 0.05 and abs(sharpe_tstat(eq)) > 2.0),
                stable=(sum(1 for s in segment_sharpes(eq, ppy, 4) if s > 0) * 2 >= 4))


def line(label, j):
    flags = ("SIG" if j["sig"] else "   ") + ("/STBL" if j["stable"] else "/    ")
    print(f"  {label:<26} ret {j['ret']*100:>7.1f}%  Sh {j['sharpe']:>5.2f}  "
          f"DD {j['dd']*100:>6.1f}%  t {j['tstat']:>5.2f} p {j['pval']:.3f}  "
          f"segs {j['segs']}  {flags}")


def oos_sharpe(eq, frac=0.3):
    split = int(len(eq) * (1 - frac))
    return sharpe_of(eq[split:], PPY)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/Volumes/My Passport/AlpcaData/cache")
    ap.add_argument("--symbols", default="SPY,QQQ,AAPL,MSFT,NVDA,IWM,XLF,XLE,XLK,WMT")
    ap.add_argument("--gammas", default="0.5,1.0,2.0,5.0")
    ap.add_argument("--cost-bps", type=float, default=2.0)
    ap.add_argument("--window", type=int, default=20)
    ap.add_argument("--out", default="data/inventory_skew_results.json")
    args = ap.parse_args()
    cache = Path(args.cache)

    def load(sym):
        p = cache / f"{sym}_1day_bars.jsonl"
        return [json.loads(l) for l in p.open() if l.strip()] if p.exists() else []

    gammas = [float(g) for g in args.gammas.split(",")]
    syms = [s.strip() for s in args.symbols.split(",")]
    results = {"context1_single_asset": {}, "context2_pairs": {}}

    # ===================== CONTEXT 1: single-asset, vs binary + vs B&H =====================
    print("=" * 88)
    print("CONTEXT 1 — A-S inventory sizing vs BINARY z-entry vs BUY-AND-HOLD (single asset, daily)")
    print("=" * 88)
    agg = {f"as_g{g}": [] for g in gammas}
    agg["binary"] = []
    agg["bh"] = []
    for sym in syms:
        bars = load(sym)
        if len(bars) < 300:
            continue
        cl = _closes(bars)
        bh = buy_and_hold(bars, PPY)
        print(f"\n[{sym}]  ({len(bars)} bars)")
        line("buy-and-hold", judge(bh.equity_curve))
        agg["bh"].append((bh.sharpe, bh.total_return, oos_sharpe(bh.equity_curve)))
        bj = judge(backtest_targets(cl, binary_target(cl, window=args.window), cost_bps=args.cost_bps))
        line("binary z-entry", bj)
        agg["binary"].append((bj["sharpe"], bj["ret"],
                              oos_sharpe(backtest_targets(cl, binary_target(cl, window=args.window),
                                                          cost_bps=args.cost_bps))))
        for g in gammas:
            eq = backtest_targets(cl, as_target(cl, window=args.window, gamma=g), cost_bps=args.cost_bps)
            j = judge(eq)
            line(f"A-S inventory (γ={g})", j)
            agg[f"as_g{g}"].append((j["sharpe"], j["ret"], oos_sharpe(eq)))

    print("\n" + "-" * 88)
    print("CONTEXT-1 SUMMARY (mean across symbols):  in-sample Sharpe | total ret | OOS Sharpe (held-out 30%)")
    for k in ["bh", "binary"] + [f"as_g{g}" for g in gammas]:
        v = agg[k]
        if not v:
            continue
        msh = sum(x[0] for x in v) / len(v)
        mret = sum(x[1] for x in v) / len(v)
        moos = sum(x[2] for x in v) / len(v)
        print(f"  {k:<16}  Sharpe {msh:>5.2f}   ret {mret*100:>7.1f}%   OOS-Sharpe {moos:>5.2f}")
        results["context1_single_asset"][k] = {"sharpe": msh, "ret": mret, "oos_sharpe": moos}

    # ===================== CONTEXT 2: pairs spread, A-S vs binary (market-neutral) =====================
    print("\n" + "=" * 88)
    print("CONTEXT 2 — A-S continuous sizing vs BINARY z-entry on a COINTEGRATED SPREAD (market-neutral)")
    print("=" * 88)
    # screen the given universe for a few stable cointegrated pairs
    bars_by = {s: load(s) for s in syms if len(load(s)) >= 300}
    screened = screen_pairs(list(bars_by), bars_by, min_overlap=250, max_half_life=40, min_half_life=3)
    print(f"screened {len(screened)} cointegrated pairs from {len(bars_by)} symbols\n")
    rows = []
    for r in screened[:6]:
        sp, _ = spread_series(bars_by[r["a"]], bars_by[r["b"]], r["hedge"])
        if len(sp) < 300:
            continue
        bin_eq = backtest_spread_targets(sp, binary_target_spread(sp, window=args.window), cost_bps=args.cost_bps * 2)
        bj = judge(bin_eq)
        print(f"[{r['a']}/{r['b']}  half-life {r['half_life']:.0f}d]")
        line("binary z-entry spread", bj)
        best_as = None
        for g in gammas:
            as_eq = backtest_spread_targets(sp, as_target_spread(sp, window=args.window, gamma=g),
                                            cost_bps=args.cost_bps * 2)
            j = judge(as_eq)
            line(f"A-S sizing (γ={g})", j)
            if best_as is None or j["sharpe"] > best_as[1]["sharpe"]:
                best_as = (g, j, oos_sharpe(as_eq))
        rows.append({"pair": f"{r['a']}/{r['b']}", "binary_sharpe": bj["sharpe"], "binary_oos": oos_sharpe(bin_eq),
                     "as_best_gamma": best_as[0], "as_sharpe": best_as[1]["sharpe"], "as_oos": best_as[2]})
        print()
    if rows:
        bwin = sum(1 for x in rows if x["as_sharpe"] > x["binary_sharpe"])
        bwin_oos = sum(1 for x in rows if x["as_oos"] > x["binary_oos"])
        print("-" * 88)
        print(f"CONTEXT-2 SUMMARY ({len(rows)} pairs):")
        print(f"  A-S beats binary on IN-SAMPLE Sharpe:  {bwin}/{len(rows)} pairs")
        print(f"  A-S beats binary on OUT-OF-SAMPLE Sharpe: {bwin_oos}/{len(rows)} pairs")
        print(f"  mean binary OOS Sharpe {sum(x['binary_oos'] for x in rows)/len(rows):.2f}  "
              f"vs A-S OOS Sharpe {sum(x['as_oos'] for x in rows)/len(rows):.2f}")
        results["context2_pairs"] = {"pairs": rows, "as_beats_binary_oos": f"{bwin_oos}/{len(rows)}"}

    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\n[done] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
