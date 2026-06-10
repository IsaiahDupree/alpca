"""
Harness-test Time-Series Momentum (TSMOM) on a diversified ETF panel — with the Kim (2016)
null baked in: does the momentum SIGN-timing add anything beyond vol-scaling?

Runs three strategies on the same panel and judges each by the harness primitives:
  tsmom     = sign(trailing 12m) * vol-scale   (momentum + vol-scale)
  long_vol  = always-long * vol-scale          (THE NULL: vol-scale only)
  ew_bh     = equal-weight buy-and-hold        (plain panel beta)
Honest bar: tsmom must beat BOTH long_vol AND ew_bh, OUT-OF-SAMPLE, net of cost. If
tsmom ~= long_vol, the "momentum" is illusory (Kim 2016) — reported as a clean negative.

Run: .venv/bin/python scripts/test_tsmom.py --cache "/Volumes/My Passport/AlpcaData/cache"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from alpca.backtest.evaluation import (  # noqa: E402
    segment_sharpes, sharpe_of, sharpe_pvalue, sharpe_tstat)
from alpca.backtest.tsmom import backtest_tsmom  # noqa: E402

PPY = 252.0
PANEL = ["SPY", "QQQ", "IWM", "DIA", "EEM", "EFA", "TLT", "GLD", "SLV"]  # equity/intl/bond/metals


def oos(eq, frac=0.3):
    n = len(eq)
    sp = int(n * (1 - frac))
    return sharpe_of(eq[:sp], PPY), sharpe_of(eq[sp:], PPY)


def judge(r):
    is_sh, oos_sh = oos(r.equity_curve)
    t, p = sharpe_tstat(r.equity_curve), sharpe_pvalue(r.equity_curve)
    segs = segment_sharpes(r.equity_curve, PPY, 4)
    return dict(mode=r.mode, full=r.sharpe, is_sh=is_sh, oos=oos_sh, ret=r.total_return,
                dd=r.max_drawdown, lev=r.avg_gross_leverage, tstat=t, pval=p,
                segs=[round(x, 2) for x in segs], sig=(p < 0.05 and abs(t) > 2.0),
                stable=(sum(1 for s in segs if s > 0) * 2 >= len(segs)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/Volumes/My Passport/AlpcaData/cache")
    ap.add_argument("--cost-bps", type=float, default=2.0)
    ap.add_argument("--lookback", type=int, default=252)
    ap.add_argument("--target-vol", type=float, default=0.10)
    ap.add_argument("--out", default="data/tsmom_results.json")
    args = ap.parse_args()
    cache = Path(args.cache)

    bars_by = {}
    for s in PANEL:
        p = cache / f"{s}_1day_bars.jsonl"
        if p.exists():
            bars_by[s] = [json.loads(l) for l in p.open() if l.strip()]
    have = [s for s in PANEL if s in bars_by]
    print(f"[ok] panel = {have} ({len(have)} assets), cost {args.cost_bps:g}bps, "
          f"lookback {args.lookback}d, target-vol {args.target_vol:.0%}\n")

    rows = []
    print(f"{'mode':<10}{'full':>7}{'IS':>7}{'OOS':>7}{'ret':>9}{'maxDD':>8}{'lev':>6}"
          f"{'tstat':>7}{'pval':>7}  segs           sig/stbl")
    print("-" * 92)
    res = {}
    for mode in ("tsmom", "long_vol", "ew_bh"):
        r = backtest_tsmom(bars_by, have, mode=mode, lookback=args.lookback,
                           target_vol=args.target_vol, cost_bps=args.cost_bps,
                           rebalance=21, periods_per_year=PPY)
        j = judge(r)
        res[mode] = j
        rows.append(j)
        flags = ("SIG" if j["sig"] else "   ") + ("/STBL" if j["stable"] else "/    ")
        print(f"{mode:<10}{j['full']:>7.2f}{j['is_sh']:>7.2f}{j['oos']:>7.2f}{j['ret']*100:>8.0f}%"
              f"{j['dd']*100:>7.1f}%{j['lev']:>6.2f}{j['tstat']:>7.2f}{j['pval']:>7.3f}  "
              f"{str(j['segs']):<16}{flags}")

    print("\n" + "=" * 92)
    tm, lv, bh = res["tsmom"], res["long_vol"], res["ew_bh"]
    print(f"Does momentum SIGN-timing add value beyond vol-scaling? (Kim 2016 test)")
    print(f"  tsmom OOS Sharpe {tm['oos']:.2f}  vs  long_vol (vol-scale only) OOS {lv['oos']:.2f}  "
          f"vs  ew buy-hold OOS {bh['oos']:.2f}")
    beats_null = tm["oos"] > lv["oos"] + 0.10
    beats_bh = tm["oos"] > bh["oos"] + 0.05
    if beats_null and beats_bh:
        verdict = "GENUINE: momentum adds value beyond vol-scaling AND beats buy-hold OOS."
    elif not beats_null:
        verdict = ("ILLUSORY (Kim 2016): tsmom ~= vol-scaled-static — the 'momentum' is just "
                   "vol-targeting, not timing. Clean negative.")
    else:
        verdict = "Momentum beats the vol-scale null but NOT buy-and-hold OOS — marginal."
    print(f"VERDICT: {verdict}")

    Path(args.out).write_text(json.dumps(
        {"panel": have, "cost_bps": args.cost_bps, "lookback": args.lookback,
         "results": rows, "verdict": verdict}, indent=2))
    print(f"\n[done] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
