"""
Harness-test the crypto FUNDING-RATE tilt: a low-turnover long/flat overlay on spot BTC/ETH
that steps OUT of crowded-long regimes (extreme-positive funding z-score), vs the honest
null of BUY-AND-HOLD. Funding from Kraken Futures (free, US-accessible); spot bars from the
Alpaca crypto cache. Funding has edge only at extremes, so the gate only deviates from
buy-hold when |z| is large -> naturally low turnover.

Bar to clear: improve risk-adjusted return (Sharpe) or cut drawdown vs simply HOLDING,
net of cost. CAVEAT: Kraken funding history is ~1 year -> short, single-regime; exploratory.

Run: .venv/bin/python scripts/test_funding_tilt.py --crypto-cache "/Volumes/My Passport/AlpcaData/crypto"
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from alpca.backtest.evaluation import max_drawdown_of, sharpe_of, sharpe_pvalue, sharpe_tstat  # noqa: E402
from alpca.data.funding import daily_funding, fetch_kraken_funding  # noqa: E402

PPY = 365.0  # crypto trades 24/7


def _date(ts):
    return datetime.fromtimestamp(float(ts), timezone.utc).strftime("%Y-%m-%d")


def zscores(series, window):
    z = [0.0] * len(series)
    for t in range(len(series)):
        if t < window:
            continue
        win = series[t - window:t]
        m = sum(win) / window
        sd = (sum((x - m) ** 2 for x in win) / window) ** 0.5
        z[t] = (series[t] - m) / sd if sd > 1e-12 else 0.0
    return z


def run_alloc(rets, allocs, cost_bps):
    eq = [100_000.0]
    prev = 0.0
    for t in range(len(rets)):
        a = allocs[t]
        turn = abs(a - prev)
        eq.append(eq[-1] * (1 + a * rets[t] - turn * cost_bps / 10_000.0))
        prev = a
    return eq


def judge(eq, frac=0.3):
    n = len(eq)
    sp = int(n * (1 - frac))
    return dict(sharpe=sharpe_of(eq, PPY), oos=sharpe_of(eq[sp:], PPY), dd=max_drawdown_of(eq),
                ret=(eq[-1] - eq[0]) / eq[0], tstat=sharpe_tstat(eq), pval=sharpe_pvalue(eq))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--crypto-cache", default="/Volumes/My Passport/AlpcaData/crypto")
    ap.add_argument("--cost-bps", type=float, default=10.0)
    ap.add_argument("--window", type=int, default=14)
    ap.add_argument("--out", default="data/funding_tilt_results.json")
    args = ap.parse_args()
    cache = Path(args.crypto_cache)

    pairs = [("BTCUSD", "PF_XBTUSD"), ("ETHUSD", "PF_ETHUSD")]
    out_all = {}
    for spot_sym, perp in pairs:
        bf = cache / f"{spot_sym}_1day_bars.jsonl"
        if not bf.exists():
            print(f"[skip] no spot bars for {spot_sym}")
            continue
        bars = [json.loads(l) for l in bf.open() if l.strip()]
        try:
            fund = daily_funding(fetch_kraken_funding(perp))
        except Exception as e:
            print(f"[skip] funding fetch failed for {perp}: {type(e).__name__} {e}")
            continue

        # align spot bars to funding dates
        rows = []
        for b in bars:
            d = _date(b["timestamp"])
            if d in fund:
                rows.append((d, float(b["close"]), fund[d]))
        rows.sort()
        if len(rows) < 120:
            print(f"[skip] only {len(rows)} aligned days for {spot_sym}")
            continue
        closes = [r[1] for r in rows]
        fseries = [r[2] for r in rows]
        rets = [(closes[t] - closes[t - 1]) / closes[t - 1] for t in range(1, len(closes))]
        z = zscores(fseries, args.window)

        print(f"\n===== {spot_sym}  ({len(rows)} aligned days, {rows[0][0]}..{rows[-1][0]}) =====")
        bh = judge(run_alloc(rets, [1.0] * len(rets), args.cost_bps))
        print(f"  buy-and-hold:        Sharpe {bh['sharpe']:.2f}  OOS {bh['oos']:.2f}  "
              f"ret {bh['ret']*100:+.0f}%  maxDD {bh['dd']*100:.1f}%")

        rows_out = []
        best = None
        for thr in (1.0, 1.5, 2.0):
            for low in (0.0, 0.5):
                # gate: if yesterday's funding z > thr (crowded long), reduce alloc today
                allocs = []
                for t in range(1, len(closes)):
                    allocs.append(low if z[t - 1] > thr else 1.0)
                j = judge(run_alloc(rets, allocs, args.cost_bps))
                # in-sample (first 70%) Sharpe for selection
                sp = int(len(rets) * 0.7)
                is_sh = sharpe_of(run_alloc(rets[:sp], allocs[:sp], args.cost_bps), PPY)
                rows_out.append({"thr": thr, "low": low, "sharpe": j["sharpe"], "is_sharpe": is_sh,
                                 "oos": j["oos"], "dd": j["dd"], "ret": j["ret"]})
                print(f"  gate z>{thr} -> {low:.1f}:   Sharpe {j['sharpe']:.2f}  IS {is_sh:.2f}  "
                      f"OOS {j['oos']:.2f}  ret {j['ret']*100:+.0f}%  maxDD {j['dd']*100:.1f}%")
                if best is None or is_sh > best["is_sharpe"]:
                    best = rows_out[-1]

        improves = best["oos"] > bh["oos"] + 0.10 or best["dd"] > bh["dd"] + 0.05
        verdict = ("gate IMPROVES on buy-hold OOS (Sharpe or drawdown)"
                   if improves else "gate does NOT beat buy-hold — funding tilt adds nothing here")
        print(f"  -> selected gate z>{best['thr']}->{best['low']}: OOS {best['oos']:.2f} "
              f"vs buy-hold OOS {bh['oos']:.2f}. {verdict}.")
        out_all[spot_sym] = {"buy_hold": bh, "grid": rows_out, "selected": best, "verdict": verdict}

    Path(args.out).write_text(json.dumps(out_all, indent=2, default=float))
    print(f"\n[done] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
