"""
Case 44 — Does mid-cap momentum's edge SURVIVE realistic borrow? (+ the borrow-free long/index-hedged form)

Case 43 showed vol-managed momentum jumps 0.39 -> 1.35 once the delisted value-traps are back in — but
that gain is from SHORTING names going to zero, which are exactly the hardest-to-borrow / no-locate names
(the adverse-selection borrow wall that killed PEAD & the SI-tilt). This re-runs vol-managed momentum on
both the survivor-only and the +delisted (point-in-time) universes under three short-side regimes:

  (1) borrow_free      — dollar-neutral L/S, zero borrow cost   (the optimistic 1.35 baseline)
  (2) adverse_borrow   — per-name daily borrow that RAMPS with the short's trailing decline + low price;
                         below a no-locate floor the name CAN'T be shorted (dropped from the short leg).
                         This is the honest stress: the crashing names you most want to short are
                         precisely the ones that go special / no-locate.
  (3) long_index_hedge — long the winners only (survivorship-clean, borrow-FREE) + short SPY to kill
                         beta. No single-name shorts at all -> no borrow wall (the EAR-PEAD pattern).
                         This is the candidate DEPLOYABLE form.

The decisive reads: how much of 1.35 survives adverse_borrow, and whether the borrow-free long/hedged
build is a real edge on the point-in-time universe.

Run: .venv/bin/python scripts/test_momentum_borrow.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from alpca.backtest.factor import _price_ret, vol_managed_momentum_signal  # noqa: E402
from alpca.backtest.evaluation import deflated_sharpe_ratio, max_drawdown_of, sharpe_of  # noqa: E402

PPY = 252.0


def _load_multi(dirs):
    out = {}
    for c in dirs:
        for p in Path(c).glob("*_1day_bars.jsonl"):
            rows = [json.loads(l) for l in p.open() if l.strip()]
            if rows:
                out[p.name.split("_1day_")[0]] = rows
    return out


def _eq(daily, s=1.0):
    e = [s]
    for x in daily:
        e.append(e[-1] * (1 + x))
    return e


def _per_year(dates, daily):
    by = {}
    for ep, x in zip(dates, daily):
        by.setdefault(time.gmtime(ep).tm_year, []).append(x)
    yr = {y: round(sharpe_of(_eq(v), PPY), 2) for y, v in by.items() if len(v) >= 30}
    return yr, sum(1 for s in yr.values() if s > 0), len(yr)


def _adverse_rate(r21: float, px: float) -> float:
    """Annualized borrow rate for shorting a name with trailing-21d return r21 at price px.
    Ramps base->special->saturating as the name collapses; np.inf = NO-LOCATE (cannot short)."""
    base, special, sat = 0.02, 0.30, 1.50
    if r21 < -0.70 or px < 2.0:
        return np.inf                                  # going to zero / penny -> no borrow available
    if r21 > -0.10:
        return base
    if r21 > -0.40:                                    # ramp base->special over [-10%, -40%]
        return base + (special - base) * (-0.10 - r21) / 0.30
    return special + (sat - special) * (-0.40 - r21) / 0.30   # special->saturating over [-40%, -70%]


def run_momentum(bars, spy_ret_by_ts, *, mode, top_frac=0.2, rebalance_days=21, cost_bps=2.0,
                 winsor=0.5):
    syms = sorted(bars)
    if len(syms) < 10:
        return None
    master = sorted({int(b["timestamp"]) for s in syms for b in bars[s]})
    price, ret = _price_ret(bars, syms, master)
    # Winsorize daily returns: |r| > winsor in these near-delisting microcaps is almost always a
    # mis-adjusted reverse-split / data artifact (WW +152x, PROK +5x), NOT a tradeable move. Clip both
    # the P&L AND the prices that feed the momentum signal so the ranking isn't corrupted by artifacts.
    if winsor:
        ret = np.clip(ret, -winsor, winsor)
        # rebuild a clean price path from clipped returns so the signal uses artifact-free momentum
        clean = np.full_like(price, np.nan)
        for j in range(price.shape[1]):
            col = price[:, j]
            first = next((i for i in range(len(col)) if np.isfinite(col[i])), None)
            if first is None:
                continue
            p = col[first]; clean[first, j] = p
            for i in range(first + 1, len(col)):
                if np.isfinite(col[i]):
                    p = p * (1.0 + ret[i, j]); clean[i, j] = p
        price = clean
    signal = vol_managed_momentum_signal(120, 21, 60)(master, syms, price)
    T, N = len(master), len(syms)
    k = max(1, int(round(N * top_frac)))
    spy = np.array([spy_ret_by_ts.get(t, 0.0) for t in master])

    eq = [1.0]; daily = []; wl = np.zeros(N); ws = np.zeros(N); spy_w = 0.0; prev = np.zeros(N)
    no_locate_drops = 0; rebals = 0
    for t in range(1, T):
        if (t - 1) % rebalance_days == 0:
            s = signal[t - 1]; ok = np.isfinite(s)
            if ok.sum() >= 2 * k:
                order = np.argsort(np.where(ok, s, np.inf))
                order = order[np.isin(order, np.where(ok)[0])]
                low, high = order[:k], order[-k:]          # low = losers (short), high = winners (long)
                wl = np.zeros(N); ws = np.zeros(N); spy_w = 0.0
                if mode == "long_index_hedge":
                    wl[high] = 1.0 / k                     # fully long winners
                    spy_w = -1.0                           # short SPY to neutralize beta (borrow-free)
                else:
                    wl[high] = 0.5 / k
                    for j in low:                          # short losers, 0.5/k each
                        ws[j] = -0.5 / k
                rebals += 1
        # borrow cost on the short book (adverse_borrow modes). The "_neutral" variant re-scales the
        # LONG leg down to match the available (post-no-locate) short notional, so dropping no-locate
        # shorts does NOT leak net-long market beta into the Sharpe — isolating realizable ALPHA.
        borrow_cost = 0.0
        if mode in ("adverse_borrow", "adverse_borrow_neutral"):
            new_ws = ws.copy()
            for j in np.where(ws < 0)[0]:
                a = t - 1 - 21
                r21 = (price[t - 1, j] / price[a, j] - 1.0) if (a >= 0 and price[a, j] > 0
                                                               and np.isfinite(price[t - 1, j])) else 0.0
                rate = _adverse_rate(r21, price[t - 1, j] if np.isfinite(price[t - 1, j]) else 0.0)
                if not np.isfinite(rate):
                    new_ws[j] = 0.0                        # no-locate -> can't short this name
                    if (t - 1) % rebalance_days == 0:
                        no_locate_drops += 1
                else:
                    borrow_cost += abs(ws[j]) * rate / PPY
            ws = new_ws
            if mode == "adverse_borrow_neutral":
                short_notional = float(-ws.sum())
                long_notional = float(wl.sum())
                if short_notional <= 0:
                    wl = wl * 0.0          # ALL shorts no-located -> dollar-neutral means FLAT, not 100% long
                elif long_notional > short_notional:
                    wl = wl * (short_notional / long_notional)   # match long to available short
        w = wl + ws
        turnover = float(np.abs(w - prev).sum())
        port = float(np.nansum(w * np.nan_to_num(ret[t]))) + spy_w * spy[t]
        port -= turnover * (cost_bps / 1e4) + borrow_cost
        eq.append(eq[-1] * (1 + port)); daily.append(port); prev = w
    yr, pos, ny = _per_year(master[1:], daily)
    return {"sharpe": round(sharpe_of(eq, PPY), 3), "maxdd": round(max_drawdown_of(eq), 3),
            "per_year": yr, "pos_years": f"{pos}/{ny}", "no_locate_drops": no_locate_drops,
            "dsr": round(deflated_sharpe_ratio(eq, n_trials=88, sharpe_variance=1e-4), 3),
            "daily": daily, "dates": master[1:]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--midcap", default="/Volumes/My Passport/AlpcaData/cache_midcap")
    ap.add_argument("--delisted", default="/Volumes/My Passport/AlpcaData/cache_midcap_delisted")
    ap.add_argument("--spy-cache", default="/Volumes/My Passport/AlpcaData/cache")
    ap.add_argument("--out", default="data/momentum_borrow.json")
    args = ap.parse_args()

    spy_bars = _load_multi([args.spy_cache]).get("SPY", [])
    spy_ret = {}
    sb = sorted(spy_bars, key=lambda b: int(b["timestamp"]))
    for i in range(1, len(sb)):
        p0 = float(sb[i - 1]["close"])
        if p0 > 0:
            spy_ret[int(sb[i]["timestamp"])] = float(sb[i]["close"]) / p0 - 1.0

    surv = _load_multi([args.midcap])
    aug = _load_multi([args.midcap, args.delisted])
    print(f"[ok] survivor {len(surv)} · +delisted {len(aug)} (+{len(aug)-len(surv)}) · SPY {len(spy_ret)} days\n")

    out = {}
    print(f"{'mode':>18}{'universe':>14}{'sharpe':>8}{'+yrs':>7}{'maxDD':>8}{'DSR':>6}{'noLoc':>7}")
    print("-" * 68)
    for mode in ("borrow_free", "adverse_borrow", "adverse_borrow_neutral", "long_index_hedge"):
        for ulabel, b in (("survivor", surv), ("+delisted", aug)):
            r = run_momentum(b, spy_ret, mode=mode)
            if r is None:
                continue
            out[f"{mode}|{ulabel}"] = {k: v for k, v in r.items() if k not in ("daily", "dates")}
            print(f"{mode:>18}{ulabel:>14}{r['sharpe']:>8.2f}{r['pos_years']:>7}{r['maxdd']:>8.2f}"
                  f"{r['dsr']:>6.2f}{r['no_locate_drops']:>7}")

    # headline deltas
    bf = out.get("borrow_free|+delisted", {}).get("sharpe", 0)
    ab = out.get("adverse_borrow_neutral|+delisted", {}).get("sharpe", 0)
    lh_s = out.get("long_index_hedge|survivor", {}).get("sharpe", 0)
    lh_d = out.get("long_index_hedge|+delisted", {}).get("sharpe", 0)
    print(f"\n[borrow stress] +delisted L/S: borrow-free {bf:.2f} -> adverse_borrow {ab:.2f} "
          f"(survives {100*ab/bf:.0f}% of the gross)" if bf else "")
    print(f"[deployable long/index-hedge] survivor {lh_s:.2f} · +delisted {lh_d:.2f} (borrow-FREE)")
    verdict = ("LONG/HEDGED IS THE EDGE — borrow-free, survives point-in-time" if lh_d > 0.4 else
               "weak — long/hedged momentum thin once survivorship-correct")
    print(f"[verdict] {verdict}")
    out["_summary"] = {"ls_borrow_free_pit": bf, "ls_adverse_borrow_pit": ab,
                       "long_hedge_survivor": lh_s, "long_hedge_pit": lh_d, "verdict": verdict}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"[saved] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
