"""Diagnostic: does cost_cal_entry / ou_sizing change ANY trade vs baseline at the validated config?
The prior saved JSON shows refined == baseline to every digit -> either a true no-op (threshold never
binds) or a broken flag. This instruments act_z vs entry_z and the ou_sizing fraction directly, and
re-runs walkforward_pairs baseline vs each flag vs both, on large-cap AND a disjoint universe.
"""
from __future__ import annotations
import json, sys, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from alpca.backtest.pairs import (walkforward_pairs, screen_pairs, ou_sigma_eq, align)  # noqa
from alpca.backtest.evaluation import max_drawdown_of, sharpe_of  # noqa

PPY = 252.0


def _load(c, min_bars=1000):
    out = {}
    for p in Path(c).glob("*_1day_bars.jsonl"):
        rows = [json.loads(l) for l in p.open() if l.strip()]
        if len(rows) >= min_bars:
            out[p.name.split("_1day_")[0]] = rows
    return out


def run(bars, **flags):
    r = walkforward_pairs(bars, train=252, test=63, top_n=10, max_half_life=30.0,
                          min_half_life=3.0, entry_z=2.0, exit_z=0.5, cost_bps=2.0,
                          max_adf=-2.86, periods_per_year=PPY, **flags)
    return r.sharpe, r.max_drawdown, r.total_return, r.n_windows, list(r.daily_returns or [])


def diag_act_z(bars):
    """Replicate the walk-forward selection loop and report, per selected pair, what act_z would be
    relative to entry_z=2.0 and how often it binds (act_z > entry_z) or is unreachable (act_z>8)."""
    syms = sorted(bars)
    sets = [set(float(b.get("timestamp", 0) or 0) for b in bars[s]) for s in syms]
    common = sorted(set.intersection(*sets)) if sets else []
    n = len(common); train, test, top_n = 252, 63, 10
    bymap = {s: {float(b["timestamp"]): b for b in bars[s]} for s in syms}
    aligned = {s: [bymap[s][t] for t in common] for s in syms}
    cost_frac = 2.0 / 1e4
    w = 0; total = 0; binds = 0; unreachable = 0; act_zs = []
    while w + train + test <= n:
        train_slice = {s: aligned[s][w:w + train] for s in syms}
        screened = screen_pairs(syms, train_slice, min_overlap=int(train * 0.8),
                                max_half_life=30.0, min_half_life=3.0, max_adf=-2.86)
        for r in screened[:top_n]:
            h = r["hedge"]
            tr_rows = align(aligned[r["a"]][w:w + train], aligned[r["b"]][w:w + train])
            la = [math.log(c) for _, c, _ in tr_rows]
            lbv = [math.log(c) for _, _, c in tr_rows]
            tr_spread = [la[k] - h * lbv[k] for k in range(len(tr_rows))]
            ou_std = ou_sigma_eq(tr_spread)
            if ou_std <= 0:
                continue
            act_z = max(2.0, 4.0 * cost_frac / ou_std)
            total += 1; act_zs.append(round(act_z, 4))
            if act_z > 2.0 + 1e-9:
                binds += 1
            if act_z > 8.0:
                unreachable += 1
        w += test
    return {"pairs_evaluated": total, "act_z_binds(>2.0)": binds,
            "act_z_unreachable(>8)": unreachable,
            "act_z_min": min(act_zs) if act_zs else None,
            "act_z_max": max(act_zs) if act_zs else None,
            "4*cost_frac": 4.0 * cost_frac}


def main():
    for name, path in [("large-cap(195)", "/Volumes/My Passport/AlpcaData/cache_largecap_sip"),
                       ("mid-cap(disjoint)", "/Volumes/My Passport/AlpcaData/cache_midcap_sip")]:
        bars = _load(path)
        print(f"\n===== {name}: {len(bars)} syms =====")
        d = diag_act_z(bars)
        print("[act_z diag]", json.dumps(d))
        base = run(bars)
        a = run(bars, cost_cal_entry=True)
        b = run(bars, ou_sizing=True, cost_cal_entry=True)
        bonly = run(bars, ou_sizing=True)  # ou_sizing without cost_cal_entry -> uses entry_z as denom
        print(f"  baseline                 Sh={base[0]:+.4f} DD={base[1]*100:+.2f}% ret={base[2]*100:+.2f}% win={base[3]}")
        print(f"  +cost_cal_entry          Sh={a[0]:+.4f} DD={a[1]*100:+.2f}% ret={a[2]*100:+.2f}% win={a[3]}")
        print(f"  +ou_sizing+cost_cal      Sh={b[0]:+.4f} DD={b[1]*100:+.2f}% ret={b[2]*100:+.2f}% win={b[3]}")
        print(f"  +ou_sizing(alone)        Sh={bonly[0]:+.4f} DD={bonly[1]*100:+.2f}% ret={bonly[2]*100:+.2f}% win={bonly[3]}")
        # exact-equality check on the daily return vectors
        def eq(x, y):
            return len(x[4]) == len(y[4]) and all(abs(p - q) < 1e-12 for p, q in zip(x[4], y[4]))
        print(f"  [identical returns?] base==cce:{eq(base,a)}  base==(cce+ou):{eq(base,b)}  base==ou_alone:{eq(base,bonly)}")


if __name__ == "__main__":
    main()
