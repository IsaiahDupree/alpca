"""Adversarial pressure-test of the short-vol / VRP second leg. Cache-only, no fetches."""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from alpca.backtest.evaluation import sharpe_of, max_drawdown_of
from alpca.backtest.combine import correlation
from alpca.live.portfolio import combine_tracks

PPY = 252.0
VOL = "/Volumes/My Passport/AlpcaData/cache_vol"


def _eq(daily, s=1.0):
    e = [s]
    for x in daily:
        e.append(e[-1] * (1 + x))
    return e


def _ret_by_ts(path):
    b = sorted([json.loads(l) for l in Path(path).open() if l.strip()], key=lambda x: int(x["timestamp"]))
    return {int(b[i]["timestamp"]): float(b[i]["close"]) / float(b[i - 1]["close"]) - 1.0
            for i in range(1, len(b)) if float(b[i - 1]["close"]) > 0}


def sh(daily):
    return sharpe_of(_eq(list(daily)), PPY)


def main():
    # pairs OOS daily returns (945 days)
    pj = json.loads(Path("data/pairs_wf_returns.json").read_text())
    pairs = {int(r["asof"]): r["ret"] for r in pj["returns"]}

    svxy = _ret_by_ts(Path(VOL) / "SVXY_1day_bars.jsonl")
    vixy = _ret_by_ts(Path(VOL) / "VIXY_1day_bars.jsonl")
    vxx = _ret_by_ts(Path(VOL) / "VXX_1day_bars.jsonl")
    short_vxx = {t: -r for t, r in vxx.items()}
    # short VIXY too (VIXY is long-vol, like VXX; short it = short-vol)
    short_vixy = {t: -r for t, r in vixy.items()}

    instruments = {
        "long_SVXY (deployed)": svxy,
        "short_VXX": short_vxx,
        "short_VIXY": short_vixy,
    }

    print("=" * 78)
    print("STANDALONE SHORT-VOL SLEEVES (full overlap with pairs)")
    print("=" * 78)
    print(f"{'instrument':>22}{'std_sh':>9}{'maxDD':>9}{'rho_pairs':>11}{'tail_rho':>10}{'2022':>8}")
    for name, s in instruments.items():
        common = sorted(set(pairs) & set(s))
        pv = [pairs[t] for t in common]
        vv = [s[t] for t in common]
        rho = correlation(pv, vv)
        thr = np.percentile(pv, 10)
        tidx = [i for i in range(len(pv)) if pv[i] <= thr]
        trho = correlation([pv[i] for i in tidx], [vv[i] for i in tidx]) if len(tidx) > 3 else 0.0
        # 2022 sharpe of the standalone sleeve
        y22 = [s[t] for t in s if time.gmtime(t).tm_year == 2022]
        print(f"{name:>22}{sh(vv):>9.2f}{max_drawdown_of(_eq(vv))*100:>8.1f}%{rho:>11.3f}{trho:>10.2f}{sh(y22):>8.2f}")

    pairs_alone = sh([pairs[t] for t in sorted(pairs)])
    print(f"\npairs-alone Sharpe (full 945-day OOS) = {pairs_alone:.3f}")

    # ---------- TEST 1: instrument fragility — deployed 92/8 combine_tracks ----------
    print("\n" + "=" * 78)
    print("TEST 1 — INSTRUMENT FRAGILITY (deployed combine_tracks, 92% pairs / 8% short-vol)")
    print("  union-of-dates book (canonical 0.92), AND pairs-overlap-only book (apples-to-apples lift)")
    print("=" * 78)
    print(f"{'instrument':>22}{'book_sh(union)':>16}{'maxDD':>9}{'overlap_sh':>12}{'lift_overlap':>14}")
    w = {"pairs": 0.92, "short_vol": 0.08, "momentum": 0.0}
    for name, s in instruments.items():
        # union book exactly as deployed backtest does
        book = combine_tracks({"pairs": pairs, "short_vol": s}, weights=w)
        union_sh = sharpe_of(book.equity_curve, PPY)
        union_dd = max_drawdown_of(book.equity_curve)
        # overlap-only: restrict short_vol to pairs dates so the 0.92-cash days don't dominate
        common = sorted(set(pairs) & set(s))
        s_ov = {t: s[t] for t in common}
        book_ov = combine_tracks({"pairs": {t: pairs[t] for t in common}, "short_vol": s_ov}, weights=w)
        ov_sh = sharpe_of(book_ov.equity_curve, PPY)
        pairs_ov = sh([pairs[t] for t in common])
        print(f"{name:>22}{union_sh:>16.3f}{union_dd*100:>8.1f}%{ov_sh:>12.3f}{ov_sh-pairs_ov:>+14.3f}")

    # ---------- TEST 1b: inverse-vol combined (the test_short_vol.py number) ----------
    print("\nInverse-vol combined Sharpe vs pairs (replicates Case-49 1.08 claim):")
    print(f"{'instrument':>22}{'invvol_sh':>11}{'pairs_alone':>13}{'lift':>9}")
    from alpca.backtest.combine import inverse_vol_weights, blend, equity_from_returns
    for name, s in instruments.items():
        common = sorted(set(pairs) & set(s))
        pv = [pairs[t] for t in common]
        vv = [s[t] for t in common]
        ivw = inverse_vol_weights({"pairs": pv, "sv": vv})
        iv_ret = blend({"pairs": pv, "sv": vv}, ivw)
        iv_sh = sharpe_of(equity_from_returns(iv_ret), PPY)
        pa = sh(pv)
        print(f"{name:>22}{iv_sh:>11.3f}{pa:>13.3f}{iv_sh-pa:>+9.3f}  (ivw sv={ivw['sv']:.2f})")

    # ---------- TEST 2: sub-period split ----------
    print("\n" + "=" * 78)
    print("TEST 2 — SUB-PERIOD: split pairs-overlap into halves (by date)")
    print("  is short-vol positive AND uncorrelated in BOTH halves, or carried by one calm stretch?")
    print("=" * 78)
    for name, s in instruments.items():
        common = sorted(set(pairs) & set(s))
        mid = len(common) // 2
        for label, idx in [("H1", common[:mid]), ("H2", common[mid:])]:
            pv = [pairs[t] for t in idx]
            vv = [s[t] for t in idx]
            rho = correlation(pv, vv)
            d0 = time.strftime("%Y-%m", time.gmtime(idx[0]))
            d1 = time.strftime("%Y-%m", time.gmtime(idx[-1]))
            ivw = inverse_vol_weights({"pairs": pv, "sv": vv})
            iv_ret = blend({"pairs": pv, "sv": vv}, ivw)
            iv_sh = sharpe_of(equity_from_returns(iv_ret), PPY)
            print(f"{name:>22} {label} [{d0}..{d1}]  sv_sh={sh(vv):>6.2f}  rho={rho:>+6.3f}  "
                  f"pairs_sh={sh(pv):>5.2f}  combo_sh={iv_sh:>5.2f}  lift={iv_sh-sh(pv):>+5.2f}")

    # ---------- TEST 3: weight sensitivity (deployed SVXY) ----------
    print("\n" + "=" * 78)
    print("TEST 3 — WEIGHT SENSITIVITY (long_SVXY, union book like deployed backtest)")
    print("  is the lift an inverse-vol artifact, or stable across nearby 92/8-style weights?")
    print("=" * 78)
    print(f"{'pairs/sv':>14}{'book_sh':>10}{'maxDD':>9}{'lift_vs_pairs100':>18}")
    s = svxy
    # pairs-100 reference under the SAME union machinery (sv weight 0)
    base = combine_tracks({"pairs": pairs, "short_vol": s}, weights={"pairs": 1.0, "short_vol": 0.0, "momentum": 0.0})
    base_sh = sharpe_of(base.equity_curve, PPY)
    for wv in [0.0, 0.04, 0.06, 0.08, 0.10, 0.12, 0.16, 0.20]:
        ww = {"pairs": round(1 - wv, 3), "short_vol": wv, "momentum": 0.0}
        book = combine_tracks({"pairs": pairs, "short_vol": s}, weights=ww)
        bsh = sharpe_of(book.equity_curve, PPY)
        bdd = max_drawdown_of(book.equity_curve)
        print(f"{1-wv:>7.2f}/{wv:<5.2f}{bsh:>10.3f}{bdd*100:>8.1f}%{bsh-base_sh:>+18.3f}")
    print(f"\n(union pairs-100 reference Sharpe under same machinery = {base_sh:.3f})")


if __name__ == "__main__":
    main()
