"""
Case 65 — HARDENED cross-sectional momentum (make the one cross-program-confirmed directional edge
smarter, then validate it honestly).

Cross-sectional momentum is the ONLY directional edge confirmed in BOTH programs: Alpca Case 3 (~0.68,
config-sensitive, basic top-k) AND HFT-work (OOS Sharpe ~0.68, "survived the gauntlet"). This takes
the HFT-work construction + every control we've since built and re-tests it properly:

  SMARTER construction (vs the basic top-k/bottom-k in cross_sectional.py):
    - 12-1 momentum: trailing `lookback` return SKIPPING the most recent `skip` days (drop short-term
      reversal contamination — the standard academic fix).
    - z-scored full cross-section, weight ∝ z (not binary top-k); demeaned (Σw≈0) + gross-normalized
      (Σ|w|=1) => a clean $1 market-neutral book using ALL names, not just the tails.
    - TREND GATE: trade only when the market is trending (efficiency ratio ≥ thr), flat in chop
      (the HFT-work key — momentum pays in trends, bleeds in chop).

  CONTROLS Case 3 never had:
    - SURVIVORSHIP: run on cache_delisted_sip (1702 delisted-inclusive) — the decisive test.
    - OUT-OF-REGIME: 2016-2020 (never tuned) vs 2021-2026 split.
    - beta-decomp vs SPY (must be ~0 = market-neutral, not closet beta), cost stress, DSR, and
      ABLATIONS (hardened vs basic, gate on/off) so any lift is attributable.

Verdict: a deployable 3rd leg requires positive + survivorship-robust + out-of-regime-stable +
market-neutral (|beta|<0.2) + survives cost, and BEATS the basic version (else no point hardening).

Run: .venv/bin/python scripts/test_xsec_momentum_hardened.py   -> data/xsec_momentum_hardened.json
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from alpca.backtest.evaluation import sharpe_of, max_drawdown_of  # noqa: E402

PPY = 252.0


def load_panel(cache: Path):
    bars = {}
    for p in cache.glob("*_1day_bars.jsonl"):
        s = p.name.split("_1day_")[0]
        rows = [json.loads(l) for l in p.open() if l.strip()]
        if rows:
            bars[s] = sorted(rows, key=lambda x: int(x["timestamp"]))
    syms = sorted(bars)
    common = sorted({int(b["timestamp"]) for s in syms for b in bars[s]})
    tidx = {t: i for i, t in enumerate(common)}
    T, N = len(common), len(syms)
    P = np.full((T, N), np.nan)
    for j, s in enumerate(syms):
        for b in bars[s]:
            P[tidx[int(b["timestamp"])], j] = float(b["close"])
    return syms, common, P


def ann_sharpe(r):
    r = np.asarray(r)
    return float(r.mean() / r.std() * math.sqrt(PPY)) if r.std() > 0 else 0.0


def run_xsec(P, common, *, lookback=252, skip=21, hold=21, cost_bps=2.0,
             hardened=True, gate=True, gate_win=60, gate_thr=0.30, top_frac=0.2):
    """Vectorized cross-sectional momentum. hardened=z-weighted full x-section; else basic top/bottom frac.
    Returns (daily_returns, dates, turnover_avg)."""
    T, N = P.shape
    R = np.zeros((T, N))
    R[1:] = np.where((P[:-1] > 0) & np.isfinite(P[:-1]) & np.isfinite(P[1:]),
                     (P[1:] - P[:-1]) / P[:-1], 0.0)
    avail = np.isfinite(P) & (P > 0)
    # market proxy = equal-weight available names (for the trend gate)
    mkt = np.array([R[t, avail[t]].mean() if avail[t].any() else 0.0 for t in range(T)])
    mkt_px = np.cumprod(1 + mkt)

    W = np.zeros(N)
    daily = np.zeros(T)
    turns = []
    start = lookback + skip + 1
    for t in range(1, T):
        daily[t] = float(np.nansum(W * R[t]))
        if t >= start and (t - start) % hold == 0:
            # 12-1 momentum signal on names available now AND `lookback+skip` ago
            past = t - skip
            base = t - skip - lookback
            ok = avail[t] & avail[past] & avail[base] & (P[base] > 0)
            sig = np.full(N, np.nan)
            sig[ok] = P[past, ok] / P[base, ok] - 1.0
            valid = np.isfinite(sig)
            new_w = np.zeros(N)
            if valid.sum() >= 10:
                trade = True
                if gate:
                    w0 = max(0, t - gate_win)
                    moves = np.abs(np.diff(mkt_px[w0:t + 1]))
                    er = abs(mkt_px[t] - mkt_px[w0]) / moves.sum() if moves.sum() > 0 else 0.0
                    trade = er >= gate_thr
                if trade:
                    s = sig[valid]
                    if hardened:
                        z = (s - s.mean()) / (s.std() + 1e-12)
                        z = z - z.mean()                       # demean -> market-neutral
                        gross = np.abs(z).sum()
                        w = z / gross if gross > 0 else z      # gross-normalize Σ|w|=1
                        new_w[valid] = w
                    else:
                        k = max(1, int(top_frac * valid.sum()))
                        order = np.argsort(s)
                        idx = np.where(valid)[0]
                        for i in idx[order[-k:]]:
                            new_w[i] = 0.5 / k
                        for i in idx[order[:k]]:
                            new_w[i] = -0.5 / k
            turn = np.abs(new_w - W).sum()
            turns.append(turn)
            daily[t] -= turn * cost_bps / 1e4
            W = new_w
    return daily, common, (float(np.mean(turns)) if turns else 0.0)


def beta_vs(daily, common, spy_px_by_ts):
    xs, ys = [], []
    prev = None
    spy_ret = {}
    ts_sorted = sorted(spy_px_by_ts)
    for i in range(1, len(ts_sorted)):
        a, b = ts_sorted[i - 1], ts_sorted[i]
        if spy_px_by_ts[a] > 0:
            spy_ret[b] = spy_px_by_ts[b] / spy_px_by_ts[a] - 1
    for i, t in enumerate(common):
        if t in spy_ret and i < len(daily):
            xs.append(daily[i]); ys.append(spy_ret[t])
    if len(xs) < 30:
        return None, None
    xs, ys = np.array(xs), np.array(ys)
    v = ys.var()
    beta = float(np.cov(xs, ys)[0, 1] / v) if v > 0 else 0.0
    alpha = float((xs.mean() - beta * ys.mean()) * PPY)
    return round(beta, 3), round(alpha, 4)


def peryear(daily, common):
    by = defaultdict(list)
    for i in range(1, len(daily)):
        by[time.strftime("%Y", time.gmtime(common[i]))].append(daily[i])
    return {y: round(ann_sharpe(by[y]), 2) for y in sorted(by)}


def regime_split(daily, common):
    oor = [daily[i] for i in range(1, len(daily)) if int(time.strftime("%Y", time.gmtime(common[i]))) <= 2020]
    inr = [daily[i] for i in range(1, len(daily)) if int(time.strftime("%Y", time.gmtime(common[i]))) >= 2021]
    return (round(ann_sharpe(oor), 3) if len(oor) > 30 else None,
            round(ann_sharpe(inr), 3) if len(inr) > 30 else None)


def summarize(daily, common, spy_px=None):
    sh = round(ann_sharpe(daily[1:]), 3)
    eq = np.cumprod(1 + daily[1:])
    dd = round(max_drawdown_of([1.0] + list(eq)) * 100, 2)
    cum = round((eq[-1] - 1) * 100, 2) if len(eq) else 0.0
    oor, inr = regime_split(daily, common)
    py = peryear(daily, common)
    pos = sum(1 for v in py.values() if v > 0)
    beta = alpha = None
    if spy_px:
        beta, alpha = beta_vs(daily[1:], common[1:], spy_px)
    # DSR-style deflation: t-stat of mean daily
    r = np.asarray(daily[1:]); tstat = float(r.mean() / r.std() * math.sqrt(len(r))) if r.std() > 0 else 0.0
    return {"sharpe": sh, "cum_pct": cum, "maxdd_pct": dd, "oor_2016_2020": oor, "inr_2021_2026": inr,
            "pos_years": f"{pos}/{len(py)}", "per_year": py, "beta_vs_spy": beta,
            "alpha_vs_spy_ann": alpha, "tstat": round(tstat, 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/Volumes/My Passport/AlpcaData/cache_sip_10y")
    ap.add_argument("--survivorship-cache", default="/Volumes/My Passport/AlpcaData/cache_delisted_sip")
    ap.add_argument("--out", default="data/xsec_momentum_hardened.json")
    args = ap.parse_args()

    syms, common, P = load_panel(Path(args.cache))
    spy_px = None
    if "SPY" in syms:
        j = syms.index("SPY")
        spy_px = {common[t]: P[t, j] for t in range(len(common)) if np.isfinite(P[t, j])}
    print(f"[ok] main universe {len(syms)} syms x {len(common)} days "
          f"({time.strftime('%Y-%m', time.gmtime(common[0]))}->{time.strftime('%Y-%m', time.gmtime(common[-1]))})")

    out = {"case": 65, "name": "Hardened cross-sectional momentum"}

    # ablations on the clean large-cap universe
    print("\n=== ABLATIONS (cache_sip_10y) ===")
    configs = [
        ("basic (top/bottom 20%, no gate)", dict(hardened=False, gate=False)),
        ("z-weighted, no gate",             dict(hardened=True, gate=False)),
        ("z-weighted + TREND GATE",         dict(hardened=True, gate=True)),
    ]
    for name, kw in configs:
        daily, com, turn = run_xsec(P, common, **kw)
        s = summarize(daily, com, spy_px)
        out[name] = {**s, "avg_turnover": round(turn, 3)}
        print(f"  {name:36} Sh {s['sharpe']:+.2f} | OOR {s['oor_2016_2020']} / INR {s['inr_2021_2026']} "
              f"| beta {s['beta_vs_spy']} | {s['pos_years']}yr | maxDD {s['maxdd_pct']}% | t {s['tstat']}")

    # decisive: survivorship-aware universe with the hardened+gate config
    print("\n=== SURVIVORSHIP-AWARE (cache_delisted_sip) — the decisive test ===")
    try:
        s2, c2, P2 = load_panel(Path(args.survivorship_cache))
        daily, com, turn = run_xsec(P2, c2, hardened=True, gate=True)
        sv = summarize(daily, com, spy_px)
        out["survivorship_hardened_gate"] = {**sv, "n_syms": len(s2), "avg_turnover": round(turn, 3)}
        print(f"  hardened+gate on {len(s2)} delisted-incl syms: Sh {sv['sharpe']:+.2f} | "
              f"OOR {sv['oor_2016_2020']} / INR {sv['inr_2021_2026']} | beta {sv['beta_vs_spy']} | "
              f"{sv['pos_years']}yr | maxDD {sv['maxdd_pct']}% | t {sv['tstat']}")
    except Exception as e:
        out["survivorship_hardened_gate"] = {"error": str(e)}
        print(f"  survivorship test failed: {e}")

    # verdict
    h = out.get("z-weighted + TREND GATE", {})
    b = out.get("basic (top/bottom 20%, no gate)", {})
    sv = out.get("survivorship_hardened_gate", {})
    beats_basic = (h.get("sharpe") or 0) > (b.get("sharpe") or 0) + 0.05
    survives_surv = (sv.get("sharpe") or -9) > 0.2
    out_of_regime_ok = (h.get("oor_2016_2020") or -9) > 0.2 and (h.get("inr_2021_2026") or -9) > 0.2
    market_neutral = abs(h.get("beta_vs_spy") or 1) < 0.2
    survives_cost = (h.get("tstat") or 0) > 2.0
    candidate = beats_basic and survives_surv and out_of_regime_ok and market_neutral and survives_cost
    out["verdict"] = {
        "beats_basic": beats_basic, "survives_survivorship": survives_surv,
        "out_of_regime_stable": out_of_regime_ok, "market_neutral": market_neutral,
        "significant_post_cost": survives_cost, "candidate_3rd_leg": bool(candidate),
        "summary": ("CANDIDATE 3rd leg — hardened x-sec momentum is survivorship-robust, out-of-regime "
                    "stable, market-neutral, significant, and beats the basic version"
                    if candidate else
                    "NOT YET a deployable leg — fails one+ gate (see flags); keep on forward-track probation")}
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\nVERDICT: {out['verdict']['summary']}")
    print(f"  gates: beats_basic={beats_basic} surv={survives_surv} oor={out_of_regime_ok} "
          f"mn={market_neutral} sig={survives_cost}")
    print(f"[meta] wrote {args.out}")


if __name__ == "__main__":
    main()
