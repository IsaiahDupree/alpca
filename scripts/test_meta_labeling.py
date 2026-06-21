"""
Case 58 — Meta-labeling (Lopez de Prado) on the DEPLOYED cointegrated-pairs basket.

The idea: don't hunt a new edge — try to AMPLIFY the one we've validated (pairs WF ~0.83)
by training a secondary classifier to predict, at entry, whether each pair trade will be
profitable, then SKIP the low-conviction trades. The honest null (stated in RESEARCH_CANDIDATES
#8): filtering an already-thin edge thins it further; estimation noise + fewer trades wash out
any precision gain.

Method (rigorous, no-lookahead):
  1. Replay the deployed-config walk-forward (train=252, test=63, top_n=10, max_adf=-2.86, 2bps)
     on the 195 large-cap daily bars, emitting PER-TRADE records with ENTRY-ONLY features:
       |entry_z|, ADF stat, half-life, |hedge|, lookback, train-spread vol, side, window#.
     Label = trade net return (after cost) > 0.  Also keep each pair's per-bar test returns so
     the basket can be faithfully reconstructed (dropping a trade = that pair sits flat).
  2. Purged, embargoed, EXPANDING-window CV: meta-model (numpy logistic regression) is trained
     only on trades that entered strictly before each OOS fold (minus an embargo) -> OOS P(profit)
     for every trade in the back 60%.
  3. Evaluate on the OOS trades: meta-model AUC, decile mean returns, and a tau sweep comparing
     the meta-FILTERED basket Sharpe vs the unfiltered baseline (same trade slots, dropped trades
     -> pair flat). Plus a SHUFFLE-LABEL PLACEBO: retrain on shuffled labels; a real signal must
     beat its own placebo.

Verdict rule: meta-labeling is a CANDIDATE only if OOS AUC > 0.55 AND the best tau lifts the
basket Sharpe materially over baseline with adequate trade count AND beats the shuffle placebo.
Otherwise REJECT (honest null confirmed).

Run: .venv/bin/python scripts/test_meta_labeling.py [--largecap PATH]
Writes: data/meta_labeling_results.json
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from alpca.backtest.pairs import align, screen_pairs  # noqa: E402
from alpca.backtest.evaluation import sharpe_of  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LC = "/Volumes/My Passport/AlpcaData/cache_largecap_sip"


# ----------------------------- trade-emitting pair backtest -----------------------------
def backtest_pairs_trades(a_bars, b_bars, *, lookback, entry_z=2.0, exit_z=0.5,
                          cost_bps=2.0, hedge, test_len, starting_equity=100_000.0):
    """Mirror of alpca.backtest.pairs.backtest_pairs (validated logic) but emitting, for the
    TEST window only: (a) the per-bar pair return series with dates, and (b) a list of trades
    each tagged with entry-only features + net return. Returns (date_ret_list, trades)."""
    rows = align(a_bars, b_bars)
    if len(rows) < lookback + 5:
        return [], []
    la = [math.log(c) for _, c, _ in rows]
    lb = [math.log(c) for _, _, c in rows]
    h = hedge
    spread = [la[i] - h * lb[i] for i in range(len(rows))]
    z_series = []
    from collections import deque
    win = deque(maxlen=lookback)
    for i in range(len(rows)):
        win.append(spread[i])
        if len(win) >= lookback:
            mu = statistics.fmean(win); sd = statistics.pstdev(win)
            z_series.append((spread[i] - mu) / sd if sd > 0 else 0.0)
        else:
            z_series.append(None)

    cash = starting_equity; qa = qb = 0.0; state = 0
    equity = []
    test_start = len(rows) - test_len            # first index of the test window
    cost = cost_bps / 1e4

    cur = None                                    # open trade dict
    trades = []
    date_ret = []                                 # (epoch, pair_return_this_bar) over test window

    def value(pa, pb):
        return cash + qa * pa + qb * pb

    def open_trade(target, i, pa, pb, z):
        nonlocal cash, qa, qb, state, cur
        eq_before = value(pa, pb)
        cash += qa * pa + qb * pb
        cash -= cost * (abs(qa) * pa + abs(qb) * pb)
        qa = qb = 0.0
        leg = 0.5 * cash
        qa = (leg / pa) * target
        qb = (leg / pb) * (-target)
        cash -= qa * pa + qb * pb
        cash -= cost * (abs(qa) * pa + abs(qb) * pb)
        state = target
        cur = {"entry_i": i, "abs_entry_z": abs(z), "side": target,
               "eq_at_open": eq_before, "dates": []}

    def close_trade(i, pa, pb):
        nonlocal cash, qa, qb, state, cur
        eq_before = value(pa, pb)
        cash += qa * pa + qb * pb
        cash -= cost * (abs(qa) * pa + abs(qb) * pb)
        qa = qb = 0.0
        state = 0
        if cur is not None:
            cur["net_ret"] = (eq_before - cur["eq_at_open"]) / cur["eq_at_open"]
            trades.append(cur)
        cur = None

    prev_eq = None
    for i in range(len(rows)):
        ts, pa, pb = rows[i]
        z = z_series[i]
        if z is not None:
            if state == 0:
                if z > entry_z:
                    open_trade(-1, i, pa, pb, z)
                elif z < -entry_z:
                    open_trade(1, i, pa, pb, z)
            elif state == 1 and z >= -exit_z:
                close_trade(i, pa, pb)
            elif state == -1 and z <= exit_z:
                close_trade(i, pa, pb)
        eq = value(pa, pb)
        if i >= test_start and prev_eq is not None and prev_eq > 0:
            r = (eq - prev_eq) / prev_eq
            date_ret.append((int(ts), r))
            if cur is not None and i > cur["entry_i"]:
                cur["dates"].append(int(ts))
        prev_eq = eq
    # close any open trade at the end so it gets a label
    if cur is not None:
        ts, pa, pb = rows[-1]
        close_trade(len(rows) - 1, pa, pb)
    return date_ret, trades


# ----------------------------- walk-forward replay collecting trades -----------------------------
def collect(bars_by_sym, *, train=252, test=63, top_n=10, max_adf=-2.86, entry_z=2.0):
    syms = sorted(bars_by_sym)
    sets = [set(float(b.get("timestamp", 0) or 0) for b in bars_by_sym[s]) for s in syms]
    common = sorted(set.intersection(*sets)) if sets else []
    n = len(common)
    bymap = {s: {float(b["timestamp"]): b for b in bars_by_sym[s]} for s in syms}
    aligned = {s: [bymap[s][t] for t in common] for s in syms}

    all_trades = []                 # each: features + label + dates + (w, pair_id)
    window_pairs = []               # per window: {pair_id: {epoch: ret}}
    baseline_daily = []             # (epoch, basket_ret) baseline, per the deployed aggregation
    w = 0; wi = 0
    while w + train + test <= n:
        train_slice = {s: aligned[s][w:w + train] for s in syms}
        screened = screen_pairs(syms, train_slice, min_overlap=int(train * 0.8),
                                max_half_life=30.0, min_half_life=3.0, max_adf=max_adf)
        per_pair_series = {}        # pair_id -> {epoch: ret}
        for r in screened[:top_n]:
            seg_a = aligned[r["a"]][w:w + train + test]
            seg_b = aligned[r["b"]][w:w + train + test]
            lb = int(max(20, min(120, r["half_life"] * 3)))
            # train-spread vol (no-lookahead feature)
            tr = align(aligned[r["a"]][w:w + train], aligned[r["b"]][w:w + train])
            tla = [math.log(c) for _, c, _ in tr]; tlb = [math.log(c) for _, _, c in tr]
            tspread = [tla[k] - r["hedge"] * tlb[k] for k in range(len(tr))]
            tvol = statistics.pstdev(tspread) if len(tspread) > 2 else 0.0
            date_ret, trades = backtest_pairs_trades(
                seg_a, seg_b, lookback=lb, entry_z=entry_z, hedge=r["hedge"], test_len=test)
            if not date_ret:
                continue
            pid = f"{r['a']}|{r['b']}|{wi}"
            per_pair_series[pid] = {e: rr for e, rr in date_ret}
            for t in trades:
                if not t.get("dates"):
                    continue
                all_trades.append({
                    "w": wi, "pair": pid,
                    "abs_entry_z": t["abs_entry_z"], "adf": r["adf"], "half_life": r["half_life"],
                    "abs_hedge": abs(r["hedge"]), "lookback": lb, "train_vol": tvol,
                    "side": t["side"], "entry_epoch": min(t["dates"]),
                    "net_ret": t["net_ret"], "label": 1 if t["net_ret"] > 0 else 0,
                    "dates": t["dates"],
                })
        if per_pair_series:
            window_pairs.append(per_pair_series)
            # deployed aggregation: per test-bar, mean across the window's pairs
            all_epochs = sorted({e for d in per_pair_series.values() for e in d})
            for e in all_epochs:
                vals = [d[e] for d in per_pair_series.values() if e in d]
                if vals:
                    baseline_daily.append((e, sum(vals) / len(per_pair_series)))
            wi += 1
        w += test
    return all_trades, window_pairs, baseline_daily


# ----------------------------- numpy logistic regression -----------------------------
def fit_logit(X, y, *, l2=1.0, iters=500, lr=0.3):
    Xb = np.hstack([np.ones((len(X), 1)), X])
    wv = np.zeros(Xb.shape[1])
    m = len(X)
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-Xb @ wv))
        g = Xb.T @ (p - y) / m
        g[1:] += l2 * wv[1:] / m
        wv -= lr * g
    return wv


def predict_logit(wv, X):
    Xb = np.hstack([np.ones((len(X), 1)), X])
    return 1.0 / (1.0 + np.exp(-Xb @ wv))


def auc(y, p):
    pos = p[y == 1]; neg = p[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    # Mann-Whitney U
    allv = np.concatenate([pos, neg])
    order = allv.argsort()
    ranks = np.empty(len(allv)); ranks[order] = np.arange(1, len(allv) + 1)
    r_pos = ranks[:len(pos)].sum()
    u = r_pos - len(pos) * (len(pos) + 1) / 2
    return u / (len(pos) * len(neg))


# ----------------------------- basket Sharpe from a set of kept trades -----------------------------
def basket_sharpe(window_pairs, kept_dates_by_pair):
    """Reconstruct deployed-aggregation daily returns keeping only `kept_dates_by_pair[pair]`
    (a set of epochs) as live; all other pair-days flat (0). Denominator stays = #pairs/window."""
    daily = []
    for pp in window_pairs:
        all_epochs = sorted({e for d in pp.values() for e in d})
        for e in all_epochs:
            s = 0.0
            for pid, d in pp.items():
                if e in d and e in kept_dates_by_pair.get(pid, _ALL):
                    s += d[e]
            daily.append(s / len(pp))
    eq = [1.0]
    for r in daily:
        eq.append(eq[-1] * (1 + r))
    return sharpe_of(eq, 252.0), len(daily), (eq[-1] - 1.0)


class _All:
    def __contains__(self, x):
        return True
_ALL = _All()


# ----------------------------- main -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--largecap", default=DEFAULT_LC)
    ap.add_argument("--out", default="data/meta_labeling_results.json")
    args = ap.parse_args()

    bars = {p.name.split("_1day_")[0]: [json.loads(l) for l in p.open() if l.strip()]
            for p in Path(args.largecap).glob("*_1day_bars.jsonl")}
    print(f"[meta] loaded {len(bars)} large-cap daily series")
    trades, window_pairs, baseline_daily = collect(bars)
    print(f"[meta] collected {len(trades)} pair-trades across {len(window_pairs)} OOS windows")

    if len(trades) < 80:
        print("[meta] too few trades for a meta-model — ABORT")
        return

    trades.sort(key=lambda t: t["entry_epoch"])
    feat_names = ["abs_entry_z", "adf", "half_life", "abs_hedge", "lookback", "train_vol", "side"]
    X = np.array([[t[f] for f in feat_names] for t in trades], dtype=float)
    y = np.array([t["label"] for t in trades], dtype=float)
    base_rate = float(y.mean())

    # purged expanding-window CV over the back 60%; embargo 5 trading days (~5*86400s)
    N = len(trades)
    n0 = int(0.4 * N)
    embargo = 5 * 86400
    epochs = np.array([t["entry_epoch"] for t in trades])
    oos_p = np.full(N, np.nan)
    oos_p_shuf = np.full(N, np.nan)
    rng = np.random.default_rng(7)
    y_shuf = y.copy(); rng.shuffle(y_shuf)

    n_folds = 5
    fold_bounds = [n0 + int((N - n0) * k / n_folds) for k in range(n_folds + 1)]
    # standardize using only the initial-train block stats (refit per fold on its train)
    for k in range(n_folds):
        lo, hi = fold_bounds[k], fold_bounds[k + 1]
        if hi <= lo:
            continue
        fold_start_epoch = epochs[lo]
        train_mask = epochs < (fold_start_epoch - embargo)
        if train_mask.sum() < 40:
            continue
        mu = X[train_mask].mean(0); sd = X[train_mask].std(0); sd[sd == 0] = 1.0
        Xs = (X - mu) / sd
        wv = fit_logit(Xs[train_mask], y[train_mask])
        oos_p[lo:hi] = predict_logit(wv, Xs[lo:hi])
        wv_s = fit_logit(Xs[train_mask], y_shuf[train_mask])
        oos_p_shuf[lo:hi] = predict_logit(wv_s, Xs[lo:hi])

    ev = ~np.isnan(oos_p)
    y_ev = y[ev]; p_ev = oos_p[ev]; ret_ev = np.array([t["net_ret"] for t in trades])[ev]
    a = auc(y_ev, p_ev)
    a_shuf = auc(y[~np.isnan(oos_p_shuf)], oos_p_shuf[~np.isnan(oos_p_shuf)])

    # decile mean net return by predicted prob
    order = p_ev.argsort()
    deciles = []
    for d in range(10):
        idx = order[d * len(order) // 10:(d + 1) * len(order) // 10]
        if len(idx):
            deciles.append(round(float(ret_ev[idx].mean()) * 1e4, 2))  # bps

    # tau sweep: keep trades (in the OOS eval region) with prob >= tau; pre-eval trades kept as-is
    eval_pairs_windows = window_pairs  # full reconstruction; we only filter eval-region trades
    base_sh, base_days, base_ret = basket_sharpe(eval_pairs_windows, {})  # baseline: keep all
    sweep = []
    ev_idx = np.where(ev)[0]
    for tau in [0.45, 0.50, 0.55, 0.60]:
        # build kept_dates_by_pair: drop dates belonging to eval-region trades with prob<tau
        dropped_dates = {}
        n_kept = n_drop = 0
        for j in ev_idx:
            t = trades[j]
            if oos_p[j] >= tau:
                n_kept += 1
            else:
                n_drop += 1
                dropped_dates.setdefault(t["pair"], set()).update(t["dates"])
        # kept = ALL dates except dropped ones, per pair
        kept = {}
        for pp in eval_pairs_windows:
            for pid, d in pp.items():
                dd = dropped_dates.get(pid)
                kept[pid] = (set(d) - dd) if dd else set(d)
        sh, days, tot = basket_sharpe(eval_pairs_windows, kept)
        sweep.append({"tau": tau, "sharpe": round(sh, 3), "kept": n_kept, "dropped": n_drop,
                      "total_ret": round(tot, 4)})

    best = max(sweep, key=lambda s: s["sharpe"])
    candidate = (a > 0.55 and best["sharpe"] > base_sh + 0.10 and a > a_shuf + 0.05
                 and best["kept"] > 0.4 * len(ev_idx))
    verdict = ("CANDIDATE — meta-model adds OOS information and lifts the basket"
               if candidate else
               "REJECT — meta-model has no usable OOS edge; filtering does not lift the basket (honest null confirmed)")

    out = {
        "case": 58, "name": "Meta-labeling on the deployed pairs basket",
        "n_trades": N, "n_windows": len(window_pairs), "base_rate_win": round(base_rate, 3),
        "oos_eval_trades": int(ev.sum()),
        "meta_auc_oos": round(float(a), 3), "shuffle_auc_oos": round(float(a_shuf), 3),
        "decile_mean_ret_bps": deciles,
        "baseline_basket_sharpe": round(base_sh, 3), "baseline_days": base_days,
        "tau_sweep": sweep, "best": best, "verdict": verdict, "candidate": bool(candidate),
        "features": feat_names,
    }
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(json.dumps({k: out[k] for k in
          ["meta_auc_oos", "shuffle_auc_oos", "baseline_basket_sharpe", "best", "verdict"]}, indent=2))
    print(f"[meta] wrote {args.out}")


if __name__ == "__main__":
    main()
