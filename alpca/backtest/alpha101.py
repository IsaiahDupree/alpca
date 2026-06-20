"""
Alpha101 (Kakushadze 2015, arXiv:1601.00991) — a REPRESENTATIVE, DE-COLLINEARIZED subset of the
published formulaic-alpha library, written as `signal_fn(master, syms, price) -> (T,N)` builders for
the generic cross-sectional engine in `alpca.backtest.factor`.

WHY A SUBSET, NOT ALL 101: the 101 (and the overlapping Alpha158/Qlib158 and GTJA191/Guotai-Junan-191
libraries) are NOT 101/158/191 independent ideas — they are dense re-parameterizations of a handful of
microstructure families. We implement ONE clean representative per family so the test covers the CLASS
rather than 450 near-clones. Family map (and the Alpha158/GTJA191 duplicates each subsumes):

  A. RANK-REVERSAL on returns/close
       Alpha101: #101 is trend; the reversal core is rank(-returns)/rank(close).
       == Alpha158 ROC/RESI buckets, == GTJA191 #1,#9,#42 (rank of -delta close / -returns).
  B. VOLUME-PRICE CORRELATION / COVARIANCE  (open/close/vwap vs volume over a window)
       Alpha101: #2 corr(rank dvol, rank dclose), #4 ts_rank(rank low), #6 corr(open,vol),
                 #12 sign(dvol)*-dclose, #13 cov(rank close, rank vol).
       == Alpha158 CORR/CORD/WVMA buckets, == GTJA191 #6,#11,#13,#40 (price-volume corr).
  C. DECAY-LINEAR / WEIGHTED MOMENTUM  (linearly-weighted trailing window)
       Alpha101: #101-family decay_linear momentum; the wdMA family.
       == Alpha158 WMA/EMA buckets, == GTJA191 #8,#14 (decay-weighted price/return).
  D. HIGH/LOW/CLOSE MICROSTRUCTURE  (intraday range, close vs hi/lo, gaps)
       Alpha101: #53 ((close-low)-(high-close))/(close-low), #54 (low-close)*open^5/...,
                 #101 (close-open)/(high-low).
       == Alpha158 KMID/KLEN/KUP/KLOW/KSFT (the "K-bar" bucket), == GTJA191 #18,#28 (close/range).
  E. RETURNS-BASED VOL/SKEW  (signedpower, stddev, ts_rank of returns)
       Alpha101: #22 -delta(corr(high,vol))*rank(stddev(close)), #40 -rank(stddev(high))*corr,
                 #20-style open-vs-delay gaps.
       == Alpha158 STD/VSTD buckets, == GTJA191 #20,#52 (vol/skew of returns).

NO LOOK-AHEAD: every builder fills sig[t] from windows ending at t (info known AS OF day t); the engine
then holds the book through t+1 from signal[t]. Each builder uses ONLY OHLCV (vwap is proxied by
typical price (H+L+C)/3 — no extra data). signedpower / cross-sectional rank are applied where the
published formula nests them; the engine applies the OUTER cross-sectional rank + long/short itself, so
a builder returns the raw alpha value and the engine ranks it.

The `long_high` flag per factor encodes the published SIGN of the alpha (most reversal alphas are
"long the low value"); the batch script carries it.
"""

from __future__ import annotations

import numpy as np


# ---------- panel loader: build aligned OHLCV(+vwap proxy) once, captured by each builder ----------
def _ohlcv(bars_by_sym, master, syms):
    idx = {t: i for i, t in enumerate(master)}
    T, N = len(master), len(syms)
    O = np.full((T, N), np.nan); H = np.full((T, N), np.nan)
    L = np.full((T, N), np.nan); C = np.full((T, N), np.nan)
    V = np.full((T, N), np.nan)
    for j, s in enumerate(syms):
        for b in bars_by_sym[s]:
            i = idx[int(b["timestamp"])]
            O[i, j] = float(b["open"]); H[i, j] = float(b["high"])
            L[i, j] = float(b["low"]);  C[i, j] = float(b["close"])
            V[i, j] = float(b["volume"])
    with np.errstate(invalid="ignore"):
        VWAP = (H + L + C) / 3.0          # typical-price proxy for vwap (OHLCV-only constraint)
        RET = np.full((T, N), np.nan)
        RET[1:] = np.where(C[:-1] > 0, C[1:] / C[:-1] - 1.0, np.nan)
    return dict(O=O, H=H, L=L, C=C, V=V, VWAP=VWAP, RET=RET)


# ---------- time-series operators (column-wise; all causal: window ends at t) ----------
def _delay(x, d):
    out = np.full_like(x, np.nan)
    if d < x.shape[0]:
        out[d:] = x[:-d]
    return out


def _delta(x, d):
    return x - _delay(x, d)


def _ts_sum(x, d):
    out = np.full_like(x, np.nan)
    for t in range(d - 1, x.shape[0]):
        out[t] = np.nansum(x[t - d + 1:t + 1], axis=0)
    return out


def _ts_mean(x, d):
    out = np.full_like(x, np.nan)
    for t in range(d - 1, x.shape[0]):
        w = x[t - d + 1:t + 1]
        cnt = np.sum(np.isfinite(w), axis=0)
        out[t] = np.where(cnt > 0, np.nansum(w, axis=0) / np.maximum(cnt, 1), np.nan)
    return out


def _ts_std(x, d):
    out = np.full_like(x, np.nan)
    for t in range(d - 1, x.shape[0]):
        out[t] = np.nanstd(x[t - d + 1:t + 1], axis=0)
    return out


def _ts_rank(x, d):
    """fraction of the trailing d-window the current value exceeds (0..1), per column."""
    out = np.full_like(x, np.nan)
    for t in range(d - 1, x.shape[0]):
        w = x[t - d + 1:t + 1]
        cur = x[t]
        out[t] = np.where(np.isfinite(cur), np.sum(w <= cur, axis=0) / float(d), np.nan)
    return out


def _ts_min(x, d):
    out = np.full_like(x, np.nan)
    for t in range(d - 1, x.shape[0]):
        out[t] = np.nanmin(x[t - d + 1:t + 1], axis=0)
    return out


def _ts_max(x, d):
    out = np.full_like(x, np.nan)
    for t in range(d - 1, x.shape[0]):
        out[t] = np.nanmax(x[t - d + 1:t + 1], axis=0)
    return out


def _ts_argmax(x, d):
    out = np.full_like(x, np.nan)
    for t in range(d - 1, x.shape[0]):
        w = x[t - d + 1:t + 1]
        out[t] = np.where(np.all(~np.isfinite(w), axis=0), np.nan, np.nanargmax(w, axis=0))
    return out


def _ts_argmin(x, d):
    out = np.full_like(x, np.nan)
    for t in range(d - 1, x.shape[0]):
        w = x[t - d + 1:t + 1]
        out[t] = np.where(np.all(~np.isfinite(w), axis=0), np.nan, np.nanargmin(w, axis=0))
    return out


def _decay_linear(x, d):
    """linearly-weighted trailing average, MOST-RECENT-day heaviest — Alpha101 decay_linear.
    Window w = x[t-d+1 : t+1] has w[0]=oldest, w[-1]=day t, so weights must INCREASE toward the
    end: wts = [1,2,...,d] (was np.arange(d,0,-1) — an inverted, old-heavy anti-decay; fixed after the
    Case-56 audit caught it. The fix does not change the gauntlet verdict: recent-weighted decay
    momentum is still momentum, already shown to be beta, and decay_mom did not survive either way)."""
    out = np.full_like(x, np.nan)
    wts = np.arange(1, d + 1, dtype=float)
    wts /= wts.sum()
    for t in range(d - 1, x.shape[0]):
        w = x[t - d + 1:t + 1]
        ok = np.isfinite(w)
        wm = np.where(ok, w, 0.0)
        wn = (wts[:, None] * ok).sum(axis=0)
        out[t] = np.where(wn > 0, (wts[:, None] * wm).sum(axis=0) / wn, np.nan)
    return out


def _rolling_corr(a, b, d):
    """per-column Pearson corr over trailing d. Causal."""
    out = np.full(a.shape, np.nan)
    for t in range(d - 1, a.shape[0]):
        aw = a[t - d + 1:t + 1]; bw = b[t - d + 1:t + 1]
        ok = np.isfinite(aw) & np.isfinite(bw)
        for j in range(a.shape[1]):
            m = ok[:, j]
            if m.sum() < max(3, d // 2):
                continue
            x = aw[m, j]; y = bw[m, j]
            sx, sy = x.std(), y.std()
            if sx > 1e-12 and sy > 1e-12:
                out[t, j] = float(np.mean((x - x.mean()) * (y - y.mean())) / (sx * sy))
    return out


def _rolling_cov(a, b, d):
    out = np.full(a.shape, np.nan)
    for t in range(d - 1, a.shape[0]):
        aw = a[t - d + 1:t + 1]; bw = b[t - d + 1:t + 1]
        ok = np.isfinite(aw) & np.isfinite(bw)
        for j in range(a.shape[1]):
            m = ok[:, j]
            if m.sum() < max(3, d // 2):
                continue
            x = aw[m, j]; y = bw[m, j]
            out[t, j] = float(np.mean((x - x.mean()) * (y - y.mean())))
    return out


def _xrank(x):
    """CROSS-SECTIONAL rank per row, scaled to (0,1] — Alpha101 rank() used INSIDE a formula
    (the engine applies the OUTER rank, but nested ranks must be materialized here)."""
    out = np.full(x.shape, np.nan)
    for t in range(x.shape[0]):
        row = x[t]
        ok = np.isfinite(row)
        if ok.sum() < 2:
            continue
        order = np.argsort(np.argsort(np.where(ok, row, np.inf)))
        r = (order + 1).astype(float)
        r[~ok] = np.nan
        out[t] = r / ok.sum()
    return out


def _signedpower(x, p):
    return np.sign(x) * (np.abs(x) ** p)


def _scale(x, a=1.0):
    out = np.full(x.shape, np.nan)
    for t in range(x.shape[0]):
        row = x[t]; ok = np.isfinite(row)
        s = np.nansum(np.abs(row))
        if s > 0:
            out[t] = np.where(ok, row / s * a, np.nan)
    return out


# ---------- the representative alpha builders ----------
# Signature matches factor.py signal builders: builder(bars_by_sym) -> fn(master, syms, price) -> (T,N).
# `price` (close) is passed by the engine but we recompute panels from bars for full OHLCV.

def _mk(bars_by_sym, compute):
    def fn(master, syms, price):
        p = _ohlcv(bars_by_sym, master, syms)
        return compute(p)
    return fn


# --- Family A: rank-reversal ---
def alpha101_reversal_returns(bars_by_sym, d=5):
    """A: short-horizon return reversal. Alpha101 #101-reversal core ~ rank(-sum returns). long_high=False.
    Subsumes GTJA191 #1/#9 (rank -delta close)."""
    return _mk(bars_by_sym, lambda p: _ts_sum(p["RET"], d))


def alpha101_a3_neg_corr_open_vol(bars_by_sym, d=10):
    """A/B: Alpha101 #3 = -correlation(rank(open), rank(volume), 10). long_high=True (alpha already negated)."""
    return _mk(bars_by_sym, lambda p: -_rolling_corr(_xrank(p["O"]), _xrank(p["V"]), d))


# --- Family B: volume-price correlation/covariance ---
def alpha101_a2(bars_by_sym, d=6):
    """B: Alpha101 #2 = -1*correlation(rank(delta(log(volume),2)), rank((close-open)/open), 6). long_high=True."""
    def c(p):
        lv = np.log(np.where(p["V"] > 0, p["V"], np.nan))
        x = _xrank(_delta(lv, 2))
        y = _xrank(np.where(p["O"] > 0, (p["C"] - p["O"]) / p["O"], np.nan))
        return -_rolling_corr(x, y, d)
    return _mk(bars_by_sym, c)


def alpha101_a6(bars_by_sym, d=10):
    """B: Alpha101 #6 = -1*correlation(open, volume, 10). long_high=True."""
    return _mk(bars_by_sym, lambda p: -_rolling_corr(p["O"], p["V"], d))


def alpha101_a13(bars_by_sym, d=5):
    """B: Alpha101 #13 = -1*rank(covariance(rank(close), rank(volume), 5)). long_high=True
    (engine ranks the negated cov; nested rank materialized)."""
    return _mk(bars_by_sym, lambda p: -_rolling_cov(_xrank(p["C"]), _xrank(p["V"]), d))


def alpha101_a16(bars_by_sym, d=5):
    """B: Alpha101 #16 = -1*rank(covariance(rank(high), rank(volume), 5)). long_high=True."""
    return _mk(bars_by_sym, lambda p: -_rolling_cov(_xrank(p["H"]), _xrank(p["V"]), d))


def alpha101_a12(bars_by_sym):
    """B: Alpha101 #12 = sign(delta(volume,1)) * (-1*delta(close,1)). long_high=True."""
    return _mk(bars_by_sym, lambda p: np.sign(_delta(p["V"], 1)) * (-_delta(p["C"], 1)))


# --- Family C: decay-linear / weighted momentum ---
def alpha101_decay_mom(bars_by_sym, d=10):
    """C: decay_linear momentum core (wdMA of returns). long_high=True (trend)."""
    return _mk(bars_by_sym, lambda p: _decay_linear(p["RET"], d))


def alpha101_a8(bars_by_sym, d=5):
    """C: Alpha101 #8 = -1*rank((sum(open,5)*sum(returns,5)) - delay(sum(open,5)*sum(returns,5),10)). long_high=True."""
    def c(p):
        prod = _ts_sum(p["O"], d) * _ts_sum(p["RET"], d)
        return -(prod - _delay(prod, 10))
    return _mk(bars_by_sym, c)


# --- Family D: high/low/close microstructure ---
def alpha101_a53(bars_by_sym, d=9):
    """D: Alpha101 #53 = -1*delta(((close-low)-(high-close))/(close-low), 9). long_high=True."""
    def c(p):
        denom = np.where((p["C"] - p["L"]) != 0, p["C"] - p["L"], np.nan)
        x = ((p["C"] - p["L"]) - (p["H"] - p["C"])) / denom
        return -_delta(x, d)
    return _mk(bars_by_sym, c)


def alpha101_a54(bars_by_sym):
    """D: Alpha101 #54 = -1*((low-close)*open^5) / ((low-high)*close^5). long_high=True."""
    def c(p):
        denom = (p["L"] - p["H"]) * (p["C"] ** 5)
        denom = np.where(denom != 0, denom, np.nan)
        return -((p["L"] - p["C"]) * (p["O"] ** 5)) / denom
    return _mk(bars_by_sym, c)


def alpha101_kbar_close_pos(bars_by_sym):
    """D: K-bar close position in range = (close-low)/(high-low). Alpha158 KMID/KSFT family. long_high=False
    (high close-in-range = bought up intraday -> reverses)."""
    def c(p):
        denom = np.where((p["H"] - p["L"]) > 0, p["H"] - p["L"], np.nan)
        return (p["C"] - p["L"]) / denom
    return _mk(bars_by_sym, c)


def alpha101_a101(bars_by_sym):
    """D: Alpha101 #101 = (close-open)/((high-low)+0.001). Intraday trend strength. long_high=True."""
    return _mk(bars_by_sym, lambda p: (p["C"] - p["O"]) / ((p["H"] - p["L"]) + 0.001))


# --- Family E: returns-based vol/skew ---
def alpha101_a22(bars_by_sym, dc=5, ds=20):
    """E: Alpha101 #22 = -1*(delta(correlation(high,volume,5),5) * rank(stddev(close,20))). long_high=True."""
    def c(p):
        corr = _rolling_corr(p["H"], p["V"], dc)
        return -(_delta(corr, dc) * _xrank(_ts_std(p["C"], ds)))
    return _mk(bars_by_sym, c)


def alpha101_a40(bars_by_sym, ds=10, dc=10):
    """E: Alpha101 #40 = -1*rank(stddev(high,10)) * correlation(high,volume,10). long_high=True."""
    def c(p):
        return -_xrank(_ts_std(p["H"], ds)) * _rolling_corr(p["H"], p["V"], dc)
    return _mk(bars_by_sym, c)


def alpha101_ret_std(bars_by_sym, d=20):
    """E: idiosyncratic-vol-ish: trailing stddev of returns. Alpha158 STD bucket. long_high=False
    (low-vol anomaly: long low-vol, short high-vol)."""
    return _mk(bars_by_sym, lambda p: _ts_std(p["RET"], d))


def alpha101_a20(bars_by_sym):
    """E: Alpha101 #20 = -1*rank(open-delay(high,1)) * rank(open-delay(close,1)) * rank(open-delay(low,1)).
    Overnight-gap reversal. long_high=True (already negated)."""
    def c(p):
        return -(_xrank(p["O"] - _delay(p["H"], 1))
                 * _xrank(p["O"] - _delay(p["C"], 1))
                 * _xrank(p["O"] - _delay(p["L"], 1)))
    return _mk(bars_by_sym, c)


def alpha101_signedpower_ret(bars_by_sym, d=5, pw=2.0):
    """E: signedpower vol-weighted reversal: -signedpower(sum(returns,d), p). long_high=True (negated)."""
    return _mk(bars_by_sym, lambda p: -_signedpower(_ts_sum(p["RET"], d), pw))


# --- extra B/C representatives to broaden cross-section ---
def alpha101_a4(bars_by_sym, d=9):
    """B: Alpha101 #4 = -1*ts_rank(rank(low), 9). long_high=True (negated)."""
    return _mk(bars_by_sym, lambda p: -_ts_rank(_xrank(p["L"]), d))


def alpha101_a9(bars_by_sym):
    """A/D: Alpha101 #9 conditional momentum core ~ delta(close,1) gated by its recent min/max.
    Simplified to ts_argmax(close,5)-based persistence proxy. long_high=True."""
    def c(p):
        dc = _delta(p["C"], 1)
        cond_up = _ts_min(dc, 5) > 0
        cond_dn = _ts_max(dc, 5) < 0
        return np.where(cond_up, dc, np.where(cond_dn, dc, -dc))
    return _mk(bars_by_sym, c)


def alpha101_vwap_close_dev(bars_by_sym, d=5):
    """B/D: deviation of close from trailing vwap-proxy mean (mean-reversion to typical price). long_high=False."""
    def c(p):
        return (p["C"] - _ts_mean(p["VWAP"], d)) / np.where(p["C"] > 0, p["C"], np.nan)
    return _mk(bars_by_sym, c)
