"""
Portfolio-of-strategies combiner — turn weak-but-uncorrelated legs into a deployable book.

The load-bearing math (Bailey/practitioner): combining k strategies each of Sharpe S with
average pairwise correlation rho, equal-risk-weighted, gives

    S_combined = S * sqrt(k) / sqrt(1 + (k-1)*rho)

So four uncorrelated (rho=0) Sharpe-0.5 legs -> 1.0; at rho=0.3 -> only ~0.69. Correlation
is destiny: stacking correlated betas (momentum/reversal/TSMOM/PCA — all secretly the same
beta) buys NOTHING. The only configuration that helps is genuinely uncorrelated legs:
market-neutral basket + risk-reduced beta + small crypto-long + event-clock seasonality.

This module does NOT invent alpha. It (1) measures the cross-leg correlation matrix (the
metric the whole edifice lives or dies on), (2) blends with the robust default — inverse-vol
weighting + a half-Kelly leverage cap (de Prado: at low N, inverse-vol beats fancy optimizers
OOS), (3) reports the combined Sharpe vs the equal-weight null, and (4) translates Sharpe into
the honest expected ANNUAL and DAILY return so "X% per day" fantasies die on contact.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Dict, List


def combined_sharpe_formula(sharpe: float, k: int, rho: float) -> float:
    """Theoretical combined Sharpe of k equal-risk legs, each Sharpe `sharpe`, avg corr `rho`."""
    if k < 1:
        return 0.0
    denom = math.sqrt(max(1e-9, 1.0 + (k - 1) * rho))
    return sharpe * math.sqrt(k) / denom


def _stats(r: List[float]):
    if len(r) < 2:
        return 0.0, 0.0
    return statistics.fmean(r), statistics.pstdev(r)


def correlation(a: List[float], b: List[float]) -> float:
    n = min(len(a), len(b))
    if n < 3:
        return 0.0
    a, b = a[-n:], b[-n:]
    ma, mb = statistics.fmean(a), statistics.fmean(b)
    sa = statistics.pstdev(a)
    sb = statistics.pstdev(b)
    if sa <= 0 or sb <= 0:
        return 0.0
    cov = sum((a[i] - ma) * (b[i] - mb) for i in range(n)) / n
    return cov / (sa * sb)


def correlation_matrix(streams: Dict[str, List[float]]):
    names = list(streams)
    return names, [[correlation(streams[i], streams[j]) for j in names] for i in names]


def _align_tail(streams: Dict[str, List[float]]):
    """Truncate every stream to the common tail length (the most recent overlap). Valid when
    the legs share a trading calendar; cross-calendar legs (e.g. crypto 365d) are an
    approximation — combine same-calendar legs for a rigorous result."""
    m = min((len(v) for v in streams.values()), default=0)
    return {k: v[-m:] for k, v in streams.items()}, m


def inverse_vol_weights(streams: Dict[str, List[float]]) -> Dict[str, float]:
    """Weights proportional to 1/volatility (equal risk contribution under low correlation)."""
    vols = {k: _stats(v)[1] for k, v in streams.items()}
    inv = {k: (1.0 / s if s > 1e-12 else 0.0) for k, s in vols.items()}
    tot = sum(inv.values()) or 1.0
    return {k: w / tot for k, w in inv.items()}


def blend(streams: Dict[str, List[float]], weights: Dict[str, float]) -> List[float]:
    aligned, m = _align_tail(streams)
    return [sum(weights.get(k, 0.0) * aligned[k][t] for k in aligned) for t in range(m)]


def half_kelly_leverage(sharpe_annual: float, ann_vol: float, fraction: float = 0.5,
                        cap: float = 2.0) -> float:
    """Fractional-Kelly leverage scalar. Full-Kelly leverage for a sleeve ~ Sharpe/vol;
    half-Kelly (fraction=0.5) is the standard haircut for estimation error. Capped."""
    if ann_vol <= 0:
        return 0.0
    return max(0.0, min(cap, fraction * sharpe_annual / ann_vol))


def equity_from_returns(returns: List[float], starting: float = 100_000.0) -> List[float]:
    eq = [starting]
    for r in returns:
        eq.append(eq[-1] * (1 + r))
    return eq


def return_translation(sharpe_annual: float, ann_vol: float, ppy: float = 252.0,
                       risk_free: float = 0.04) -> dict:
    """Translate an annual Sharpe + vol target into honest expected returns. Makes explicit
    that the daily edge is tiny relative to daily noise (why per-day ROI targeting is wrong)."""
    excess_annual = sharpe_annual * ann_vol
    total_annual = excess_annual + risk_free
    daily_excess = excess_annual / ppy
    daily_vol = ann_vol / math.sqrt(ppy)
    return {
        "sharpe_annual": sharpe_annual, "ann_vol": ann_vol,
        "expected_excess_annual": excess_annual, "expected_total_annual": total_annual,
        "expected_daily_excess": daily_excess, "daily_vol": daily_vol,
        "noise_to_edge_ratio": (daily_vol / daily_excess) if daily_excess > 1e-9 else float("inf"),
    }


@dataclass
class ComboReport:
    legs: Dict[str, dict]                       # per-leg sharpe/vol/ann_sharpe
    corr_names: List[str]
    corr_matrix: List[List[float]]
    avg_abs_corr: float
    equalweight_sharpe: float                   # the NULL
    invvol_sharpe: float
    invvol_weights: Dict[str, float]
    n_days: int
    ppy: float
    translation: dict
    invvol_equity: List[float] = field(default_factory=list)


def evaluate_combo(streams: Dict[str, List[float]], *, ppy: float = 252.0,
                   target_vol: float = 0.08, kelly_fraction: float = 0.5) -> ComboReport:
    """Blend `streams` (name -> daily returns) by inverse-vol, compare to the equal-weight
    null, measure the correlation matrix, and translate the combined Sharpe into expected
    annual/daily return. The combiner only 'wins' if invvol_sharpe > equalweight_sharpe AND
    the legs are genuinely low-correlation."""
    from alpca.backtest.evaluation import sharpe_of

    aligned, m = _align_tail(streams)
    legs = {}
    for k, v in aligned.items():
        mean, sd = _stats(v)
        legs[k] = {"sharpe": (mean / sd if sd > 0 else 0.0), "vol": sd,
                   "ann_sharpe": (mean / sd * math.sqrt(ppy) if sd > 0 else 0.0)}

    names, mat = correlation_matrix(aligned)
    offdiag = [abs(mat[i][j]) for i in range(len(names)) for j in range(len(names)) if i != j]
    avg_abs_corr = sum(offdiag) / len(offdiag) if offdiag else 0.0

    ew = {k: 1.0 / len(aligned) for k in aligned}
    ew_eq = equity_from_returns(blend(aligned, ew))
    iv = inverse_vol_weights(aligned)
    iv_ret = blend(aligned, iv)
    iv_eq = equity_from_returns(iv_ret)
    iv_sharpe_ann = sharpe_of(iv_eq, ppy)

    # vol-target the inverse-vol blend with a half-Kelly cap, then translate
    _, sd = _stats(iv_ret)
    realized_vol = sd * math.sqrt(ppy)
    lev = half_kelly_leverage(iv_sharpe_ann, realized_vol, fraction=kelly_fraction)
    applied_vol = min(target_vol, lev * realized_vol) if lev > 0 else target_vol
    translation = return_translation(iv_sharpe_ann, applied_vol, ppy)

    return ComboReport(
        legs=legs, corr_names=names, corr_matrix=mat, avg_abs_corr=avg_abs_corr,
        equalweight_sharpe=sharpe_of(ew_eq, ppy), invvol_sharpe=iv_sharpe_ann,
        invvol_weights=iv, n_days=m, ppy=ppy, translation=translation, invvol_equity=iv_eq)
