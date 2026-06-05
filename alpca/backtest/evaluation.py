"""
Honest strategy evaluation — the project's most valuable asset. "Profitable" must mean
"beats buy-and-hold, risk-adjusted, statistically significant, stable across regimes, and
out-of-sample." This harness exists because the discovery sweep showed strategies that look
great in a bull market while merely capturing BETA, and pairs baskets whose in-sample Sharpe
(1.78) evaporated out-of-sample (0.43). Nothing is believed until it clears these bars.

evaluate() -> EvalReport with: return / Sharpe / Sortino / Calmar / vol / maxDD / exposure;
the buy-and-hold benchmark; beta / alpha / information-ratio vs B&H; the Sharpe t-statistic
and p-value (is the edge distinguishable from luck?); per-segment Sharpes (regime stability);
in-sample vs out-of-sample; and a verdict that requires significance + stability, not just a
positive backtest. All dependency-free (no numpy/scipy).
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import List, Optional

_TRADING_DAYS = 252


# ----------------------------------------------------------------- return series
def _returns(equity: List[float]) -> List[float]:
    return [(equity[i] - equity[i - 1]) / equity[i - 1]
            for i in range(1, len(equity)) if equity[i - 1] > 0]


def sharpe_of(equity: List[float], periods_per_year: float) -> float:
    r = _returns(equity)
    if len(r) < 2:
        return 0.0
    sd = statistics.pstdev(r)
    return statistics.fmean(r) / sd * math.sqrt(periods_per_year) if sd > 0 else 0.0


def sortino_of(equity: List[float], periods_per_year: float) -> float:
    r = _returns(equity)
    if len(r) < 2:
        return 0.0
    downside = [min(0.0, x) for x in r]
    dd = math.sqrt(sum(x * x for x in downside) / len(downside))
    return statistics.fmean(r) / dd * math.sqrt(periods_per_year) if dd > 0 else 0.0


def vol_of(equity: List[float], periods_per_year: float) -> float:
    r = _returns(equity)
    return statistics.pstdev(r) * math.sqrt(periods_per_year) if len(r) >= 2 else 0.0


def max_drawdown_of(equity: List[float]) -> float:
    peak = equity[0] if equity else 0.0
    dd = 0.0
    for x in equity:
        peak = max(peak, x)
        if peak > 0:
            dd = min(dd, (x - peak) / peak)
    return dd


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def sharpe_tstat(equity: List[float]) -> float:
    """t-stat that the mean per-period return > 0 == per-period Sharpe * sqrt(n)."""
    r = _returns(equity)
    if len(r) < 3:
        return 0.0
    sd = statistics.pstdev(r)
    return statistics.fmean(r) / sd * math.sqrt(len(r)) if sd > 0 else 0.0


def sharpe_pvalue(equity: List[float]) -> float:
    """Two-sided p-value for 'returns are not just noise' (normal approx, large n)."""
    return 2.0 * (1.0 - _normal_cdf(abs(sharpe_tstat(equity))))


def beta_alpha(strat_eq: List[float], bench_eq: List[float], ppy: float):
    """OLS of strategy returns on benchmark returns -> (beta, annualized alpha)."""
    sr, br = _returns(strat_eq), _returns(bench_eq)
    n = min(len(sr), len(br))
    if n < 3:
        return 0.0, 0.0
    sr, br = sr[-n:], br[-n:]
    mb, ms = statistics.fmean(br), statistics.fmean(sr)
    var = sum((x - mb) ** 2 for x in br)
    if var <= 0:
        return 0.0, ms * ppy
    beta = sum((br[i] - mb) * (sr[i] - ms) for i in range(n)) / var
    return beta, (ms - beta * mb) * ppy


def information_ratio(strat_eq: List[float], bench_eq: List[float], ppy: float) -> float:
    sr, br = _returns(strat_eq), _returns(bench_eq)
    n = min(len(sr), len(br))
    if n < 2:
        return 0.0
    excess = [sr[-n:][i] - br[-n:][i] for i in range(n)]
    sd = statistics.pstdev(excess)
    return statistics.fmean(excess) / sd * math.sqrt(ppy) if sd > 0 else 0.0


def segment_sharpes(equity: List[float], ppy: float, k: int = 4) -> List[float]:
    """Sharpe of each of k equal slices — does the edge hold across sub-periods/regimes?"""
    n = len(equity)
    if n < k * 3:
        return [sharpe_of(equity, ppy)]
    return [round(sharpe_of(equity[s * n // k:(s + 1) * n // k + 1], ppy), 2) for s in range(k)]


def infer_periods_per_year(bars: List[dict]) -> float:
    from alpca.data.calendar import session_date
    counts = {}
    for b in bars:
        ts = float(b.get("timestamp", 0) or 0)
        if ts > 1e8:
            d = session_date(ts)
            counts[d] = counts.get(d, 0) + 1
    if len(counts) >= 2:
        return float(statistics.median(counts.values())) * _TRADING_DAYS
    return float(_TRADING_DAYS)


@dataclass
class _BH:
    total_return: float
    sharpe: float
    maxdd: float
    equity_curve: List[float] = field(default_factory=list)


def buy_and_hold(bars: List[dict], periods_per_year: float, starting_equity: float = 100_000.0) -> _BH:
    closes = [float(b["close"]) for b in bars if b.get("close")]
    if len(closes) < 2 or closes[0] <= 0:
        return _BH(0.0, 0.0, 0.0, [])
    eq = [starting_equity * (c / closes[0]) for c in closes]
    return _BH((eq[-1] - eq[0]) / eq[0], sharpe_of(eq, periods_per_year), max_drawdown_of(eq), eq)


def _exposure(trades, bars: List[dict]) -> float:
    if not bars:
        return 0.0
    ts = [float(b.get("timestamp", 0) or 0) for b in bars]
    span = (ts[-1] - ts[0]) or 1.0
    held = 0.0
    for t in trades:
        ein = getattr(t, "entry_ts", None)
        eout = getattr(t, "exit_ts", None) or ts[-1]
        if ein is not None:
            held += max(0.0, float(eout) - float(ein))
    return min(1.0, held / span) if span > 0 else 0.0


@dataclass
class EvalReport:
    name: str
    n_bars: int
    periods_per_year: float
    strat_return: float
    strat_sharpe: float
    strat_sortino: float
    strat_vol: float
    strat_maxdd: float
    calmar: float
    n_trades: int
    exposure: float
    bh_return: float
    bh_sharpe: float
    bh_maxdd: float
    excess_return: float
    beats_return: bool
    beats_sharpe: bool
    beta: float
    alpha: float
    info_ratio: float
    sharpe_tstat: float
    sharpe_pvalue: float
    significant: bool          # p < 0.05 AND |tstat| > 2
    segment_sharpes: List[float]
    stable: bool               # positive Sharpe in a majority of segments
    is_return: float
    oos_return: float
    bh_oos_return: float
    oos_beats_bh: bool
    verdict: str

    def render(self) -> str:
        L = [f"=== {self.name} — honest evaluation ({self.n_bars} bars) ==="]
        L.append(f"  return {self.strat_return:+.1%}  Sharpe {self.strat_sharpe:.2f}  "
                 f"Sortino {self.strat_sortino:.2f}  Calmar {self.calmar:.2f}  "
                 f"vol {self.strat_vol:.1%}  maxDD {self.strat_maxdd:.1%}  exposure {self.exposure:.0%}")
        L.append(f"  vs B&H: excess {self.excess_return:+.1%}  beta {self.beta:.2f}  "
                 f"alpha {self.alpha:+.1%}  info-ratio {self.info_ratio:.2f}  "
                 f"(B&H ret {self.bh_return:+.1%} Sharpe {self.bh_sharpe:.2f})")
        L.append(f"  significance: Sharpe t-stat {self.sharpe_tstat:.2f}  p={self.sharpe_pvalue:.3f}  "
                 f"significant={self.significant}")
        L.append(f"  stability: segment Sharpes {self.segment_sharpes}  stable={self.stable}")
        L.append(f"  OOS: in-sample {self.is_return:+.1%}  out-of-sample {self.oos_return:+.1%}  "
                 f"OOS beats B&H={self.oos_beats_bh}")
        L.append(f"  VERDICT: {self.verdict}")
        return "\n".join(L)


def evaluate(strategy_name: str, bars: List[dict], *, periods_per_year: Optional[float] = None,
             oos_frac: float = 0.3, fill_model=None) -> EvalReport:
    """Backtest `strategy_name` and judge it honestly: vs buy-and-hold, statistically
    significant, stable across regimes, and out-of-sample."""
    from alpca.backtest.runner_backtest import backtest_resting
    from alpca.strategies.registry import make

    ppy = periods_per_year or infer_periods_per_year(bars)
    allow_short = strategy_name.endswith("-ls")

    def run(bs):
        return backtest_resting(make(strategy_name), bs, allow_short=allow_short, fill_model=fill_model)

    full = run(bars)
    eqc = full.equity_curve
    s_ret = full.total_return
    s_sharpe = sharpe_of(eqc, ppy)
    s_dd = getattr(full, "max_drawdown", max_drawdown_of(eqc))
    years = len(eqc) / ppy if ppy else 1.0
    ann_ret = ((1 + s_ret) ** (1 / years) - 1) if years > 0 and s_ret > -1 else s_ret
    calmar = ann_ret / abs(s_dd) if s_dd < 0 else 0.0

    bh = buy_and_hold(bars, ppy)
    beta, alpha = beta_alpha(eqc, bh.equity_curve, ppy)
    ir = information_ratio(eqc, bh.equity_curve, ppy)
    tstat = sharpe_tstat(eqc)
    pval = sharpe_pvalue(eqc)
    significant = pval < 0.05 and abs(tstat) > 2.0
    segs = segment_sharpes(eqc, ppy, k=4)
    stable = sum(1 for s in segs if s > 0) * 2 >= len(segs)

    n = len(bars)
    split = max(2, int(n * (1 - oos_frac)))
    is_ret = run(bars[:split]).total_return if split < n else s_ret
    oos_ret = run(bars[split:]).total_return if split < n else s_ret
    bh_oos = buy_and_hold(bars[split:], ppy).total_return
    beats_return = s_ret > bh.total_return
    beats_sharpe = s_sharpe > bh.sharpe
    oos_beats = oos_ret > bh_oos

    if beats_sharpe and significant and stable and oos_beats:
        verdict = "GENUINE: beats B&H risk-adjusted, statistically significant, stable, holds OOS."
    elif beats_sharpe and not significant:
        verdict = "Better Sharpe than B&H but NOT statistically significant — could be luck."
    elif beats_sharpe and not stable:
        verdict = "Better Sharpe but UNSTABLE across regimes — regime-dependent, not robust."
    elif beats_sharpe:
        verdict = "Better risk-adjusted than B&H (RISK-REDUCED exposure) but does not beat OOS — not a market-beater."
    elif beats_return and not oos_beats:
        verdict = "Beats B&H in-sample but NOT out-of-sample — overfit / regime luck."
    else:
        verdict = "Underperforms buy-and-hold — BETA, no demonstrated edge."

    return EvalReport(
        name=strategy_name, n_bars=n, periods_per_year=ppy,
        strat_return=s_ret, strat_sharpe=s_sharpe, strat_sortino=sortino_of(eqc, ppy),
        strat_vol=vol_of(eqc, ppy), strat_maxdd=s_dd, calmar=calmar,
        n_trades=getattr(full, "n_trades", 0), exposure=_exposure(getattr(full, "trades", []), bars),
        bh_return=bh.total_return, bh_sharpe=bh.sharpe, bh_maxdd=bh.maxdd,
        excess_return=s_ret - bh.total_return, beats_return=beats_return, beats_sharpe=beats_sharpe,
        beta=beta, alpha=alpha, info_ratio=ir, sharpe_tstat=tstat, sharpe_pvalue=pval,
        significant=significant, segment_sharpes=segs, stable=stable,
        is_return=is_ret, oos_return=oos_ret, bh_oos_return=bh_oos, oos_beats_bh=oos_beats,
        verdict=verdict,
    )
