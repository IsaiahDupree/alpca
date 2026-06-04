"""
Honest strategy evaluation — "profitable" must mean "beats buy-and-hold, risk-adjusted,
out-of-sample." This harness exists because the discovery sweep showed single-asset
long-biased strategies look great in a bull market while merely capturing BETA. Every
verdict here is relative to actually owning the asset, and split in-sample / out-of-sample.

Outputs an EvalReport with: strategy return/Sharpe/maxDD/exposure, the buy-and-hold
benchmark, excess return, whether it beats B&H on return AND on risk-adjusted return, the
in-sample vs out-of-sample returns, and a one-line verdict.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import List, Optional

_TRADING_DAYS = 252


@dataclass
class EvalReport:
    name: str
    n_bars: int
    periods_per_year: float
    strat_return: float
    strat_sharpe: float
    strat_maxdd: float
    n_trades: int
    exposure: float            # fraction of the window spent in a position
    bh_return: float
    bh_sharpe: float
    bh_maxdd: float
    excess_return: float       # strat_return - bh_return
    beats_return: bool
    beats_sharpe: bool
    is_return: float           # in-sample (first 1-oos_frac)
    oos_return: float          # out-of-sample (last oos_frac)
    bh_oos_return: float
    oos_beats_bh: bool
    verdict: str

    def render(self) -> str:
        L = [f"=== {self.name} — honest evaluation ({self.n_bars} bars) ==="]
        L.append(f"  strategy : ret {self.strat_return:+.1%}  Sharpe {self.strat_sharpe:.2f}  "
                 f"maxDD {self.strat_maxdd:.1%}  trades {self.n_trades}  exposure {self.exposure:.0%}")
        L.append(f"  buy&hold : ret {self.bh_return:+.1%}  Sharpe {self.bh_sharpe:.2f}  maxDD {self.bh_maxdd:.1%}")
        L.append(f"  excess   : {self.excess_return:+.1%} return   "
                 f"beats B&H: return={self.beats_return} sharpe={self.beats_sharpe}")
        L.append(f"  OOS      : in-sample {self.is_return:+.1%}  out-of-sample {self.oos_return:+.1%} "
                 f"(B&H OOS {self.bh_oos_return:+.1%})  OOS beats B&H: {self.oos_beats_bh}")
        L.append(f"  VERDICT  : {self.verdict}")
        return "\n".join(L)


def _returns(equity: List[float]) -> List[float]:
    out = []
    for i in range(1, len(equity)):
        if equity[i - 1] > 0:
            out.append((equity[i] - equity[i - 1]) / equity[i - 1])
    return out


def sharpe_of(equity: List[float], periods_per_year: float) -> float:
    r = _returns(equity)
    if len(r) < 2:
        return 0.0
    sd = statistics.pstdev(r)
    if sd <= 0:
        return 0.0
    return statistics.fmean(r) / sd * math.sqrt(periods_per_year)


def max_drawdown_of(equity: List[float]) -> float:
    peak = equity[0] if equity else 0.0
    dd = 0.0
    for x in equity:
        peak = max(peak, x)
        if peak > 0:
            dd = min(dd, (x - peak) / peak)
    return dd


def infer_periods_per_year(bars: List[dict]) -> float:
    """bars/year from the median bars-per-ET-session-date (so daily->252, 1-min->~98k)."""
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


def buy_and_hold(bars: List[dict], periods_per_year: float, starting_equity: float = 100_000.0) -> _BH:
    closes = [float(b["close"]) for b in bars if b.get("close")]
    if len(closes) < 2 or closes[0] <= 0:
        return _BH(0.0, 0.0, 0.0)
    equity = [starting_equity * (c / closes[0]) for c in closes]
    return _BH((equity[-1] - equity[0]) / equity[0], sharpe_of(equity, periods_per_year),
               max_drawdown_of(equity))


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


def evaluate(strategy_name: str, bars: List[dict], *, periods_per_year: Optional[float] = None,
             oos_frac: float = 0.3, fill_model=None) -> EvalReport:
    """Backtest `strategy_name` and judge it honestly vs buy-and-hold, with an OOS split."""
    from alpca.backtest.runner_backtest import backtest_resting
    from alpca.strategies.registry import make

    ppy = periods_per_year or infer_periods_per_year(bars)
    allow_short = strategy_name.endswith("-ls")

    def run(bs):
        return backtest_resting(make(strategy_name), bs, allow_short=allow_short, fill_model=fill_model)

    full = run(bars)
    s_ret = full.total_return
    s_sharpe = sharpe_of(full.equity_curve, ppy)
    s_dd = getattr(full, "max_drawdown", max_drawdown_of(full.equity_curve))
    bh = buy_and_hold(bars, ppy)

    n = len(bars)
    split = max(2, int(n * (1 - oos_frac)))
    is_ret = run(bars[:split]).total_return if split < n else s_ret
    oos_ret = run(bars[split:]).total_return if split < n else s_ret
    bh_oos = buy_and_hold(bars[split:], ppy).total_return

    beats_return = s_ret > bh.total_return
    beats_sharpe = s_sharpe > bh.sharpe
    oos_beats = oos_ret > bh_oos

    if beats_return and beats_sharpe and oos_beats:
        verdict = "BEATS buy-and-hold on return AND risk-adjusted, out-of-sample — genuine candidate."
    elif beats_sharpe and not beats_return:
        verdict = "Lower return but better risk-adjusted (less drawdown) — RISK-REDUCED exposure, not a market-beater."
    elif beats_return and not oos_beats:
        verdict = "Beats B&H in-sample but NOT out-of-sample — likely regime luck / overfit; do not trust."
    elif beats_return:
        verdict = "Higher return but worse risk-adjusted (more drawdown per unit return)."
    else:
        verdict = "Underperforms buy-and-hold — BETA, no demonstrated edge."

    return EvalReport(
        name=strategy_name, n_bars=n, periods_per_year=ppy,
        strat_return=s_ret, strat_sharpe=s_sharpe, strat_maxdd=s_dd,
        n_trades=getattr(full, "n_trades", 0), exposure=_exposure(getattr(full, "trades", []), bars),
        bh_return=bh.total_return, bh_sharpe=bh.sharpe, bh_maxdd=bh.maxdd,
        excess_return=s_ret - bh.total_return, beats_return=beats_return, beats_sharpe=beats_sharpe,
        is_return=is_ret, oos_return=oos_ret, bh_oos_return=bh_oos, oos_beats_bh=oos_beats,
        verdict=verdict,
    )
