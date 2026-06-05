"""
Cross-sectional momentum / relative strength — the other classic market-NEUTRAL alpha
source. Rank the universe by trailing return, go LONG the strongest names and SHORT the
weakest, dollar-neutral, rebalancing every `hold` bars. Returns come from the spread
between winners and losers, not from market direction — so (like pairs) a bull market
can't flatter it.

Returns-based accounting (weights x per-bar symbol returns), with turnover cost on each
rebalance. Multi-symbol: aligns all symbols on their common timestamps.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class CrossSectionalResult:
    symbols: List[str]
    n_bars: int
    total_return: float
    sharpe: float
    max_drawdown: float
    n_rebalances: int
    lookback: int
    hold: int
    market_neutral: bool
    equity_curve: List[float] = field(default_factory=list)


def backtest_cross_sectional_momentum(
    bars_by_sym: Dict[str, List[dict]], *, lookback: int = 60, hold: int = 20,
    top_k: int = 1, bottom_k: int = 1, cost_bps: float = 2.0,
    periods_per_year: float = 252.0, market_neutral: bool = True, reverse: bool = False,
) -> CrossSectionalResult:
    from alpca.backtest.evaluation import max_drawdown_of, sharpe_of

    syms = sorted(bars_by_sym)
    if len(syms) < (top_k + bottom_k if market_neutral else top_k) + 1:
        return CrossSectionalResult(syms, 0, 0.0, 0.0, 0.0, 0, lookback, hold, market_neutral, [])

    # align on common timestamps
    sets = [set(float(b.get("timestamp", 0) or 0) for b in bars_by_sym[s]) for s in syms]
    common = sorted(set.intersection(*sets)) if sets else []
    if len(common) < lookback + hold + 2:
        return CrossSectionalResult(syms, len(common), 0.0, 0.0, 0.0, 0, lookback, hold, market_neutral, [])

    bymap = {s: {float(b["timestamp"]): float(b["close"]) for b in bars_by_sym[s]} for s in syms}
    price = {s: [bymap[s][t] for t in common] for s in syms}
    n = len(common)

    weights = {s: 0.0 for s in syms}
    eq = 1.0
    equity = [1.0]
    rebals = 0
    for t in range(1, n):
        port_ret = sum(weights[s] * (price[s][t] / price[s][t - 1] - 1.0)
                       for s in syms if price[s][t - 1] > 0)
        eq *= (1.0 + port_ret)
        if t >= lookback and (t - lookback) % hold == 0:
            mom = {s: (price[s][t] / price[s][t - lookback] - 1.0) if price[s][t - lookback] > 0 else 0.0
                   for s in syms}
            # momentum: long winners / short losers. reversal: long losers / short winners
            # (short-horizon mean-reversion anomaly).
            ranked = sorted(syms, key=lambda s: mom[s], reverse=not reverse)
            new_w = {s: 0.0 for s in syms}
            for s in ranked[:top_k]:
                new_w[s] += 0.5 / top_k
            if market_neutral:
                for s in ranked[-bottom_k:]:
                    new_w[s] -= 0.5 / bottom_k
            turnover = sum(abs(new_w[s] - weights[s]) for s in syms)
            eq *= (1.0 - cost_bps / 1e4 * turnover)
            weights = new_w
            rebals += 1
        equity.append(eq)

    return CrossSectionalResult(
        syms, n, equity[-1] - 1.0, sharpe_of(equity, periods_per_year),
        max_drawdown_of(equity), rebals, lookback, hold, market_neutral, equity,
    )
