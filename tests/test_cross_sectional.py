"""
Cross-sectional momentum / relative strength (market-neutral long-short).
"""

import math

from alpca.backtest.cross_sectional import backtest_cross_sectional_momentum

DAY = 86400
T0 = 1_700_000_000


def _bars(closes):
    return [{"open": c, "high": c + 0.5, "low": c - 0.5, "close": c, "volume": 1000,
             "timestamp": T0 + i * DAY} for i, c in enumerate(closes)]


def test_longshort_captures_winner_minus_loser():
    n = 250
    win = _bars([100 * (1.004 ** i) for i in range(n)])     # strong uptrend
    mid = _bars([100 + 0.5 * math.sin(i / 5.0) for i in range(n)])  # flat
    lose = _bars([100 * (0.996 ** i) for i in range(n)])    # downtrend
    res = backtest_cross_sectional_momentum({"WIN": win, "MID": mid, "LOSE": lose},
                                            lookback=20, hold=10, top_k=1, bottom_k=1, cost_bps=0.0)
    assert res.n_rebalances > 0
    assert res.total_return > 0      # long the winner, short the loser -> both legs pay


def test_market_neutral_flag_runs_and_curve_length():
    n = 200
    syms = {f"S{k}": _bars([100 + k + 0.1 * i + math.sin(i / 4.0) for i in range(n)]) for k in range(4)}
    res = backtest_cross_sectional_momentum(syms, lookback=20, hold=10)
    assert res.market_neutral is True
    assert len(res.equity_curve) == res.n_bars


def test_costs_reduce_return():
    n = 250
    syms = {f"S{k}": _bars([100 * (1 + 0.001 * (k - 1)) ** i for i in range(n)]) for k in range(3)}
    free = backtest_cross_sectional_momentum(syms, lookback=20, hold=10, cost_bps=0.0)
    costly = backtest_cross_sectional_momentum(syms, lookback=20, hold=10, cost_bps=50.0)
    assert costly.total_return <= free.total_return


def test_too_few_symbols_is_safe():
    res = backtest_cross_sectional_momentum({"A": _bars([1, 2, 3])}, lookback=20)
    assert res.total_return == 0.0 and res.n_rebalances == 0


def test_reversal_inverts_momentum_on_trending_universe():
    # on a TRENDING cross-section, momentum (long winners) profits while reversal
    # (long losers) loses — confirms the inversion is wired correctly.
    n = 250
    win = _bars([100 * (1.004 ** i) for i in range(n)])
    mid = _bars([100 + 0.5 * math.sin(i / 5.0) for i in range(n)])
    lose = _bars([100 * (0.996 ** i) for i in range(n)])
    syms = {"WIN": win, "MID": mid, "LOSE": lose}
    mom = backtest_cross_sectional_momentum(syms, lookback=20, hold=10, cost_bps=0.0, reverse=False)
    rev = backtest_cross_sectional_momentum(syms, lookback=20, hold=10, cost_bps=0.0, reverse=True)
    assert rev.total_return < mom.total_return


def test_long_only_mode_is_directional():
    # market_neutral=False -> long-only top name (this is BETA, for comparison)
    n = 200
    syms = {f"S{k}": _bars([100 * (1.002 + 0.001 * k) ** i for i in range(n)]) for k in range(3)}
    res = backtest_cross_sectional_momentum(syms, lookback=20, hold=10, market_neutral=False)
    assert res.market_neutral is False
    assert len(res.equity_curve) == res.n_bars
