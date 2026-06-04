"""
Market-neutral pairs / spread mean-reversion backtester.
"""

import math

import pytest

from alpca.backtest.pairs import align, backtest_pairs, _hedge_ratio

DAY = 86400
T0 = 1_700_000_000


def _bars(closes, start=T0, step=DAY):
    return [{"open": c, "high": c + 0.5, "low": c - 0.5, "close": c, "volume": 1000,
             "timestamp": start + i * step} for i, c in enumerate(closes)]


def test_align_inner_joins_on_timestamp():
    a = _bars([10, 11, 12])
    b = _bars([20, 21, 22])
    rows = align(a, b)
    assert len(rows) == 3
    assert rows[0] == (T0, 10.0, 20.0)


def test_align_drops_unmatched_timestamps():
    a = _bars([10, 11, 12])
    b = [dict(bb, timestamp=bb["timestamp"] + 1) for bb in _bars([20, 21, 22])]  # shifted
    assert align(a, b) == []


def test_hedge_ratio_recovers_slope():
    # log(a) = 2*log(b) + c  -> hedge ~ 2
    lb = [math.log(b) for b in range(10, 60)]
    la = [2 * x + 0.5 for x in lb]
    assert _hedge_ratio(la, lb) == pytest.approx(2.0, rel=1e-6)


def _cointegrated(n=600):
    # a DOMINANT common trend (so the legs are positively correlated, hedge ~1) plus a
    # SMALLER mean-reverting spread deviation -> a real pair.
    common = [100.0 + 25.0 * math.sin(i / 60.0) for i in range(n)]
    dev = [1.5 * math.sin(i / 8.0) for i in range(n)]
    return [common[i] + dev[i] for i in range(n)], [common[i] - dev[i] for i in range(n)]


def test_cointegrated_pair_is_profitable():
    a, b = _cointegrated()
    res = backtest_pairs(_bars(a), _bars(b), lookback=40, entry_z=1.2, exit_z=0.3, cost_bps=0.0)
    assert res.hedge > 0                 # positively-correlated legs
    assert res.n_trades > 0
    assert res.total_return > 0          # the spread edge pays


def test_no_trades_when_series_too_short():
    res = backtest_pairs(_bars([1, 2, 3]), _bars([1, 2, 3]), lookback=60)
    assert res.n_trades == 0 and res.total_return == 0.0


def test_costs_reduce_return():
    a, b = _cointegrated()
    free = backtest_pairs(_bars(a), _bars(b), lookback=40, entry_z=1.2, exit_z=0.3, cost_bps=0.0)
    costly = backtest_pairs(_bars(a), _bars(b), lookback=40, entry_z=1.2, exit_z=0.3, cost_bps=20.0)
    assert costly.total_return < free.total_return
    assert free.n_trades == costly.n_trades   # cost doesn't change the signals


def test_equity_curve_length_matches_aligned_bars():
    a = _bars([100 + math.sin(i) for i in range(200)])
    b = _bars([100 + math.cos(i) for i in range(200)])
    res = backtest_pairs(a, b, lookback=30)
    assert len(res.equity_curve) == 200
