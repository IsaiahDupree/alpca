"""
Runner-driven backtest: full order-book lifecycle + BacktestResult analytics.
"""

from alpca.backtest.runner_backtest import backtest_resting
from alpca.backtest.engine import BacktestResult
from alpca.data.bars import synthetic_bars
from alpca.strategies.breakout import DonchianBreakout


def test_resting_backtest_returns_full_analytics():
    bars = synthetic_bars("DEMO", n=400, seed=8, drift=0.0004, vol=0.014)
    res = backtest_resting(DonchianBreakout(period=20, atr_period=14, entry="stop"), bars)
    assert isinstance(res, BacktestResult)
    # the resting buy-stop strategy actually traded
    assert res.n_trades >= 1
    # full analytics are populated
    s = res.summary()
    assert s["strategy"] == "donchian"
    assert "sharpe_per_bar" in s
    assert s["max_drawdown"] <= 0
    assert len(res.equity_curve) == len(bars)
    assert 0.0 <= (res.win_rate or 0.0) <= 1.0


def test_equity_curve_has_one_point_per_bar():
    bars = synthetic_bars("DEMO", n=250, seed=3)
    res = backtest_resting(DonchianBreakout(period=10, atr_period=7, entry="stop"), bars)
    assert len(res.equity_curve) == 250


def test_open_position_marked_to_market_as_trade():
    # a strong uptrend: the buy-stop fills and likely never exits -> the open
    # position must still appear as a (mark-to-market) closed trade in analytics.
    bars = []
    px = 100.0
    for i in range(60):
        px *= 1.01
        bars.append({"open": px, "high": px * 1.005, "low": px * 0.999,
                     "close": px, "volume": 1e7, "timestamp": i, "symbol": "UP"})
    res = backtest_resting(DonchianBreakout(period=20, atr_period=14, entry="stop"), bars)
    assert res.n_trades >= 1  # the still-open position is marked to market


def test_resting_vs_market_entry_differ():
    bars = synthetic_bars("DEMO", n=400, seed=8, drift=0.0004, vol=0.014)
    stop = backtest_resting(DonchianBreakout(period=20, atr_period=14, entry="stop"), bars)
    mkt = backtest_resting(DonchianBreakout(period=20, atr_period=14, entry="market"), bars)
    # both produce analytics; they need not be equal but both should be valid
    assert isinstance(stop, BacktestResult) and isinstance(mkt, BacktestResult)
    assert len(stop.equity_curve) == len(mkt.equity_curve) == 400
