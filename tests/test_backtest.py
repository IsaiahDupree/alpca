from alpca.backtest.engine import run_backtest
from alpca.strategies.breakout import DonchianBreakout
from alpca.strategies.mean_reversion import ZScoreMeanReversion


def _bar(c, h=None, l=None, ts=0):
    return {"open": c, "high": c if h is None else h, "low": c if l is None else l,
            "close": c, "volume": 1000, "timestamp": ts, "symbol": "TEST"}


def test_donchian_backtest_profitable_uptrend():
    bars = [_bar(100.0, 100.5, 99.5, ts=i) for i in range(7)]
    bars.append(_bar(110.0, 110.0, 108.0, ts=7))  # breakout
    for i, px in enumerate([112, 114, 116, 118, 120], start=8):
        bars.append(_bar(float(px), px + 1, px - 1, ts=i))
    res = run_backtest(DonchianBreakout(period=5, atr_period=3),
                       bars, slippage_bps=2.0, commission_bps=1.0)
    assert res.n_trades >= 1
    assert res.total_return > 0
    assert res.summary()["slippage_bps"] == 2.0  # cost model recorded for live comparison


def test_cost_model_makes_entry_worse_than_ref():
    bars = [_bar(100.0, 100.5, 99.5, ts=i) for i in range(7)]
    bars.append(_bar(110.0, 110.0, 108.0, ts=7))
    bars.append(_bar(111.0, 111.0, 110.0, ts=8))
    res = run_backtest(DonchianBreakout(period=5, atr_period=3),
                       bars, slippage_bps=5.0, commission_bps=0.0)
    t = res.trades[0]
    assert t.entry_price > t.entry_ref
    assert abs((t.entry_price / t.entry_ref - 1) * 10_000 - 5.0) < 1e-6


def test_metrics_present():
    bars = []
    px = 100.0
    for i in range(40):
        px += (1.0 if i % 3 else -1.5)
        bars.append(_bar(px, px + 0.5, px - 0.5, ts=i))
    res = run_backtest(ZScoreMeanReversion(lookback=10), bars, slippage_bps=2.0)
    s = res.summary()
    assert "sharpe_per_bar" in s
    assert "max_drawdown" in s
    assert s["max_drawdown"] <= 0
