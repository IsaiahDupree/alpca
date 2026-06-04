"""
Regression: an offline backtest must NOT be throttled by the live orders/min rate
limiter. That limiter keys on wall-clock time (time.monotonic); a fast replay
submits thousands of orders inside one real minute and, before the fix, tripped
RATE_LIMIT after ~60 orders — silently capping every backtest at ~30 round trips.
backtest_resting now disables the limit in its default risk config.
"""

from alpca.backtest.runner_backtest import backtest_resting
from alpca.strategies.base import Strategy


class _Flipper(Strategy):
    """Enters when flat, exits when in a position — one round trip every 2 bars."""
    name = "flipper"

    def on_bar(self, bar):
        if not self._in_position:
            return self._enter(bar["close"], 1.0, "in")
        return self._exit(bar["close"], "out")


def _bars(n):
    return [{"open": 100, "high": 100.5, "low": 99.5, "close": 100 + (i % 3) * 0.1,
             "volume": 1e7, "timestamp": float(i), "symbol": "X"} for i in range(n)]


def test_backtest_not_capped_by_rate_limit():
    # 600 bars -> ~300 round trips; the old wall-clock rate cap would stop at ~30.
    res = backtest_resting(_Flipper(), _bars(600))
    assert res.n_trades > 100, f"expected >100 trades, got {res.n_trades} (rate-limit cap regressed?)"
