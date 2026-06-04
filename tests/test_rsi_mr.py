"""
RSIMeanReversion (#2) — Wilder-RSI reversion with a volatility-regime gate, plus
the reusable wilder_rsi / rolling_return_vol helpers.
"""

import statistics

from alpca.backtest.engine import run_backtest
from alpca.data.bars import synthetic_bars
from alpca.strategies.base import BUY, EXIT, HOLD, SELL
from alpca.strategies.mean_reversion import (
    RSIMeanReversion,
    rolling_return_vol,
    wilder_rsi,
)
from alpca.strategies.registry import available, make


def _bar(c):
    return {"open": c, "high": c, "low": c, "close": c, "volume": 1000, "timestamp": 0}


# ---- helpers ---------------------------------------------------------------

def test_wilder_rsi_extremes():
    assert wilder_rsi([1, 2, 3, 4, 5], 2) == 100.0          # all gains
    assert wilder_rsi([5, 4, 3, 2, 1], 2) == 0.0            # all losses
    assert wilder_rsi([1, 2], 5) is None                    # warmup


def test_wilder_rsi_midrange():
    rsi = wilder_rsi([100, 101, 100, 101, 100, 101], 2)
    assert rsi is not None and 0.0 < rsi < 100.0


def test_rolling_return_vol():
    assert rolling_return_vol([100, 100, 100, 100], 3) == 0.0   # no moves
    closes = [100, 101, 102, 103]
    rets = [(closes[i] - closes[i - 1]) / closes[i - 1] for i in range(1, 4)]
    assert abs(rolling_return_vol(closes, 3) - statistics.pstdev(rets)) < 1e-12
    assert rolling_return_vol([100, 101], 5) is None


# ---- strategy --------------------------------------------------------------

def _run(strategy, closes):
    sides = []
    for c in closes:
        sides.append(strategy.on_bar(_bar(c)))
    return sides


def test_oversold_enters_long_and_reverts_out():
    s = RSIMeanReversion(rsi_period=2, entry_low=20.0, exit_level=50.0,
                         vol_lookback=3, stop_pct=0.5)
    closes = [100, 100.5, 100, 100.5] + [99, 98, 97, 96, 95, 94] + [99]
    got_buy = got_exit = False
    for c in closes:
        sig = s.on_bar(_bar(c))
        if sig.side == BUY:
            got_buy = True
            assert sig.metadata["rsi"] < 20.0
        if got_buy and sig.side == EXIT:
            got_exit = True
    assert got_buy and got_exit


def test_overbought_shorts_when_allowed():
    s = RSIMeanReversion(rsi_period=2, entry_high=80.0, exit_level=50.0,
                         vol_lookback=3, stop_pct=0.5, allow_short=True)
    closes = [100, 99.5, 100, 99.5] + [101, 102, 103, 104, 105, 106] + [101]
    got_short = got_cover = False
    for c in closes:
        sig = s.on_bar(_bar(c))
        if sig.side == SELL:
            got_short = True
            assert sig.metadata["rsi"] > 80.0
        if got_short and sig.side == EXIT:
            got_cover = True
    assert got_short and got_cover


def test_long_only_never_shorts():
    s = RSIMeanReversion(rsi_period=2, entry_high=80.0, vol_lookback=3,
                         allow_short=False)
    closes = [100, 99.5, 100, 99.5] + [101, 102, 103, 104, 105, 106]
    assert all(sig.side != SELL for sig in _run(s, closes))


def test_vol_gate_blocks_entry_when_out_of_band():
    # a vanishingly small vol_cap means realized vol is never "in band" -> the
    # oversold BUY is suppressed even though RSI crosses entry_low.
    s = RSIMeanReversion(rsi_period=2, entry_low=20.0, vol_lookback=3,
                         vol_floor=0.0, vol_cap=1e-12)
    closes = [100, 100.5, 100, 100.5] + [99, 98, 97, 96, 95, 94]
    assert all(sig.side != BUY for sig in _run(s, closes))


def test_warmup_holds():
    s = RSIMeanReversion(rsi_period=2)
    assert s.on_bar(_bar(100.0)).side == HOLD       # not enough closes for RSI


def test_registry_has_rsi_variants():
    assert {"rsi-mr", "rsi-mr-ls"}.issubset(set(available()))
    assert isinstance(make("rsi-mr"), RSIMeanReversion)
    assert make("rsi-mr").allow_short is False
    assert make("rsi-mr-ls").allow_short is True


def test_rsi_mr_runs_through_backtest():
    bars = synthetic_bars("RSI", n=200, seed=3)
    res = run_backtest(make("rsi-mr", rsi_period=2, vol_lookback=10),
                       bars, slippage_bps=1.0, commission_bps=0.0)
    assert res.starting_equity == 100_000.0
    assert res.ending_equity > 0
    assert res.strategy == "rsi-mr"
