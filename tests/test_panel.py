"""Invariants for the shared panel alignment (alpca/backtest/panel.py)."""

import numpy as np

from alpca.backtest.panel import aligned_returns


def _bars(ts_list, prices):
    return [{"timestamp": t, "close": p} for t, p in zip(ts_list, prices)]


def test_intersects_timestamps():
    bars = {
        "A": _bars([1, 2, 3, 4], [10, 11, 12, 13]),
        "B": _bars([2, 3, 4, 5], [20, 22, 24, 26]),  # overlap ts 2,3,4
    }
    syms, R, ts = aligned_returns(bars)
    assert syms == ["A", "B"]
    assert ts == [3, 4]                 # returns drop the first common bar (2)
    assert R.shape == (2, 2)
    assert np.isfinite(R).all()


def test_min_len_filter_drops_short_symbols():
    bars = {
        "A": _bars(list(range(10)), list(range(100, 110))),
        "B": _bars(list(range(10)), list(range(200, 210))),
        "C": _bars([0, 1], [1, 2]),     # too short
    }
    syms, R, ts = aligned_returns(bars, min_len=5)
    assert "C" not in syms and set(syms) == {"A", "B"}


def test_symbols_subset_restricts():
    bars = {s: _bars(list(range(6)), list(range(10, 16))) for s in ("A", "B", "C")}
    syms, R, ts = aligned_returns(bars, ["A", "C"])
    assert syms == ["A", "C"] and R.shape[1] == 2


def test_degenerate_returns_empty():
    syms, R, ts = aligned_returns({"A": _bars([1, 2, 3], [1, 2, 3])})  # only 1 symbol
    assert syms == [] and R.shape == (0, 0) and ts == []


def test_zero_and_negative_prices_excluded():
    bars = {
        "A": _bars([1, 2, 3], [10, 0, 12]),     # a 0 price drops ts 2 from A's map
        "B": _bars([1, 2, 3], [20, 21, 22]),
    }
    syms, R, ts = aligned_returns(bars)
    assert ts == [3]                  # only ts 1 and 3 are common & positive -> one return
