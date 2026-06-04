"""
Phase 2: tcapy-style TCA decomposition in backtest/parity.py.
"""

from alpca.backtest.engine import Trade
from alpca.backtest.parity import decompose_execution, run_parity


def _bar(ts, close, *, bid=None, ask=None, hi=None, lo=None, open_=None):
    return {"open": open_ if open_ is not None else close,
            "high": hi if hi is not None else close + 0.5,
            "low": lo if lo is not None else close - 0.5,
            "close": close, "volume": 1e6, "timestamp": ts, "symbol": "X",
            "bid": bid, "ask": ask}


def test_arrival_and_spread_from_quote():
    # buy filled at 100.05, arrival (ask) = 100.04, bid 99.96 -> mid 100.0
    bars = [_bar(0, 100.0, bid=99.96, ask=100.04, hi=100.10, lo=99.90)]
    trades = [Trade(symbol="X", entry_ts=0, entry_price=100.05, entry_ref=100.0, qty=10)]
    out = decompose_execution(trades, bars)
    assert out["n_fills_analyzed"] == 1
    # arrival slippage = (100.05-100.04)/100.04*1e4 ≈ 1.0 bps
    assert abs(out["arrival_slippage_bps"] - 0.9996) < 0.05
    # spread = (100.04-99.96)/100.0*1e4 = 8.0 bps
    assert abs(out["spread_cost_bps"] - 8.0) < 0.01


def test_best_worst_bracket_realized():
    bars = [_bar(0, 100.0, bid=99.96, ask=100.04, hi=100.50, lo=99.50)]
    trades = [Trade(symbol="X", entry_ts=0, entry_price=100.05, entry_ref=100.0, qty=10)]
    out = decompose_execution(trades, bars)
    assert out["best_slippage_bps"] <= out["arrival_slippage_bps"] <= out["worst_slippage_bps"]


def test_sell_side_arrival_sign():
    # short entry (qty<0): arrival = bid = 100.04; fill 99.95 below bid -> adverse +
    bars = [_bar(0, 100.0, bid=100.04, ask=100.12)]
    trades = [Trade(symbol="X", entry_ts=0, entry_price=99.95, entry_ref=100.0, qty=-10)]
    out = decompose_execution(trades, bars)
    assert out["arrival_slippage_bps"] > 0          # sold below the bid = adverse


def test_transient_vs_permanent_impact():
    # buy at 100.10; price reverts to 100.00 from +1 bar onward
    bars = [_bar(i, 100.10 if i == 0 else 100.00) for i in range(40)]
    trades = [Trade(symbol="X", entry_ts=0, entry_price=100.10, entry_ref=100.10, qty=10)]
    out = decompose_execution(trades, bars)
    # fill above the reverted mid -> positive impact at both horizons
    assert out["transient_impact_bps"] is not None and out["transient_impact_bps"] > 0
    assert out["permanent_impact_bps"] is not None and out["permanent_impact_bps"] > 0


def test_no_quote_falls_back_to_open():
    bars = [_bar(0, 100.0, open_=99.90)]   # no bid/ask
    trades = [Trade(symbol="X", entry_ts=0, entry_price=100.00, entry_ref=100.0, qty=10)]
    out = decompose_execution(trades, bars)
    assert out["spread_cost_bps"] is None              # no quote -> no spread
    assert out["arrival_slippage_bps"] is not None     # arrival = open used


def test_run_parity_populates_tca():
    bars = [_bar(i, 100 + (i % 5) * 0.2, bid=100 + (i % 5) * 0.2 - 0.02,
                 ask=100 + (i % 5) * 0.2 + 0.02) for i in range(60)]
    rep = run_parity("donchian", bars, symbol="X")
    d = rep.to_dict()
    assert "tca" in d
    assert d["tca"]["n_fills_analyzed"] >= 0           # runs end to end, block present
    assert isinstance(rep.render(), str)
