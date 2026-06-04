"""
L1 Order-Flow-Imbalance strategy + the pure ofi_event increment.
"""

from alpca.backtest.runner_backtest import backtest_resting
from alpca.strategies.base import BUY, EXIT, HOLD, SELL
from alpca.strategies.order_flow import L1OFI, ofi_event
from alpca.strategies.registry import available, make


def _qbar(close, bid, ask, bs, az, ts=0.0):
    return {"open": close, "high": close + 0.2, "low": close - 0.2, "close": close,
            "volume": 1e6, "timestamp": ts, "symbol": "X",
            "bid": bid, "ask": ask, "bid_size": bs, "ask_size": az}


# ---- pure increment --------------------------------------------------------

def test_ofi_event_bid_up_is_bullish():
    # bid price rose -> ΔW = +new bid_size; ask unchanged same size -> ΔV = 0
    assert ofi_event(99.01, 50, 101, 100, 99.00, 100, 101, 100) == 50


def test_ofi_event_ask_up_is_bullish():
    # ask price rose (supply retreated) -> ΔV = -prev_ask_size -> OFI positive
    assert ofi_event(99, 100, 101.01, 100, 99, 100, 101.00, 100) == 100


def test_ofi_event_ask_down_is_bearish():
    # ask price fell (aggressive selling) -> ΔV = +ask_size -> OFI negative
    assert ofi_event(99, 100, 100.99, 100, 99, 100, 101.00, 100) == -100


def test_ofi_event_bid_down_is_bearish():
    assert ofi_event(98.99, 100, 101, 100, 99.00, 100, 101, 100) == -100


def test_ofi_event_same_prices_uses_size_delta():
    # both prices unchanged: ΔW = Δbid_size, ΔV = Δask_size -> OFI = ΔW - ΔV
    # bid_size +30, ask_size +10 -> 30 - 10 = 20
    assert ofi_event(99, 130, 101, 110, 99, 100, 101, 100) == 20


# ---- strategy --------------------------------------------------------------

def test_persistent_buy_pressure_enters_long_then_exits():
    s = L1OFI(window=5, entry=0.1, exit=0.02)
    got_buy = got_exit = False
    # 6 bars of rising bids (strong +OFI), then flat quotes so OFI decays out
    seq = [(99.0, 200, 101.0, 100), (99.1, 200, 101.0, 100), (99.2, 200, 101.0, 100),
           (99.3, 200, 101.0, 100), (99.4, 200, 101.0, 100), (99.5, 200, 101.0, 100)]
    for i, (bid, bs, ask, az) in enumerate(seq):
        sig = s.on_bar(_qbar(100 + i * 0.05, bid, ask, bs, az, ts=i))
        if sig.side == BUY:
            got_buy = True
            assert sig.metadata["ofi"] > 0
    # now stable book (no new imbalance) -> normalized OFI decays below exit -> EXIT
    for j in range(6):
        sig = s.on_bar(_qbar(100.3, 99.5, 101.0, 100, 100, ts=10 + j))
        if got_buy and sig.side == EXIT:
            got_exit = True
    assert got_buy and got_exit


def test_short_on_sell_pressure_when_allowed():
    s = L1OFI(window=5, entry=0.1, exit=0.02, allow_short=True)
    got_short = False
    seq = [(99.0, 100, 101.0, 200), (99.0, 100, 100.9, 200), (99.0, 100, 100.8, 200),
           (99.0, 100, 100.7, 200), (99.0, 100, 100.6, 200), (99.0, 100, 100.5, 200)]
    for i, (bid, bs, ask, az) in enumerate(seq):
        sig = s.on_bar(_qbar(100 - i * 0.05, bid, ask, bs, az, ts=i))
        if sig.side == SELL:
            got_short = True
            assert sig.metadata["ofi"] < 0
    assert got_short


def test_missing_quote_holds():
    s = L1OFI(window=5)
    assert s.on_bar({"open": 100, "high": 100, "low": 100, "close": 100,
                     "volume": 1, "timestamp": 0}).side == HOLD


def test_registry_has_ofi():
    assert {"ofi", "ofi-ls"}.issubset(set(available()))
    assert isinstance(make("ofi"), L1OFI)
    assert make("ofi-ls").allow_short is True


def test_ofi_runs_through_runner():
    bars = []
    for i in range(40):
        # alternate buy-pressure and neutral blocks so it enters and exits
        if (i // 8) % 2 == 0:
            bars.append(_qbar(100 + i * 0.02, 99 + i * 0.01, 101, 200, 100, ts=float(i)))
        else:
            bars.append(_qbar(100 + i * 0.02, 99.0, 101.0, 100, 100, ts=float(i)))
    res = backtest_resting(make("ofi", window=5, entry=0.1, exit=0.02), bars)
    assert res.ending_equity > 0
