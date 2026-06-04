"""
Offline tests for the quote merge logic that backs the microprice backtest path
(attach_quotes_to_bars). The live fetch_alpaca_quotes hits the network and is
exercised by scripts/download_data.py, not here.
"""

from alpca.data.bars import attach_quotes_to_bars
from alpca.strategies.microstructure import microprice_signal


def _bar(close, ts):
    return {"open": close, "high": close + 0.5, "low": close - 0.5, "close": close,
            "volume": 1e6, "timestamp": ts, "symbol": "SPY"}


def _q(bid, ask, bs, as_, ts):
    return {"bid": bid, "ask": ask, "bid_size": bs, "ask_size": as_, "timestamp": ts, "symbol": "SPY"}


def test_attaches_last_quote_at_or_before_bar():
    bars = [_bar(100, 10), _bar(101, 20), _bar(102, 30)]
    quotes = [_q(99, 101, 9, 1, 5), _q(100, 102, 1, 9, 15), _q(101, 103, 5, 5, 25)]
    merged = attach_quotes_to_bars(bars, quotes)
    # bar@10 -> quote@5 (bull, heavy bid_size); bar@20 -> quote@15 (bear); bar@30 -> quote@25 (flat)
    assert merged[0]["bid"] == 99 and merged[0]["ask"] == 101
    assert microprice_signal(*(merged[0][k] for k in ("bid", "ask", "bid_size", "ask_size"))) == "bull"
    assert merged[1]["quote_ts"] == 15
    assert microprice_signal(*(merged[1][k] for k in ("bid", "ask", "bid_size", "ask_size"))) == "bear"
    assert microprice_signal(*(merged[2][k] for k in ("bid", "ask", "bid_size", "ask_size"))) == "flat"


def test_no_lookahead_bar_before_first_quote_is_unenriched():
    bars = [_bar(100, 1), _bar(101, 50)]
    quotes = [_q(99, 101, 9, 1, 10)]
    merged = attach_quotes_to_bars(bars, quotes)
    assert "bid" not in merged[0]          # no quote existed at or before ts=1
    assert merged[1]["bid"] == 99          # ts=50 picks up the ts=10 quote


def test_empty_quotes_leaves_bars_unchanged():
    bars = [_bar(100, 1), _bar(101, 2)]
    merged = attach_quotes_to_bars(bars, [])
    assert all("bid" not in b for b in merged)
    assert [b["close"] for b in merged] == [100, 101]


def test_max_staleness_drops_cross_day_quotes():
    # bar at ts=100000 with the only quote at ts=10 (way stale) -> not attached
    bars = [_bar(100, 10), _bar(101, 100_000)]
    quotes = [_q(99, 101, 9, 1, 9)]
    merged = attach_quotes_to_bars(bars, quotes, max_staleness_s=60)
    assert merged[0]["bid"] == 99          # ts=10 within 60s of bar ts=10? (10-10=0) yes
    assert "bid" not in merged[1]          # 100000-9 >> 60s -> dropped as stale


def test_inputs_sorted_defensively():
    bars = [_bar(102, 30), _bar(100, 10)]          # out of order
    quotes = [_q(101, 103, 5, 5, 25), _q(99, 101, 9, 1, 5)]
    merged = attach_quotes_to_bars(bars, quotes)
    assert [b["timestamp"] for b in merged] == [10, 30]
    assert merged[0]["quote_ts"] == 5
