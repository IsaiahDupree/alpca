import asyncio
import time

from alpca.data.feed import ReplayBarSource, Tick, parse_ts, _bar_obj_to_dict


def test_parse_ts_variants():
    # epoch seconds passthrough
    assert abs(parse_ts(1_700_000_000) - 1_700_000_000) < 1
    # ns -> s
    assert abs(parse_ts(1_700_000_000_000_000_000) - 1_700_000_000) < 1
    # ms -> s
    assert abs(parse_ts(1_700_000_000_000) - 1_700_000_000) < 1
    # RFC3339
    assert parse_ts("2023-11-14T22:13:20Z") > 1_600_000_000
    assert parse_ts(None) == 0.0


def test_tick_feed_latency():
    t = Tick(symbol="SPY", source_ts=1000.0, recv_ts=1000.025, price=500.0, kind="trade")
    assert abs(t.feed_latency_ms - 25.0) < 1e-6


class _FakeBar:
    def __init__(self, ts):
        self.timestamp = ts
        self.open = self.high = self.low = self.close = 100.0
        self.volume = 1000


def test_bar_obj_to_dict_stamps_recv_ts():
    before = time.time()
    d = _bar_obj_to_dict("SPY", _FakeBar(1_700_000_000))
    assert d["symbol"] == "SPY"
    assert d["timestamp"] == 1_700_000_000
    assert d["recv_ts"] >= before


def test_replay_source_yields_all_and_tracks_latency():
    bars = []
    now = time.time()
    for i in range(5):
        bars.append({"open": 100, "high": 101, "low": 99, "close": 100, "volume": 1,
                     "timestamp": now - 0.05, "recv_ts": now, "symbol": "SPY"})

    async def go():
        src = ReplayBarSource(bars)
        seen = [b async for b in src]
        return seen, src.latency.stats()

    seen, stats = asyncio.run(go())
    assert len(seen) == 5
    assert stats["n"] == 5
    # ~50ms feed latency
    assert 30 < stats["p50_ms"] < 80
