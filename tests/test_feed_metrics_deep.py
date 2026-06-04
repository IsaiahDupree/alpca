"""
Deep, deterministic, offline tests for:
  - alpca/data/feed.py    : parse_ts, Tick, _bar_obj_to_dict, _FeedLatencyTracker,
                            ReplayBarSource, QuoteEnrichedBarSource (with a local
                            stub quote provider).
  - alpca/data/bars.py    : attach_quotes_to_bars (two-pointer merge + staleness),
                            synthetic_bars (shape + determinism).
  - alpca/metrics/latency.py : percentile, StageStats.from_values,
                            build_latency_report, LatencyReport.

No network: nothing here calls fetch_alpaca_* or any Alpaca SDK client. The
QuoteEnrichedBarSource is fed a tiny in-process async stub that mimics the real
`async latest()` contract, so the merge logic is exercised end-to-end.

All inputs are explicit / fixed-seed; no wall-clock or unseeded RNG dependence
(parse_ts(time.time()) style calls compare only structural invariants).
"""

from __future__ import annotations

import asyncio
import math
from datetime import datetime, timezone

import pytest

from alpca.data.feed import (
    ReplayBarSource,
    QuoteEnrichedBarSource,
    Tick,
    parse_ts,
    _bar_obj_to_dict,
    _FeedLatencyTracker,
)
from alpca.data.bars import attach_quotes_to_bars, synthetic_bars
from alpca.metrics.latency import (
    percentile,
    StageStats,
    LatencyReport,
    build_latency_report,
    LATENCY_STAGES,
)
from alpca.execution.order import Order, Side, OrderType, Fill, OrderStatus


# --------------------------------------------------------------------------- helpers
def _run(coro):
    """Run an async coroutine to completion on a fresh event loop (deterministic)."""
    return asyncio.run(coro)


async def _drain(source):
    """Collect every item an async-iterable bar source yields, in order."""
    out = []
    async for item in source:
        out.append(item)
    return out


class _FakeBar:
    """A minimal alpaca-py Bar-like object for _bar_obj_to_dict."""

    def __init__(self, ts, o=100.0, h=101.0, lo=99.0, c=100.5, v=1234, symbol="SPY"):
        self.timestamp = ts
        self.open = o
        self.high = h
        self.low = lo
        self.close = c
        self.volume = v
        self.symbol = symbol


class _StubQuoteProvider:
    """Mimics AlpacaQuoteCache.latest(): async, returns a fixed quote dict or None.

    `behavior` is one of: "ok" (return quote), "none" (return None), "raise".
    """

    def __init__(self, quote=None, behavior="ok"):
        self._quote = quote
        self._behavior = behavior
        self.calls = 0

    async def latest(self):
        self.calls += 1
        if self._behavior == "raise":
            raise RuntimeError("simulated quote fetch failure")
        if self._behavior == "none":
            return None
        return dict(self._quote) if self._quote is not None else None


def _bar(ts, symbol="SPY", close=100.0):
    return {
        "open": close, "high": close, "low": close, "close": close,
        "volume": 1000.0, "timestamp": float(ts), "symbol": symbol,
    }


def _quote(ts, bid=99.0, ask=101.0, bs=10.0, as_=12.0, symbol="SPY"):
    return {
        "bid": bid, "ask": ask, "bid_size": bs, "ask_size": as_,
        "timestamp": float(ts), "symbol": symbol,
    }


# ===========================================================================
# parse_ts: ns / us / ms / s / RFC3339 / datetime / degenerate
# ===========================================================================
# Reference instant: 2023-11-14T22:13:20Z == 1_700_000_000 epoch seconds.
_BASE_S = 1_700_000_000


@pytest.mark.parametrize("value,expected", [
    (1_700_000_000_000_000_000, _BASE_S),        # ns  (>1e17) /1e9
    (1_700_000_000_000_000, _BASE_S),            # us  (>1e14) /1e6
    (1_700_000_000_000, _BASE_S),                # ms  (>1e11) /1e3
    (1_700_000_000, _BASE_S),                    # s   passthrough
    (1_700_000_000.5, _BASE_S + 0.5),            # float seconds passthrough
])
def test_parse_ts_numeric_scales(value, expected):
    assert abs(parse_ts(value) - expected) < 1.0


def test_parse_ts_zero_and_none_and_negative():
    assert parse_ts(None) == 0.0
    assert parse_ts(0) == 0.0          # 0 is not > any threshold -> passthrough
    assert parse_ts(0.0) == 0.0
    assert parse_ts(-5) == -5.0        # negatives fall through to passthrough


def test_parse_ts_small_int_passthrough():
    # A tiny int (< 1e11) is treated as raw seconds, not rescaled.
    assert parse_ts(42) == 42.0
    assert parse_ts(1_000_000_000) == 1_000_000_000.0  # ~1e9, still seconds


def test_parse_ts_boundary_thresholds():
    # Just above the ms threshold (1e11) -> divided by 1e3.
    v = 1e11 + 1
    assert parse_ts(v) == pytest.approx(v / 1e3)
    # Just above the us threshold (1e14) -> divided by 1e6.
    v = 1e14 + 1
    assert parse_ts(v) == pytest.approx(v / 1e6)
    # Clearly above the ns threshold (1e17) -> divided by 1e9.
    # (1e17 + 1 is NOT representable as a distinct float, so use 1.1e17.)
    v = 1.1e17
    assert parse_ts(v) == pytest.approx(v / 1e9)


def test_parse_ts_extreme_inf_nan():
    # inf > 1e17 -> inf/1e9 == inf (graceful, no crash).
    assert parse_ts(float("inf")) == float("inf")
    # NaN comparisons are all False -> falls to final passthrough, returns NaN.
    assert math.isnan(parse_ts(float("nan")))


@pytest.mark.parametrize("s,ok", [
    ("2023-11-14T22:13:20Z", True),
    ("2023-11-14T22:13:20+00:00", True),
    ("2023-11-14T22:13:20", True),     # naive -> local tz timestamp, still a float
    ("  2023-11-14T22:13:20Z  ", True),  # stripped
    ("not-a-date", False),             # unparseable -> 0.0
    ("", False),
])
def test_parse_ts_rfc3339_strings(s, ok):
    out = parse_ts(s)
    assert isinstance(out, float)
    if ok:
        assert out > 1_600_000_000
    else:
        assert out == 0.0


def test_parse_ts_z_aware_matches_epoch():
    # The Z-aware RFC3339 string must resolve to the exact UTC epoch.
    assert parse_ts("2023-11-14T22:13:20Z") == pytest.approx(_BASE_S, abs=1.0)


def test_parse_ts_datetime_objects():
    dt = datetime(2023, 11, 14, 22, 13, 20, tzinfo=timezone.utc)
    assert parse_ts(dt) == pytest.approx(_BASE_S, abs=1.0)


def test_parse_ts_object_with_timestamp_method():
    class _HasTs:
        def timestamp(self):
            return 12345.0
    assert parse_ts(_HasTs()) == 12345.0

    class _BadTs:
        def timestamp(self):
            raise ValueError("boom")
    assert parse_ts(_BadTs()) == 0.0


def test_parse_ts_unknown_object_returns_zero():
    assert parse_ts(object()) == 0.0


# ===========================================================================
# Tick.feed_latency_ms
# ===========================================================================
@pytest.mark.parametrize("src,recv,expected_ms", [
    (1000.0, 1000.025, 25.0),
    (1000.0, 1000.0, 0.0),
    (1000.0, 999.990, -10.0),    # negative latency (clock skew) preserved
    (0.0, 1.5, 1500.0),
])
def test_tick_feed_latency(src, recv, expected_ms):
    t = Tick(symbol="SPY", source_ts=src, recv_ts=recv, price=1.0)
    assert t.feed_latency_ms == pytest.approx(expected_ms)


def test_tick_defaults():
    t = Tick(symbol="X", source_ts=1.0, recv_ts=2.0)
    assert t.kind == "trade"
    assert t.price is None and t.bid is None and t.ask is None


# ===========================================================================
# _bar_obj_to_dict
# ===========================================================================
def test_bar_obj_to_dict_fields_and_types():
    b = _FakeBar(ts=_BASE_S, o=1, h=2, lo=0.5, c=1.5, v=999, symbol="ZZZ")
    d = _bar_obj_to_dict("ZZZ", b)
    assert set(d) == {"open", "high", "low", "close", "volume", "timestamp", "recv_ts", "symbol"}
    assert d["open"] == 1.0 and d["high"] == 2.0 and d["low"] == 0.5 and d["close"] == 1.5
    assert d["volume"] == 999.0
    assert d["symbol"] == "ZZZ"
    assert d["timestamp"] == pytest.approx(_BASE_S, abs=1.0)
    # recv_ts is a real wall-clock stamp >= the parsed venue ts here, and a float.
    assert isinstance(d["recv_ts"], float)


def test_bar_obj_to_dict_missing_attrs_default_zero():
    class _Empty:
        pass
    d = _bar_obj_to_dict("SYM", _Empty())
    assert d["open"] == 0.0 and d["close"] == 0.0 and d["volume"] == 0.0
    assert d["timestamp"] == 0.0     # no timestamp attr -> parse_ts(None) -> 0.0
    assert d["symbol"] == "SYM"


def test_bar_obj_to_dict_none_volume_coerced():
    b = _FakeBar(ts=_BASE_S)
    b.volume = None
    d = _bar_obj_to_dict("SPY", b)
    assert d["volume"] == 0.0   # `getattr(...,0.0) or 0.0` collapses None -> 0.0


# ===========================================================================
# _FeedLatencyTracker
# ===========================================================================
def test_feed_latency_tracker_empty():
    t = _FeedLatencyTracker()
    assert t.stats() == {"n": 0}


def test_feed_latency_tracker_records_and_stats():
    t = _FeedLatencyTracker()
    # recv - timestamp in seconds -> *1000 ms. Construct known latencies.
    for recv_minus_ts_ms in [10, 20, 30, 40, 50]:
        t.record({"timestamp": 1000.0, "recv_ts": 1000.0 + recv_minus_ts_ms / 1000.0})
    s = t.stats()
    assert s["n"] == 5
    assert s["mean_ms"] == pytest.approx(30.0)
    assert s["max_ms"] == pytest.approx(50.0)
    # p50: vals[n//2] = vals[2] = 30
    assert s["p50_ms"] == pytest.approx(30.0)
    # p95: index min(n-1, int(0.95*(n-1))) = min(4, int(3.8)) = min(4,3) = 3 -> 40
    assert s["p95_ms"] == pytest.approx(40.0)


def test_feed_latency_tracker_ignores_missing_fields():
    t = _FeedLatencyTracker()
    t.record({"timestamp": 0})          # falsy timestamp -> skipped
    t.record({"recv_ts": 1.0})          # no timestamp -> skipped
    t.record({})                        # nothing -> skipped
    assert t.stats() == {"n": 0}


def test_feed_latency_tracker_window_cap():
    t = _FeedLatencyTracker(window=3)
    # timestamp must be truthy (non-zero) or record() skips the sample.
    for i in range(10):
        t.record({"timestamp": 1000.0, "recv_ts": 1000.0 + float(i) / 1000.0})
    # Only the last 3 samples (i=7,8,9 -> 7,8,9 ms) survive the deque cap.
    s = t.stats()
    assert s["n"] == 3
    assert s["max_ms"] == pytest.approx(9.0)
    assert s["mean_ms"] == pytest.approx(8.0)


# ===========================================================================
# ReplayBarSource — async iteration order + latency recording
# ===========================================================================
def test_replay_preserves_order_and_count():
    bars = [_bar(i, close=100 + i) for i in range(5)]
    src = ReplayBarSource(bars)
    got = _run(_drain(src))
    assert [b["timestamp"] for b in got] == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert [b["close"] for b in got] == [100, 101, 102, 103, 104]
    assert len(got) == 5


def test_replay_empty():
    assert _run(_drain(ReplayBarSource([]))) == []
    assert ReplayBarSource([]).latency.stats() == {"n": 0}


def test_replay_does_not_mutate_input_list():
    bars = [_bar(0), _bar(1)]
    src = ReplayBarSource(bars)
    _run(_drain(src))
    bars.append(_bar(99))
    # ReplayBarSource copied the list at construction; second drain unaffected.
    got = _run(_drain(src))
    assert len(got) == 2


def test_replay_records_feed_latency():
    bars = [
        {"timestamp": 1000.0, "recv_ts": 1000.012, "symbol": "S", "close": 1.0},
        {"timestamp": 1001.0, "recv_ts": 1001.018, "symbol": "S", "close": 1.0},
    ]
    src = ReplayBarSource(bars)
    _run(_drain(src))
    s = src.latency.stats()
    assert s["n"] == 2
    assert s["max_ms"] == pytest.approx(18.0)


def test_replay_is_reiterable():
    # __aiter__ is a fresh generator each call, so the source can be drained twice.
    src = ReplayBarSource([_bar(0), _bar(1)])
    first = _run(_drain(src))
    second = _run(_drain(src))
    assert len(first) == len(second) == 2


# ===========================================================================
# QuoteEnrichedBarSource — merge stub quote onto each bar
# ===========================================================================
def test_quote_enriched_merges_quote_fields():
    bars = [_bar(0, close=100.0), _bar(1, close=101.0)]
    q = {"bid": 99.5, "ask": 100.5, "bid_size": 5.0, "ask_size": 7.0,
         "quote_ts": 0.5, "quote_recv_ts": 0.6}
    src = QuoteEnrichedBarSource(ReplayBarSource(bars), _StubQuoteProvider(q, "ok"))
    got = _run(_drain(src))
    assert len(got) == 2
    for g in got:
        assert g["bid"] == 99.5 and g["ask"] == 100.5
        assert g["bid_size"] == 5.0 and g["ask_size"] == 7.0
        # original bar fields preserved
        assert "close" in g and g["symbol"] == "SPY"


def test_quote_enriched_quote_overrides_overlapping_keys():
    # If the quote dict shares a key with the bar, the quote value wins ({**bar,**q}).
    bars = [{"timestamp": 0.0, "symbol": "SPY", "close": 100.0, "bid": 1.0}]
    q = {"bid": 42.0}
    src = QuoteEnrichedBarSource(ReplayBarSource(bars), _StubQuoteProvider(q, "ok"))
    got = _run(_drain(src))
    assert got[0]["bid"] == 42.0
    assert got[0]["close"] == 100.0


def test_quote_enriched_none_leaves_bar_unchanged():
    bars = [_bar(0), _bar(1)]
    prov = _StubQuoteProvider(behavior="none")
    src = QuoteEnrichedBarSource(ReplayBarSource(bars), prov)
    got = _run(_drain(src))
    assert all("bid" not in g for g in got)
    assert prov.calls == 2   # called once per bar


def test_quote_enriched_exception_is_swallowed():
    bars = [_bar(0), _bar(1), _bar(2)]
    prov = _StubQuoteProvider(behavior="raise")
    src = QuoteEnrichedBarSource(ReplayBarSource(bars), prov)
    got = _run(_drain(src))    # must not raise; stream never stalls
    assert len(got) == 3
    assert all("bid" not in g for g in got)


def test_quote_enriched_empty_source():
    src = QuoteEnrichedBarSource(ReplayBarSource([]), _StubQuoteProvider({"bid": 1.0}, "ok"))
    assert _run(_drain(src)) == []


def test_quote_enriched_latency_proxy():
    inner = ReplayBarSource([_bar(0)])
    src = QuoteEnrichedBarSource(inner, _StubQuoteProvider({"bid": 1.0}, "ok"))
    # latency tracker is borrowed from the wrapped source.
    assert src.latency is inner.latency


# ===========================================================================
# attach_quotes_to_bars — two-pointer merge, staleness, ordering, no look-ahead
# ===========================================================================
def test_attach_basic_last_prior_quote():
    bars = [_bar(10), _bar(20), _bar(30)]
    quotes = [_quote(5, bid=1), _quote(15, bid=2), _quote(25, bid=3)]
    out = attach_quotes_to_bars(bars, quotes)
    # bar@10 -> quote@5 (bid 1); bar@20 -> quote@15 (bid 2); bar@30 -> quote@25 (bid 3)
    assert [b["bid"] for b in out] == [1, 2, 3]
    assert [b["quote_ts"] for b in out] == [5, 15, 25]


def test_attach_quote_exactly_at_bar_ts_is_included():
    # quote ts <= bar ts (inclusive) — a quote stamped at the bar instant attaches.
    bars = [_bar(10)]
    quotes = [_quote(10, bid=7)]
    out = attach_quotes_to_bars(bars, quotes)
    assert out[0]["bid"] == 7
    assert out[0]["quote_ts"] == 10


def test_attach_bars_before_first_quote_unenriched():
    bars = [_bar(1), _bar(2), _bar(50)]
    quotes = [_quote(10, bid=9)]
    out = attach_quotes_to_bars(bars, quotes)
    assert "bid" not in out[0] and "bid" not in out[1]   # before first quote
    assert out[2]["bid"] == 9                            # after the quote


def test_attach_no_lookahead_future_quote_ignored():
    # A quote stamped AFTER the bar must never be attached (no look-ahead).
    bars = [_bar(10)]
    quotes = [_quote(20, bid=123)]
    out = attach_quotes_to_bars(bars, quotes)
    assert "bid" not in out[0]


def test_attach_handles_unsorted_inputs():
    bars = [_bar(30), _bar(10), _bar(20)]
    quotes = [_quote(25, bid=3), _quote(5, bid=1), _quote(15, bid=2)]
    out = attach_quotes_to_bars(bars, quotes)
    # Output is sorted by bar timestamp ascending, then merged correctly.
    assert [b["timestamp"] for b in out] == [10, 20, 30]
    assert [b["bid"] for b in out] == [1, 2, 3]


def test_attach_empty_quotes_returns_bars_unenriched():
    bars = [_bar(1), _bar(2)]
    out = attach_quotes_to_bars(bars, [])
    assert len(out) == 2
    assert all("bid" not in b for b in out)


def test_attach_empty_bars():
    assert attach_quotes_to_bars([], [_quote(1)]) == []


def test_attach_does_not_mutate_input_bars():
    bars = [_bar(10)]
    quotes = [_quote(5, bid=1)]
    out = attach_quotes_to_bars(bars, quotes)
    assert "bid" not in bars[0]      # original untouched
    assert out[0] is not bars[0]     # new dict produced
    assert out[0]["bid"] == 1


@pytest.mark.parametrize("stale,attached", [
    (None, True),     # no limit -> attach regardless of age
    (10.0, True),     # gap is exactly 10s (20-10) -> <= 10 attaches
    (9.0, False),     # gap 10s > 9s -> too stale, not attached
    (0.0, False),     # zero tolerance, 10s gap -> not attached
])
def test_attach_max_staleness(stale, attached):
    bars = [_bar(20)]
    quotes = [_quote(10, bid=55)]
    out = attach_quotes_to_bars(bars, quotes, max_staleness_s=stale)
    if attached:
        assert out[0]["bid"] == 55
    else:
        assert "bid" not in out[0]


def test_attach_staleness_fresh_quote_attaches_old_skipped():
    # Two bars: one with a fresh quote, one only reachable by a too-old quote.
    bars = [_bar(12), _bar(100)]
    quotes = [_quote(10, bid=1)]   # 2s before first bar, 90s before second
    out = attach_quotes_to_bars(bars, quotes, max_staleness_s=5.0)
    assert out[0]["bid"] == 1         # 2s old -> fresh
    assert "bid" not in out[1]        # 90s old -> stale


# ===========================================================================
# synthetic_bars — shape + determinism + invariants
# ===========================================================================
def test_synthetic_bars_shape_and_keys():
    bars = synthetic_bars("ABC", n=10, seed=7)
    assert len(bars) == 10
    for b in bars:
        assert set(b) == {"open", "high", "low", "close", "volume", "timestamp", "symbol"}
        assert b["symbol"] == "ABC"


def test_synthetic_bars_deterministic_same_seed():
    a = synthetic_bars("T", n=20, seed=42)
    b = synthetic_bars("T", n=20, seed=42)
    assert a == b


def test_synthetic_bars_different_seed_differs():
    a = synthetic_bars("T", n=20, seed=1)
    b = synthetic_bars("T", n=20, seed=2)
    assert a != b


def test_synthetic_bars_timestamps_monotonic():
    bars = synthetic_bars("T", n=15, seed=3)
    ts = [b["timestamp"] for b in bars]
    assert ts == [float(i) for i in range(15)]
    assert ts == sorted(ts)


def test_synthetic_bars_ohlc_invariants():
    bars = synthetic_bars("T", n=100, seed=9)
    for b in bars:
        # high is the max of open/close inflated up; low is the min deflated down.
        assert b["high"] >= max(b["open"], b["close"]) - 1e-9
        assert b["low"] <= min(b["open"], b["close"]) + 1e-9
        assert b["high"] >= b["low"]
        assert b["close"] > 0.0
        assert b["volume"] >= 1000.0


def test_synthetic_bars_open_continuity():
    # Each bar's open equals the previous bar's close (rounded), a chained walk.
    bars = synthetic_bars("T", n=30, seed=11)
    for prev, cur in zip(bars, bars[1:]):
        assert cur["open"] == prev["close"]


def test_synthetic_bars_zero_n():
    assert synthetic_bars("T", n=0, seed=0) == []


def test_synthetic_bars_zero_vol_is_pure_drift():
    # vol=0 -> deterministic drift only; close = open*exp(drift) every step,
    # and high==low==max/min(open,close) collapse with no noise.
    bars = synthetic_bars("T", n=5, seed=0, vol=0.0, drift=0.0, start_price=100.0)
    for b in bars:
        # drift 0 + no noise -> flat price
        assert b["open"] == pytest.approx(b["close"])
        assert b["high"] == pytest.approx(b["open"])
        assert b["low"] == pytest.approx(b["open"])


def test_synthetic_bars_price_floor():
    # Even with huge negative drift, price is floored at 0.01 (max(0.01, ...)).
    bars = synthetic_bars("T", n=50, seed=5, drift=-5.0, vol=0.0, start_price=100.0)
    assert all(b["close"] >= 0.01 for b in bars)
    assert all(b["low"] >= 0.0 for b in bars)


# ===========================================================================
# percentile — interpolation, edges
# ===========================================================================
def test_percentile_empty_is_nan():
    assert math.isnan(percentile([], 0.5))


def test_percentile_single_value():
    assert percentile([7.0], 0.0) == 7.0
    assert percentile([7.0], 0.5) == 7.0
    assert percentile([7.0], 1.0) == 7.0


@pytest.mark.parametrize("q,expected", [
    (0.0, 0.0),
    (0.25, 25.0),
    (0.5, 50.0),
    (0.75, 75.0),
    (1.0, 100.0),
])
def test_percentile_linear_interpolation(q, expected):
    # 0..100 in steps of 10 -> 11 points; linear-interp percentile is exact here.
    vals = [float(x) for x in range(0, 101, 10)]
    assert percentile(vals, q) == pytest.approx(expected)


def test_percentile_interpolates_between_points():
    # Two points: at q=0.5 -> midpoint.
    assert percentile([10.0, 20.0], 0.5) == pytest.approx(15.0)
    # at q=0.25 -> 10 + 0.25*(20-10) = 12.5
    assert percentile([10.0, 20.0], 0.25) == pytest.approx(12.5)


def test_percentile_monotonic_in_q():
    vals = sorted([3.0, 1.0, 4.0, 1.5, 5.0, 9.0, 2.0])
    prev = percentile(vals, 0.0)
    for q in [0.1, 0.25, 0.5, 0.75, 0.9, 1.0]:
        cur = percentile(vals, q)
        assert cur >= prev - 1e-9
        prev = cur


# ===========================================================================
# StageStats.from_values — filtering, empties, NaN handling
# ===========================================================================
def test_stagestats_empty_count_zero():
    s = StageStats.from_values("x", [])
    assert s.count == 0 and s.name == "x"
    assert s.mean is None and s.p50 is None and s.max is None


def test_stagestats_filters_none_and_nan():
    s = StageStats.from_values("x", [1.0, None, 2.0, float("nan"), 3.0])
    assert s.count == 3
    assert s.mean == pytest.approx(2.0)
    assert s.min == 1.0 and s.max == 3.0
    assert s.p50 == pytest.approx(2.0)


def test_stagestats_all_none_is_empty():
    s = StageStats.from_values("x", [None, None])
    assert s.count == 0
    assert s.mean is None


def test_stagestats_single_value():
    s = StageStats.from_values("x", [5.0])
    assert s.count == 1
    assert s.mean == s.p50 == s.p95 == s.p99 == s.min == s.max == 5.0


def test_stagestats_sorts_before_percentiles():
    # Unsorted input must be sorted internally; min/max/percentiles correct.
    s = StageStats.from_values("x", [9.0, 1.0, 5.0, 3.0, 7.0])
    assert s.min == 1.0 and s.max == 9.0
    assert s.p50 == pytest.approx(5.0)
    assert s.mean == pytest.approx(5.0)


# ===========================================================================
# Order latency properties (deterministic, fixed timestamps)
# ===========================================================================
def _order_with_lifecycle(signal, submit, ack, fill, *, qty=10.0,
                          intended=100.0, avg=None, side=Side.BUY):
    o = Order(symbol="SPY", side=side, qty=qty, order_type=OrderType.MARKET)
    o.signal_ts = signal
    o.submit_ts = submit
    o.ack_ts = ack
    o.fill_ts = fill
    o.intended_price = intended
    if avg is not None:
        o.avg_fill_price = avg
    return o


def test_order_stage_latencies_ms():
    o = _order_with_lifecycle(100.0, 100.010, 100.030, 100.075)
    assert o.signal_to_submit_ms == pytest.approx(10.0)
    assert o.submit_to_ack_ms == pytest.approx(20.0)
    assert o.ack_to_fill_ms == pytest.approx(45.0)
    assert o.submit_to_fill_ms == pytest.approx(65.0)
    assert o.signal_to_fill_ms == pytest.approx(75.0)


def test_order_latency_none_when_stage_missing():
    o = _order_with_lifecycle(100.0, None, None, None)
    assert o.signal_to_submit_ms is None
    assert o.submit_to_ack_ms is None
    assert o.signal_to_fill_ms is None


def test_order_slippage_buy_positive_when_paid_more():
    # Bought at 101 vs intended 100 -> +100 bps (worse).
    o = _order_with_lifecycle(0, 1, 2, 3, intended=100.0, avg=101.0, side=Side.BUY)
    assert o.slippage_bps == pytest.approx(100.0)


def test_order_slippage_sell_sign_flipped():
    # Sold at 99 vs intended 100 -> received less -> +100 bps (worse for a sell).
    o = _order_with_lifecycle(0, 1, 2, 3, intended=100.0, avg=99.0, side=Side.SELL)
    assert o.slippage_bps == pytest.approx(100.0)


def test_order_slippage_none_without_intended_or_fill():
    assert _order_with_lifecycle(0, 1, 2, 3, intended=None, avg=101.0).slippage_bps is None
    o = _order_with_lifecycle(0, 1, 2, 3, intended=100.0, avg=None)
    assert o.slippage_bps is None
    # intended == 0 guarded -> None
    assert _order_with_lifecycle(0, 1, 2, 3, intended=0.0, avg=1.0).slippage_bps is None


# ===========================================================================
# build_latency_report + LatencyReport
# ===========================================================================
def _filled_order(sig, sub, ack, fil, intended, avg, side=Side.BUY, qty=10.0):
    o = Order(symbol="SPY", side=side, qty=qty)
    o.signal_ts, o.submit_ts, o.ack_ts = sig, sub, ack
    o.intended_price = intended
    o.add_fill(Fill(ts=fil, price=avg, qty=qty))   # sets fill_ts, avg_fill_price, status
    return o


def test_build_report_counts_filled():
    filled = _filled_order(0, 0.01, 0.02, 0.05, 100.0, 100.5)
    pending = Order(symbol="SPY", side=Side.BUY, qty=10.0)  # never filled
    pending.signal_ts = 0.0
    rep = build_latency_report([filled, pending])
    assert rep.n_orders == 2
    assert rep.n_filled == 1
    assert isinstance(rep.stages, list) and len(rep.stages) == len(LATENCY_STAGES)


def test_build_report_stage_names_in_order():
    rep = build_latency_report([_filled_order(0, 0.01, 0.02, 0.05, 100.0, 100.0)])
    names = [s.name for s in rep.stages]
    assert names == [n for (n, _prop) in LATENCY_STAGES]
    assert names == ["signal->submit", "submit->ack", "ack->fill",
                     "submit->fill", "signal->fill"]


def test_build_report_empty():
    rep = build_latency_report([])
    assert rep.n_orders == 0 and rep.n_filled == 0
    assert all(s.count == 0 for s in rep.stages)
    assert rep.slippage_bps.count == 0


def test_build_report_known_percentiles():
    # 3 filled orders with signal->fill of 10ms, 20ms, 30ms.
    orders = [
        _filled_order(0.0, 0.0, 0.0, 0.010, 100.0, 100.0),
        _filled_order(0.0, 0.0, 0.0, 0.020, 100.0, 100.0),
        _filled_order(0.0, 0.0, 0.0, 0.030, 100.0, 100.0),
    ]
    rep = build_latency_report(orders)
    s2f = next(s for s in rep.stages if s.name == "signal->fill")
    assert s2f.count == 3
    assert s2f.min == pytest.approx(10.0)
    assert s2f.max == pytest.approx(30.0)
    assert s2f.p50 == pytest.approx(20.0)
    assert s2f.mean == pytest.approx(20.0)


def test_build_report_slippage_aggregated():
    orders = [
        _filled_order(0, 0, 0, 0.01, 100.0, 101.0, side=Side.BUY),   # +100 bps
        _filled_order(0, 0, 0, 0.01, 100.0, 100.0, side=Side.BUY),   # 0 bps
    ]
    rep = build_latency_report(orders)
    assert rep.slippage_bps.count == 2
    assert rep.slippage_bps.max == pytest.approx(100.0)
    assert rep.slippage_bps.min == pytest.approx(0.0)
    assert rep.slippage_bps.mean == pytest.approx(50.0)


def test_report_to_dict_structure():
    rep = build_latency_report([_filled_order(0, 0.01, 0.02, 0.05, 100.0, 100.5)])
    d = rep.to_dict()
    assert set(d) == {"n_orders", "n_filled", "stages", "slippage_bps"}
    assert isinstance(d["stages"], list)
    assert all("name" in s and "count" in s for s in d["stages"])
    assert "mean" in d["slippage_bps"]


def test_report_render_is_string_with_stage_lines():
    rep = build_latency_report([_filled_order(0, 0.01, 0.02, 0.05, 100.0, 100.5)])
    out = rep.render()
    assert isinstance(out, str)
    assert "Latency report" in out
    assert "signal->fill" in out
    assert "slippage(bps)" in out
    # n/a appears for stages that have no data (this order has all stages, but
    # the header/format still renders); just assert multi-line output.
    assert out.count("\n") >= len(LATENCY_STAGES)


def test_report_render_handles_empty_orders():
    out = build_latency_report([]).render()
    assert "0 orders, 0 filled" in out
    assert "n/a" in out   # empty stages render as n/a
