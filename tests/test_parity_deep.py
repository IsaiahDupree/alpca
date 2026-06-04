"""
Deep, deterministic tests for alpca/backtest/parity.py.

Covers:
  - decompose_execution: arrival / spread / best-worst bracket / transient+1 /
    permanent+30 across crafted Trades + bars (buys, shorts, quotes-present vs
    open-fallback, custom offsets, degenerate/edge inputs).
  - ParityReport: derived properties (return_gap, slippage_gap_bps), to_dict
    structure + tca block + rounding, render() text.
  - run_parity / backtest_resting smoke on synthetic bars (allow_short,
    rate-limit disabled), all fully offline (SimAdapter, no network).

Pure/offline only. No mocks, no network, no RNG without a fixed seed.
"""

from __future__ import annotations

import math

import pytest

from alpca.backtest.engine import BacktestResult, Trade
from alpca.backtest.parity import (
    ParityReport,
    decompose_execution,
    run_parity,
)
from alpca.backtest.runner_backtest import backtest_resting
from alpca.execution.fills import FillModel
from alpca.strategies.registry import make


# --------------------------------------------------------------------------
# tiny self-contained helpers (no imports from other tests/ files)
# --------------------------------------------------------------------------
def _bar(ts, close, *, bid=None, ask=None, hi=None, lo=None, open_=None):
    """One OHLCV(+optional NBBO) bar dict, the shape parity.py consumes."""
    return {
        "open": open_ if open_ is not None else close,
        "high": hi if hi is not None else close + 0.5,
        "low": lo if lo is not None else close - 0.5,
        "close": close,
        "volume": 1e6,
        "timestamp": float(ts),
        "symbol": "X",
        "bid": bid,
        "ask": ask,
    }


def _trade(ts, fill, qty, *, ref=None):
    return Trade(
        symbol="X",
        entry_ts=float(ts),
        entry_price=fill,
        entry_ref=ref if ref is not None else fill,
        qty=qty,
    )


def _sine_bars(n=80, *, mid=100.0, amp=10.0, period=6.0, quotes=False):
    """Deterministic oscillating series (no RNG). Triggers donchian trades."""
    out = []
    for i in range(n):
        px = mid + amp * math.sin(i / period)
        bid = px - 0.02 if quotes else None
        ask = px + 0.02 if quotes else None
        out.append(_bar(i, round(px, 4), bid=bid, ask=ask, hi=px + 0.3, lo=px - 0.3))
    return out


def _full_report(**over):
    """A fully-populated ParityReport with overridable fields."""
    base = dict(
        strategy="s",
        symbol="Y",
        n_bars=3,
        bt_total_return=0.10,
        bt_n_trades=2,
        bt_slippage_bps=2.0,
        live_total_return=0.07,
        live_entries=2,
        live_fills=2,
        live_rejects=0,
        live_realized_slippage_mean_bps=3.456,
        live_realized_slippage_p95_bps=5.6789,
        signal_to_fill_p50_ms=12.34,
        signal_to_fill_p95_ms=99.99,
        arrival_slippage_bps=1.2345,
        spread_cost_bps=8.0,
        best_slippage_bps=-1.0,
        worst_slippage_bps=2.0,
        transient_impact_bps=0.5,
        permanent_impact_bps=0.25,
        n_fills_analyzed=2,
    )
    base.update(over)
    return ParityReport(**base)


# ==========================================================================
# decompose_execution — arrival slippage (quote present)
# ==========================================================================
@pytest.mark.parametrize(
    "fill,qty,bid,ask,expected_bps",
    [
        # buy: arrival = ask. fill above ask = adverse (positive).
        (100.05, 10, 99.96, 100.04, (100.05 - 100.04) / 100.04 * 1e4),
        # buy filled exactly at ask -> zero arrival slippage
        (100.04, 10, 99.96, 100.04, 0.0),
        # buy filled BELOW ask (price improvement) -> negative (favorable)
        (100.00, 10, 99.96, 100.04, (100.00 - 100.04) / 100.04 * 1e4),
        # short (qty<0): arrival = bid. sold below bid = adverse (positive).
        (99.95, -10, 100.04, 100.12, (99.95 - 100.04) / 100.04 * -1.0 * 1e4),
        # short filled exactly at bid -> zero
        (100.04, -10, 100.04, 100.12, 0.0),
        # short filled ABOVE bid (improvement) -> negative
        (100.10, -10, 100.04, 100.12, (100.10 - 100.04) / 100.04 * -1.0 * 1e4),
    ],
)
def test_arrival_slippage_quote(fill, qty, bid, ask, expected_bps):
    bars = [_bar(0, 100.0, bid=bid, ask=ask)]
    out = decompose_execution([_trade(0, fill, qty)], bars)
    assert out["n_fills_analyzed"] == 1
    assert out["arrival_slippage_bps"] == pytest.approx(expected_bps, abs=1e-9)


def test_spread_cost_exact():
    # spread = (ask-bid)/mid*1e4 ; mid = 100.0 -> 8.0 bps
    bars = [_bar(0, 100.0, bid=99.96, ask=100.04)]
    out = decompose_execution([_trade(0, 100.05, 10)], bars)
    assert out["spread_cost_bps"] == pytest.approx(8.0, abs=1e-9)


def test_short_arrival_sign_is_adverse():
    bars = [_bar(0, 100.0, bid=100.04, ask=100.12)]
    out = decompose_execution([_trade(0, 99.95, -10)], bars)
    assert out["arrival_slippage_bps"] > 0  # sold below bid = adverse cost


# ==========================================================================
# decompose_execution — open fallback (no usable quote)
# ==========================================================================
@pytest.mark.parametrize(
    "bid,ask",
    [
        (None, None),       # no quote at all
        (99.96, None),      # half quote
        (None, 100.04),     # half quote
        (100.04, 100.04),   # ask == bid (not ask > bid) -> degenerate
        (100.10, 100.00),   # crossed book ask < bid -> degenerate
    ],
)
def test_no_usable_quote_falls_back_to_open(bid, ask):
    # open = 99.90; buy fill 100.00 -> arrival from open
    bars = [_bar(0, 100.0, open_=99.90, bid=bid, ask=ask)]
    out = decompose_execution([_trade(0, 100.00, 10)], bars)
    assert out["spread_cost_bps"] is None  # spread only when ask > bid
    assert out["arrival_slippage_bps"] == pytest.approx(
        (100.00 - 99.90) / 99.90 * 1e4, abs=1e-9
    )


def test_open_fallback_zero_when_open_missing_uses_fill():
    # no quote, open falsy (0) -> arrival falls back to fill -> arrival==fill -> 0
    b = _bar(0, 100.0)
    b["open"] = 0
    out = decompose_execution([_trade(0, 100.00, 10)], [b])
    assert out["arrival_slippage_bps"] == pytest.approx(0.0, abs=1e-12)


# ==========================================================================
# decompose_execution — best / worst bracket
# ==========================================================================
def test_best_worst_brackets_arrival():
    bars = [_bar(0, 100.0, bid=99.96, ask=100.04, hi=100.50, lo=99.50)]
    out = decompose_execution([_trade(0, 100.05, 10)], bars)
    assert out["best_slippage_bps"] <= out["arrival_slippage_bps"] <= out["worst_slippage_bps"]


def test_best_worst_exact_values_buy():
    # arrival = ask = 100.04; lo=99.50, hi=100.50
    arrival = 100.04
    s_lo = (99.50 - arrival) / arrival * 1e4
    s_hi = (100.50 - arrival) / arrival * 1e4
    bars = [_bar(0, 100.0, bid=99.96, ask=100.04, hi=100.50, lo=99.50)]
    out = decompose_execution([_trade(0, 100.05, 10)], bars)
    assert out["best_slippage_bps"] == pytest.approx(min(s_lo, s_hi), abs=1e-9)
    assert out["worst_slippage_bps"] == pytest.approx(max(s_lo, s_hi), abs=1e-9)


def test_best_worst_sign_flips_for_short():
    # For a short, side=-1 flips which extreme is best/worst relative to a buy.
    arrival = 100.04  # bid
    side = -1.0
    s_lo = (99.50 - arrival) / arrival * side * 1e4
    s_hi = (100.50 - arrival) / arrival * side * 1e4
    bars = [_bar(0, 100.0, bid=100.04, ask=100.12, hi=100.50, lo=99.50)]
    out = decompose_execution([_trade(0, 99.95, -10)], bars)
    assert out["best_slippage_bps"] == pytest.approx(min(s_lo, s_hi), abs=1e-9)
    assert out["worst_slippage_bps"] == pytest.approx(max(s_lo, s_hi), abs=1e-9)


def test_best_worst_none_when_no_high_low():
    b = _bar(0, 100.0, bid=99.96, ask=100.04)
    b["high"] = None
    b["low"] = None
    out = decompose_execution([_trade(0, 100.05, 10)], [b])
    assert out["best_slippage_bps"] is None
    assert out["worst_slippage_bps"] is None
    assert out["arrival_slippage_bps"] is not None  # arrival still computed


# ==========================================================================
# decompose_execution — transient (+1) / permanent (+30) impact
# ==========================================================================
def test_transient_permanent_positive_when_price_reverts():
    # buy at 100.10; mid reverts to 100.00 from bar +1 onward
    bars = [_bar(i, 100.10 if i == 0 else 100.00) for i in range(40)]
    out = decompose_execution([_trade(0, 100.10, 10)], bars)
    assert out["transient_impact_bps"] == pytest.approx((100.10 - 100.00) / 100.10 * 1e4, abs=1e-9)
    assert out["permanent_impact_bps"] == pytest.approx((100.10 - 100.00) / 100.10 * 1e4, abs=1e-9)


def test_impact_uses_mid_when_quote_present_at_future_bar():
    # future bar has quote -> impact uses its mid, not close
    bars = [_bar(0, 100.10)]
    bars.append(_bar(1, 999.0, bid=99.0, ask=101.0))  # close is decoy; mid=100.0
    out = decompose_execution([_trade(0, 100.10, 10)], bars, transient_offset=1, permanent_offset=5)
    assert out["transient_impact_bps"] == pytest.approx((100.10 - 100.0) / 100.10 * 1e4, abs=1e-9)


@pytest.mark.parametrize("toff,poff", [(0, 0), (2, 5), (1, 3)])
def test_custom_offsets(toff, poff):
    # close 100 at bar0, 98 elsewhere
    bars = [_bar(i, 100.0 if i == 0 else 98.0) for i in range(10)]
    out = decompose_execution([_trade(0, 105.0, 10)], bars, transient_offset=toff, permanent_offset=poff)
    # bar at offset 0 has close 100 ; offsets > 0 have close 98
    fut_t = 100.0 if toff == 0 else 98.0
    fut_p = 100.0 if poff == 0 else 98.0
    assert out["transient_impact_bps"] == pytest.approx((105.0 - fut_t) / 105.0 * 1e4, abs=1e-9)
    assert out["permanent_impact_bps"] == pytest.approx((105.0 - fut_p) / 105.0 * 1e4, abs=1e-9)


def test_negative_offset_out_of_range_yields_none():
    bars = [_bar(i, 100.0) for i in range(5)]
    out = decompose_execution([_trade(0, 105.0, 10)], bars, transient_offset=-1, permanent_offset=-1)
    assert out["transient_impact_bps"] is None
    assert out["permanent_impact_bps"] is None


def test_offset_past_end_yields_none():
    bars = [_bar(i, 100.0) for i in range(3)]
    out = decompose_execution([_trade(0, 105.0, 10)], bars, transient_offset=1, permanent_offset=99)
    assert out["transient_impact_bps"] is not None      # +1 in range
    assert out["permanent_impact_bps"] is None           # +99 out of range


def test_impact_future_mid_nonpositive_skipped():
    # future bar close 0 -> fb_mid falsy -> skipped, list empty -> None
    bars = [_bar(0, 100.0), _bar(1, 0.0)]
    out = decompose_execution([_trade(0, 105.0, 10)], bars, transient_offset=1, permanent_offset=1)
    assert out["transient_impact_bps"] is None


def test_impact_out_of_order_bars_index_by_sorted_ts():
    # timestamps shuffled; only the bar at ts==5 has close 100, rest 98
    ordering = [9, 5, 2, 7, 1, 3, 6, 4, 8, 0]
    bars = [_bar(t, 100.0 if t == 5 else 98.0) for t in ordering]
    out = decompose_execution([_trade(5, 105.0, 10)], bars, transient_offset=1, permanent_offset=2)
    # bar +1 after ts5 (sorted) is ts6 close 98
    assert out["transient_impact_bps"] == pytest.approx((105.0 - 98.0) / 105.0 * 1e4, abs=1e-9)


# ==========================================================================
# decompose_execution — skip / degenerate trade filters
# ==========================================================================
def test_empty_trades_all_none():
    out = decompose_execution([], [_bar(0, 100.0)])
    for k in ("arrival_slippage_bps", "spread_cost_bps", "best_slippage_bps",
              "worst_slippage_bps", "transient_impact_bps", "permanent_impact_bps"):
        assert out[k] is None
    assert out["n_fills_analyzed"] == 0


def test_empty_bars_all_none():
    out = decompose_execution([_trade(0, 100.0, 10)], [])
    assert out["n_fills_analyzed"] == 0
    assert out["arrival_slippage_bps"] is None


@pytest.mark.parametrize(
    "fill,qty,ts",
    [
        (100.0, 10, 999),   # no bar at this ts -> skipped
        (0.0, 10, 0),       # entry_price 0 (falsy) -> skipped
        (-5.0, 10, 0),      # entry_price <= 0 -> skipped
        (100.0, 0, 0),      # qty == 0 -> skipped
    ],
)
def test_degenerate_trades_skipped(fill, qty, ts):
    bars = [_bar(0, 100.0, bid=99.96, ask=100.04)]
    out = decompose_execution([_trade(ts, fill, qty)], bars)
    assert out["n_fills_analyzed"] == 0
    assert out["arrival_slippage_bps"] is None


def test_negative_qty_short_is_analyzed_not_skipped():
    bars = [_bar(0, 100.0, bid=99.96, ask=100.04)]
    out = decompose_execution([_trade(0, 100.0, -10)], bars)
    assert out["n_fills_analyzed"] == 1  # shorts ARE analyzed (only qty==0 skipped)


def test_missing_timestamp_key_maps_to_zero():
    # a bar with no 'timestamp' key is keyed at 0.0; a trade at ts 0 matches it.
    b = _bar(0, 100.0, bid=99.96, ask=100.04)
    del b["timestamp"]
    out = decompose_execution([_trade(0, 100.05, 10)], [b])
    assert out["n_fills_analyzed"] == 1


# ==========================================================================
# decompose_execution — NaN / inf / extreme magnitude (graceful, no crash)
# ==========================================================================
def test_nan_entry_price_not_skipped_yields_nan():
    # NaN is not falsy and `nan <= 0` is False, so it passes the filter.
    bars = [_bar(0, 100.0, hi=100.5, lo=99.5)]
    out = decompose_execution([_trade(0, float("nan"), 10)], bars)
    assert out["n_fills_analyzed"] == 1
    assert math.isnan(out["arrival_slippage_bps"])


def test_inf_entry_price_yields_inf():
    bars = [_bar(0, 100.0, bid=99.0, ask=101.0)]
    out = decompose_execution([_trade(0, float("inf"), 10)], bars)
    assert out["n_fills_analyzed"] == 1
    assert math.isinf(out["arrival_slippage_bps"])


def test_extreme_small_magnitude_prices():
    bars = [_bar(0, 1e-6, bid=0.9e-6, ask=1.1e-6, hi=2e-6, lo=1e-9)]
    out = decompose_execution([_trade(0, 1.05e-6, 10)], bars)
    assert out["n_fills_analyzed"] == 1
    assert math.isfinite(out["arrival_slippage_bps"])


def test_extreme_large_magnitude_prices():
    bars = [_bar(0, 1e9, bid=0.999e9, ask=1.001e9)]
    out = decompose_execution([_trade(0, 1.0005e9, 10)], bars)
    assert math.isfinite(out["arrival_slippage_bps"])
    assert out["arrival_slippage_bps"] < 0  # filled below ask


# ==========================================================================
# decompose_execution — aggregation & idempotency
# ==========================================================================
def test_aggregation_is_mean_over_fills():
    bars = [_bar(0, 100.0, bid=99.96, ask=100.04), _bar(1, 100.0, bid=99.96, ask=100.04)]
    t0 = _trade(0, 100.04, 10)   # arrival 0
    t1 = _trade(1, 100.14, 10)   # arrival (100.14-100.04)/100.04*1e4
    out = decompose_execution([t0, t1], bars)
    a1 = (100.14 - 100.04) / 100.04 * 1e4
    assert out["n_fills_analyzed"] == 2
    assert out["arrival_slippage_bps"] == pytest.approx((0.0 + a1) / 2.0, abs=1e-9)


def test_idempotent_repeated_calls():
    bars = [_bar(0, 100.0, bid=99.96, ask=100.04, hi=100.5, lo=99.5)]
    trades = [_trade(0, 100.05, 10)]
    a = decompose_execution(trades, bars)
    b = decompose_execution(trades, bars)
    assert a == b


def test_does_not_mutate_inputs():
    bars = [_bar(0, 100.0, bid=99.96, ask=100.04)]
    trades = [_trade(0, 100.05, 10)]
    bars_copy = [dict(bars[0])]
    decompose_execution(trades, bars)
    assert bars[0] == bars_copy[0]
    assert trades[0].entry_price == 100.05


# ==========================================================================
# ParityReport — properties
# ==========================================================================
def test_return_gap_property():
    r = _full_report(live_total_return=0.07, bt_total_return=0.10)
    assert r.return_gap == pytest.approx(-0.03, abs=1e-12)


def test_slippage_gap_property():
    r = _full_report(live_realized_slippage_mean_bps=3.456, bt_slippage_bps=2.0)
    assert r.slippage_gap_bps == pytest.approx(1.456, abs=1e-9)


def test_slippage_gap_none_when_mean_none():
    r = _full_report(live_realized_slippage_mean_bps=None)
    assert r.slippage_gap_bps is None


# ==========================================================================
# ParityReport — to_dict structure + tca block + rounding
# ==========================================================================
def test_to_dict_structure_keys():
    d = _full_report().to_dict()
    assert set(d.keys()) == {"strategy", "symbol", "n_bars", "backtest", "live_path", "gap", "tca"}
    assert set(d["backtest"].keys()) == {"total_return", "n_trades", "assumed_slippage_bps"}
    assert set(d["tca"].keys()) == {
        "n_fills_analyzed", "arrival_slippage_bps", "spread_cost_bps",
        "best_slippage_bps", "worst_slippage_bps",
        "transient_impact_bps", "permanent_impact_bps",
    }


def test_to_dict_rounding_backtest_and_gap():
    r = _full_report(bt_total_return=0.123456, live_total_return=0.111111)
    d = r.to_dict()
    assert d["backtest"]["total_return"] == round(0.123456, 4)
    assert d["live_path"]["total_return"] == round(0.111111, 4)
    assert d["gap"]["return_gap"] == round(0.111111 - 0.123456, 4)


def test_to_dict_rounding_live_slippage_and_latency():
    d = _full_report(
        live_realized_slippage_mean_bps=3.456,
        live_realized_slippage_p95_bps=5.6789,
        signal_to_fill_p50_ms=12.34,
        signal_to_fill_p95_ms=99.99,
    ).to_dict()
    assert d["live_path"]["realized_slippage_mean_bps"] == 3.46  # round 2
    assert d["live_path"]["realized_slippage_p95_bps"] == 5.68
    assert d["live_path"]["signal_to_fill_p50_ms"] == 12.3       # round 1
    assert d["live_path"]["signal_to_fill_p95_ms"] == 100.0


def test_to_dict_tca_rounding_r2():
    d = _full_report(
        arrival_slippage_bps=1.2345,
        spread_cost_bps=8.005,
        best_slippage_bps=-1.005,
        worst_slippage_bps=2.0,
        transient_impact_bps=0.5,
        permanent_impact_bps=0.255,
        n_fills_analyzed=2,
    ).to_dict()
    assert d["tca"]["n_fills_analyzed"] == 2
    assert d["tca"]["arrival_slippage_bps"] == 1.23
    assert d["tca"]["spread_cost_bps"] == round(8.005, 2)
    assert d["tca"]["permanent_impact_bps"] == round(0.255, 2)


def test_to_dict_none_passthrough():
    d = _full_report(
        live_realized_slippage_mean_bps=None,
        live_realized_slippage_p95_bps=None,
        signal_to_fill_p50_ms=None,
        signal_to_fill_p95_ms=None,
        arrival_slippage_bps=None,
        spread_cost_bps=None,
        best_slippage_bps=None,
        worst_slippage_bps=None,
        transient_impact_bps=None,
        permanent_impact_bps=None,
        n_fills_analyzed=0,
    ).to_dict()
    assert d["live_path"]["realized_slippage_mean_bps"] is None
    assert d["live_path"]["signal_to_fill_p50_ms"] is None
    assert d["gap"]["slippage_gap_bps"] is None
    assert d["tca"]["arrival_slippage_bps"] is None
    assert d["tca"]["n_fills_analyzed"] == 0


def test_to_dict_assumed_slippage_passthrough_not_rounded_key():
    d = _full_report(bt_slippage_bps=2.5).to_dict()
    assert d["backtest"]["assumed_slippage_bps"] == 2.5


# ==========================================================================
# ParityReport — render()
# ==========================================================================
def test_render_returns_str_with_headers():
    txt = _full_report().render()
    assert isinstance(txt, str)
    assert "Parity:" in txt
    assert "total_return" in txt
    assert "TCA decomposition" in txt  # n_fills_analyzed > 0


def test_render_shows_rejects_warning():
    txt = _full_report(live_rejects=3).render()
    assert "3 live orders were risk-blocked" in txt


def test_render_na_slippage_when_none():
    txt = _full_report(live_realized_slippage_mean_bps=None).render()
    assert "n/a" in txt


def test_render_omits_tca_block_when_no_fills():
    txt = _full_report(n_fills_analyzed=0).render()
    assert "TCA decomposition" not in txt


# ==========================================================================
# run_parity — offline smoke (SimAdapter, no network)
# ==========================================================================
@pytest.mark.parametrize("strategy", ["donchian", "zscore"])
def test_run_parity_smoke_end_to_end(strategy):
    bars = _sine_bars(60, quotes=True)
    rep = run_parity(strategy, bars, symbol="X")
    assert isinstance(rep, ParityReport)
    assert rep.strategy == strategy
    assert rep.symbol == "X"
    assert rep.n_bars == 60
    d = rep.to_dict()
    assert "tca" in d and "backtest" in d and "live_path" in d
    assert d["tca"]["n_fills_analyzed"] >= 0
    assert isinstance(rep.render(), str)
    assert rep.live_rejects >= 0


def test_run_parity_deterministic_same_seed():
    bars = _sine_bars(60, quotes=True)
    a = run_parity("donchian", bars, symbol="X", sim_seed=11)
    b = run_parity("donchian", bars, symbol="X", sim_seed=11)
    assert a.to_dict() == b.to_dict()


def test_run_parity_n_bars_matches_input():
    bars = _sine_bars(40, quotes=True)
    rep = run_parity("donchian", bars, symbol="ABC")
    assert rep.n_bars == len(bars) == 40
    assert rep.symbol == "ABC"


# ==========================================================================
# backtest_resting — offline smoke (rate-limit disabled, allow_short)
# ==========================================================================
def test_backtest_resting_returns_result_and_provenance_slippage():
    bars = _sine_bars(80)
    res = backtest_resting(make("donchian"), bars)
    assert isinstance(res, BacktestResult)
    # default fill model half_spread_bps = 1.0 is echoed on the result
    assert res.slippage_bps == pytest.approx(1.0)
    assert res.symbol == "X"
    assert res.n_trades >= 0


def test_backtest_resting_donchian_generates_trades():
    bars = _sine_bars(80)
    res = backtest_resting(make("donchian"), bars)
    assert res.n_trades >= 1  # the sine series breaks out of the Donchian channel


def test_backtest_resting_deterministic_same_seed():
    bars = _sine_bars(80)
    a = backtest_resting(make("donchian"), bars, seed=7)
    b = backtest_resting(make("donchian"), bars, seed=7)
    assert a.total_return == b.total_return
    assert a.n_trades == b.n_trades


def test_backtest_resting_empty_bars_graceful():
    res = backtest_resting(make("donchian"), [])
    assert isinstance(res, BacktestResult)
    assert res.n_trades == 0
    assert res.total_return == 0.0


@pytest.mark.parametrize("allow_short", [True, False])
def test_backtest_resting_allow_short_flag_runs(allow_short):
    bars = _sine_bars(80)
    res = backtest_resting(make("zscore-ls"), bars, allow_short=allow_short)
    assert isinstance(res, BacktestResult)
    assert res.n_trades >= 0  # runs without rejecting/crashing either way


def test_backtest_resting_custom_fill_model_changes_provenance():
    bars = _sine_bars(80)
    fm = FillModel(half_spread_bps=5.0, impact_coef_bps=0.0,
                   participation_cap=0.10, min_tick=0.01)
    res = backtest_resting(make("donchian"), bars, fill_model=fm)
    assert res.slippage_bps == pytest.approx(5.0)


def test_backtest_resting_symbol_from_bar():
    bars = _sine_bars(80)
    for b in bars:
        b["symbol"] = "ZZZ"
    res = backtest_resting(make("donchian"), bars)
    assert res.symbol == "ZZZ"
