"""
Robustness tests for every registered strategy (alpca/strategies/*) and a
LiveRunner replay (alpca/runtime/runner.py) over DEGENERATE bar streams.

The point: no live network, no mocks. We build REAL strategies through the real
registry, drive them with deterministic synthetic bars, and assert INVARIANTS:
  * no crash on empty / single / two-bar / flat / zero-volume / no-quote /
    monotonic-then-crash / huge-gap / NaN / negative streams,
  * warmup before any actionable signal can fire,
  * reset() returns a strategy to its warmup (post-reset) state,
  * a runner replay over the same streams keeps consistent signed-position
    accounting and never raises.

Everything is offline and deterministic (RNG seeded with a fixed value where it
appears, via the real SimAdapter seed). We deliberately SKIP anything needing
live Alpaca / websockets — none of it is touched here.
"""

from __future__ import annotations

import asyncio
import math
from typing import Dict, List, Optional

import pytest

from alpca.config import RiskConfig
from alpca.data.feed import ReplayBarSource
from alpca.execution.adapters.sim import SimAdapter
from alpca.execution.router import ExecutionRouter
from alpca.risk.risk_engine import RiskEngine
from alpca.runtime.runner import LiveRunner
from alpca.strategies.base import BUY, EXIT, HOLD, SELL, Signal
from alpca.strategies.registry import available, make


# --------------------------------------------------------------------------- helpers
def bar(o: float, h: float, l: float, c: float, *, ts: float = 0.0,
        volume: float = 1.0, sym: str = "X",
        bid: Optional[float] = None, ask: Optional[float] = None,
        bid_size: Optional[float] = None, ask_size: Optional[float] = None,
        quote: bool = False) -> Dict[str, float]:
    """Build one OHLCV bar. The base.Strategy contract requires open/high/low/
    close to be present (the runner indexes bar['close'] directly), so we always
    set them; optional keys (timestamp/volume/quote) vary per stream."""
    b: Dict[str, float] = {
        "open": o, "high": h, "low": l, "close": c,
        "volume": volume, "timestamp": ts, "symbol": sym,
    }
    if quote:
        # default to a balanced book unless the caller overrides sizes
        mid = c
        b["bid"] = bid if bid is not None else mid - 0.01
        b["ask"] = ask if ask is not None else mid + 0.01
        b["bid_size"] = bid_size if bid_size is not None else 100.0
        b["ask_size"] = ask_size if ask_size is not None else 100.0
    return b


def flat_bar(price: float, *, ts: float = 0.0, **kw) -> Dict[str, float]:
    return bar(price, price, price, price, ts=ts, **kw)


# Each degenerate stream is a (name, list-of-bars) pair. Streams are SHORT and
# deterministic; we parametrize every strategy across every stream below.
def _empty() -> List[Dict[str, float]]:
    return []


def _single() -> List[Dict[str, float]]:
    return [bar(100, 101, 99, 100, ts=0, quote=True)]


def _two() -> List[Dict[str, float]]:
    return [bar(100, 101, 99, 100, ts=0, quote=True),
            bar(100, 102, 99, 101, ts=60, quote=True)]


def _all_flat() -> List[Dict[str, float]]:
    # every OHLC identical -> std==0, atr==0, no breakout level moves
    return [flat_bar(100.0, ts=i * 60, quote=True, bid_size=100.0, ask_size=100.0)
            for i in range(80)]


def _zero_volume() -> List[Dict[str, float]]:
    # real price motion but volume==0 on every bar (strategies must not divide by it)
    out = []
    for i in range(80):
        p = 100.0 + math.sin(i / 5.0)
        out.append(bar(p, p + 0.4, p - 0.4, p, ts=i * 60, volume=0.0, quote=True))
    return out


def _no_quote() -> List[Dict[str, float]]:
    # genuine price action but NO bid/ask/sizes -> microstructure/OFI must skip
    out = []
    for i in range(80):
        p = 100.0 + 0.1 * i
        out.append(bar(p, p + 0.5, p - 0.5, p, ts=i * 60, quote=False))
    return out


def _up_then_crash() -> List[Dict[str, float]]:
    # 40 bars monotonic up (breakouts/trend enter), then a hard crash down
    out = []
    for i in range(40):
        p = 100.0 + i
        out.append(bar(p - 0.3, p + 0.5, p - 0.5, p, ts=i * 60, quote=True))
    for i in range(40):
        p = 140.0 - 2.5 * i
        out.append(bar(p + 0.3, p + 0.5, p - 0.5, p, ts=(40 + i) * 60, quote=True))
    return out


def _huge_gaps() -> List[Dict[str, float]]:
    # alternating massive overnight gaps up/down (extreme magnitudes, real ts gaps)
    out = []
    px = 100.0
    for i in range(80):
        px = px * (3.0 if i % 2 == 0 else 1.0 / 3.0)
        out.append(bar(px, px * 1.01, px * 0.99, px, ts=i * 86400, quote=True))
    return out


def _nan_close() -> List[Dict[str, float]]:
    # warm up cleanly, then inject NaN/inf closes; strategies must not raise
    out = [bar(100, 101, 99, 100.0 + 0.01 * i, ts=i * 60, quote=True) for i in range(70)]
    out.append(bar(100, 101, 99, float("nan"), ts=70 * 60, quote=True))
    out.append(bar(100, 101, 99, float("inf"), ts=71 * 60, quote=True))
    out.append(bar(100, 101, 99, 100.0, ts=72 * 60, quote=True))
    return out


def _negative_drift() -> List[Dict[str, float]]:
    # negative-but-positive prices trending down (mean-reversion BUY territory)
    out = []
    for i in range(80):
        p = 50.0 - 0.4 * i + 0.5 * math.cos(i / 3.0)
        p = max(p, 0.5)  # keep strictly positive (a price), still a strong downtrend
        out.append(bar(p, p + 0.3, p - 0.3, p, ts=i * 60, quote=True))
    return out


STREAMS = {
    "empty": _empty,
    "single": _single,
    "two": _two,
    "all_flat": _all_flat,
    "zero_volume": _zero_volume,
    "no_quote": _no_quote,
    "up_then_crash": _up_then_crash,
    "huge_gaps": _huge_gaps,
    "nan_close": _nan_close,
    "negative_drift": _negative_drift,
}

STRATS = available()

# Strategies that build a population stdev / mean over the rolling close window
# (statistics.pstdev / fmean). On Python 3.14 those raise ValueError on a non-
# finite value, so these four used to crash on a NaN/inf close. The source now
# guards non-finite closes and HOLDs on a NaN/inf bar, so they survive like the
# others (see test_nan_close_is_handled_by_stats_strategies). These four still
# differ from the comparison-only strategies in HOW they survive: they explicitly
# skip the bad bar with a "non-finite close" HOLD rather than relying on NaN
# comparisons being all-False, so we track them as a distinct group below.
STATS_NAN_FRAGILE = {"zscore", "zscore-ls", "rsi-mr", "rsi-mr-ls"}


def _drive(strat, stream: List[Dict[str, float]]) -> List[Signal]:
    sigs = []
    for b in stream:
        sig = strat.on_bar(dict(b))  # copy: strategy must not mutate the caller's bar
        assert isinstance(sig, Signal)
        assert sig.side in (BUY, SELL, EXIT, HOLD)
        sigs.append(sig)
    return sigs


# --------------------------------------------------------------------------- registry
def test_available_is_sorted_and_nonempty():
    assert STRATS == sorted(STRATS)
    assert len(STRATS) >= 10
    # known anchors present
    for k in ("donchian", "orb", "zscore", "rsi-mr", "gap-fade", "microprice", "ofi"):
        assert k in STRATS


def test_make_unknown_raises_keyerror():
    with pytest.raises(KeyError):
        make("does-not-exist")


@pytest.mark.parametrize("name", STRATS)
def test_make_builds_a_strategy_with_name(name):
    s = make(name)
    assert s.name  # non-empty id
    # fresh strategy is flat
    assert s._in_position is False
    assert s._entry_price == 0.0


# --------------------------------------------------------------------------- robustness: on_bar
@pytest.mark.parametrize("stream_name", list(STREAMS))
@pytest.mark.parametrize("name", STRATS)
def test_strategy_survives_degenerate_stream(name, stream_name):
    """No registered strategy may raise on ANY degenerate stream, and every
    emitted Signal must be well-formed (valid side, finite-or-None strength).

    This now holds for the four statistics-based strategies (z-score / RSI-MR)
    on the nan_close stream too: they guard non-finite closes and HOLD on a
    NaN/inf bar instead of feeding it into statistics.pstdev/fmean, so they
    survive gracefully like every other strategy (no special-casing)."""
    strat = make(name)
    sigs = _drive(strat, STREAMS[stream_name]())
    for sig in sigs:
        # strength is always a real number in [0, 1] in this codebase
        assert sig.strength == sig.strength  # not NaN
        assert 0.0 <= sig.strength <= 1.0
        if sig.is_actionable:
            assert sig.side in (BUY, SELL, EXIT)


@pytest.mark.parametrize("name", STRATS)
def test_no_actionable_signal_before_minimum_warmup(name):
    """On the empty and single-bar streams no strategy can have completed warmup,
    so it must never emit an actionable entry. (One bar can't form a channel, a
    z-score window, an RSI, a gap, or an OFI window.)"""
    for stream_name in ("empty", "single"):
        strat = make(name)
        sigs = _drive(strat, STREAMS[stream_name]())
        assert all(not s.is_actionable for s in sigs), (name, stream_name)


# Every strategy must hold on a perfectly flat (zero-information) tape. (This used to
# exempt rsi-mr-ls, which shorted because Wilder RSI returned its all-up sentinel 100
# on a flat series; wilder_rsi now returns NEUTRAL 50 on a truly flat series, so
# rsi-mr-ls — like the momentum brains — correctly holds. No exceptions.)
_FLAT_NO_ENTRY = STRATS


@pytest.mark.parametrize("name", _FLAT_NO_ENTRY)
def test_all_flat_stream_never_enters(name):
    """With every OHLC identical: std==0, ATR==0, no channel break, zero gap,
    locked-ish book -> no strategy should find an entry. It must hold throughout."""
    strat = make(name)
    sigs = _drive(strat, STREAMS["all_flat"]())
    assert all(not s.is_actionable for s in sigs)


def test_rsi_ls_no_longer_enters_on_flat_tape():
    """Regression for the wilder_rsi fix: a perfectly flat series is now NEUTRAL
    (RSI 50), so neither the RSI mean-reversion nor RSI momentum long/short variants
    open a position on a zero-information tape."""
    flat = STREAMS["all_flat"]()
    for name in ("rsi-mr-ls", "rsi-momentum-ls"):
        s = make(name)
        assert all(not sig.is_actionable for sig in _drive(s, [dict(b) for b in flat]))
        assert s._side == ""


@pytest.mark.parametrize("name", STRATS)
def test_reset_returns_to_warmup(name):
    """After feeding a full stream, reset() must wipe state back to the same
    post-construction warmup: flat, zero entry price, and (re-running the empty
    stream) no actionable output — identical to a freshly-made strategy."""
    strat = make(name)
    _drive(strat, STREAMS["up_then_crash"]())
    strat.reset()
    assert strat._in_position is False
    assert strat._entry_price == 0.0
    if hasattr(strat, "_side"):
        assert strat._side == ""
    # a fresh strategy and the reset one must agree that bar-0 is non-actionable
    fresh = make(name)
    s_reset = strat.on_bar(_single()[0])
    s_fresh = fresh.on_bar(_single()[0])
    assert s_reset.is_actionable == s_fresh.is_actionable
    assert s_reset.side == s_fresh.side


@pytest.mark.parametrize("name", STRATS)
def test_reset_is_idempotent(name):
    strat = make(name)
    _drive(strat, STREAMS["huge_gaps"]())
    strat.reset()
    snap = (strat._in_position, strat._entry_price, getattr(strat, "_side", None))
    strat.reset()
    assert (strat._in_position, strat._entry_price, getattr(strat, "_side", None)) == snap


@pytest.mark.parametrize("name", sorted(set(STRATS) - STATS_NAN_FRAGILE))
def test_nan_close_does_not_corrupt_into_lingering_actionable(name):
    """For the comparison-only strategies a NaN/inf close must not crash: NaN
    comparisons are all False so the strategy can't *enter* off the NaN bar, and
    after a clean bar returns it still produces a valid Signal for every bar."""
    strat = make(name)
    sigs = _drive(strat, STREAMS["nan_close"]())
    assert len(sigs) == len(STREAMS["nan_close"]())


@pytest.mark.parametrize("name", sorted(STATS_NAN_FRAGILE))
def test_nan_close_is_handled_by_stats_strategies(name):
    """FIXED (was a real source bug): z-score / RSI-MR used to pass the rolling
    close window straight into statistics.pstdev/fmean, which on Python 3.14
    raises ValueError on a non-finite value. The source now guards non-finite
    closes and HOLDs on a NaN/inf bar instead, so these four strategies are now
    NaN-robust. We assert the new graceful behavior: feeding a NaN close does not
    raise, the NaN/inf bars return a HOLD, and the strategy keeps working on the
    subsequent finite bars."""
    strat = make(name)
    stream = STREAMS["nan_close"]()
    # Does not raise, and emits exactly one well-formed Signal per bar.
    sigs = _drive(strat, stream)
    assert len(sigs) == len(stream)
    # The nan_close stream is: 70 clean warmup bars, a NaN close, an inf close,
    # then a final clean 100.0 close. The two non-finite bars must HOLD and never
    # produce an actionable entry off the bad bar.
    nan_sig, inf_sig = sigs[70], sigs[71]
    assert nan_sig.side == HOLD and not nan_sig.is_actionable
    assert inf_sig.side == HOLD and not inf_sig.is_actionable
    assert "non-finite close" in nan_sig.reason
    assert "non-finite close" in inf_sig.reason
    # Strategy keeps working: the final finite bar still yields a valid Signal.
    final_sig = sigs[-1]
    assert isinstance(final_sig, Signal)
    assert final_sig.side in (BUY, SELL, EXIT, HOLD)


# --------------------------------------------------------------------------- quote-gated strategies
def test_microprice_no_quote_stream_is_all_hold():
    s = make("microprice")
    sigs = _drive(s, STREAMS["no_quote"]())
    assert all(sig.side == HOLD for sig in sigs)
    assert all("no quote" in sig.reason for sig in sigs)


def test_ofi_no_quote_stream_is_all_hold():
    s = make("ofi")
    sigs = _drive(s, STREAMS["no_quote"]())
    assert all(sig.side == HOLD for sig in sigs)


def test_microprice_long_short_enters_on_imbalanced_book():
    """A real, heavily bid-imbalanced book (bid_size >> ask_size) makes the
    microprice tilt 'bull' -> the standalone tilt follower goes LONG. This is the
    documented core behaviour; we verify the exact branch, not just no-crash."""
    s = make("microprice")  # k=0.5 default
    # bid_size huge vs ask_size tiny -> tilt rides toward the ask, > k
    quotes = [bar(100, 100.2, 99.8, 100.0, ts=i * 60, quote=True,
                  bid=99.95, ask=100.05, bid_size=900.0, ask_size=10.0)
              for i in range(5)]
    sigs = _drive(s, quotes)
    assert sigs[0].side == BUY
    assert s._side == "LONG"
    # subsequent bars hold the long while the tilt stays bull
    assert all(x.side == HOLD for x in sigs[1:])


def test_microprice_short_requires_allow_short_flag():
    """A bear book (ask_size >> bid_size) only opens a SHORT for the *-ls variant;
    the long-only default must hold flat instead."""
    bear = [bar(100, 100.2, 99.8, 100.0, ts=i * 60, quote=True,
                bid=99.95, ask=100.05, bid_size=10.0, ask_size=900.0)
            for i in range(4)]
    long_only = make("microprice")
    assert all(x.side != SELL for x in _drive(long_only, bear))

    ls = make("microprice-ls")
    sigs = _drive(ls, [dict(b) for b in bear])
    assert sigs[0].side == SELL
    assert ls._side == "SHORT"


def test_ofi_pure_functions_directional_and_degenerate():
    """ofi_event sign on clean cases + microprice helpers on degenerate quotes."""
    from alpca.strategies.order_flow import ofi_event
    from alpca.strategies.microstructure import (
        microprice, microprice_tilt, microprice_signal,
    )
    # signature: ofi_event(bid, bid_size, ask, ask_size, prev_bid, prev_bid_size,
    #                       prev_ask, prev_ask_size).
    # bid ROSE (dW=+bid_size=50) and ask ROSE (dV=-prev_ask_size=-60) -> e=50+60=110>0
    assert ofi_event(101, 50, 101, 40, 100, 30, 100, 60) == 110.0
    # bid FELL (dW=-prev_bid_size=-30) and ask FELL (dV=+ask_size=40) -> e=-30-40=-70<0
    assert ofi_event(99, 50, 99, 40, 100, 30, 100, 60) == -70.0
    # microprice with balanced sizes == mid
    assert microprice(100.0, 100.10, 100.0, 100.0) == pytest.approx(100.05)
    # degenerate quotes return None, never raise
    assert microprice(None, 100, 1, 1) is None
    assert microprice(100, 100.1, 0, 0) is None           # zero total size
    assert microprice_tilt(100, 100, 50, 50) is None      # locked (half<=0)
    assert microprice_tilt(101, 100, 50, 50) is None      # crossed
    assert microprice_signal(None, None, None, None) is None
    # tilt toward ask when bid_size dominates -> 'bull' (k=0)
    assert microprice_signal(100.0, 100.10, 900.0, 10.0, k=0.0) == "bull"
    assert microprice_signal(100.0, 100.10, 10.0, 900.0, k=0.0) == "bear"


def test_mean_reversion_pure_functions_edges():
    from alpca.strategies.mean_reversion import wilder_rsi, rolling_return_vol
    # too-short series -> None
    assert wilder_rsi([1.0, 2.0], 5) is None
    # all-up window -> RSI 100 (avg_loss == 0 branch)
    assert wilder_rsi([1.0, 2.0, 3.0, 4.0], 2) == 100.0
    # all-down window -> RSI 0
    assert wilder_rsi([4.0, 3.0, 2.0, 1.0], 2) == pytest.approx(0.0)
    # RSI bounded
    vals = [100 + math.sin(i) for i in range(50)]
    r = wilder_rsi(vals, 14)
    assert 0.0 <= r <= 100.0
    # vol of a flat series is 0; None when too short
    assert rolling_return_vol([100.0] * 30, 10) == pytest.approx(0.0)
    assert rolling_return_vol([100.0, 101.0], 10) is None


def test_breakout_atr_pure_function_edges():
    from alpca.strategies.breakout import _atr
    # too few bars -> None
    assert _atr([1, 2], [0, 1], [1, 2], 5) is None
    # flat bars -> ATR 0
    n = 20
    assert _atr([100.0] * n, [100.0] * n, [100.0] * n, 5) == pytest.approx(0.0)
    # known true range: each bar high-low = 2, no gaps -> ATR == 2
    highs = [101.0] * n
    lows = [99.0] * n
    closes = [100.0] * n
    assert _atr(highs, lows, closes, 5) == pytest.approx(2.0)


# --------------------------------------------------------------------------- deterministic entry behaviour
def test_donchian_enters_on_clean_breakout_then_exits_on_crash():
    s = make("donchian", period=5, atr_period=3)
    sigs = _drive(s, STREAMS["up_then_crash"]())
    sides = [x.side for x in sigs]
    assert BUY in sides
    assert EXIT in sides
    # the BUY must precede the first EXIT (enter before exit)
    assert sides.index(BUY) < sides.index(EXIT)


def test_zscore_enters_long_on_oversold_then_holds_state():
    # construct a clearly oversold final bar: long flat run then a sharp drop
    closes = [100.0] * 60 + [90.0]
    stream = [flat_bar(c, ts=i * 60, quote=True) for i, c in enumerate(closes)]
    s = make("zscore", lookback=60, entry_z=2.0)
    sigs = _drive(s, stream)
    assert sigs[-1].side == BUY
    assert s._side == "LONG"


def test_zscore_long_only_never_shorts_but_ls_can():
    # overbought final bar: flat run then a sharp jump up -> z >> entry_z
    closes = [100.0] * 60 + [110.0]
    stream = [flat_bar(c, ts=i * 60, quote=True) for i, c in enumerate(closes)]
    long_only = make("zscore", lookback=60)
    assert all(x.side != SELL for x in _drive(long_only, [dict(b) for b in stream]))
    ls = make("zscore-ls", lookback=60)
    sigs = _drive(ls, [dict(b) for b in stream])
    assert sigs[-1].side == SELL
    assert ls._side == "SHORT"


# --------------------------------------------------------------------------- runner replay
def _build_runner(name: str, symbol: str = "X", seed: int = 7) -> LiveRunner:
    # The *-ls registry entries already inject allow_short=True into the strategy,
    # so the RUNNER's RiskConfig must also allow shorts for the short leg to fill.
    is_ls = name.endswith("-ls")
    risk = RiskEngine(RiskConfig(allow_short=is_ls), day_start_equity=100_000)
    adapter = SimAdapter(seed=seed, sleep=False)  # seeded -> deterministic
    router = ExecutionRouter(adapter, risk, None, fill_timeout_s=1.0)
    return LiveRunner(make(name), symbol, router, starting_equity=100_000)


def _run(runner: LiveRunner, stream: List[Dict[str, float]]):
    return asyncio.run(runner.run(ReplayBarSource([dict(b) for b in stream])))


@pytest.mark.parametrize("stream_name", list(STREAMS))
@pytest.mark.parametrize("name", STRATS)
def test_runner_replay_never_crashes_and_counts_bars(name, stream_name):
    """Driving the REAL LiveRunner over every degenerate stream must not raise,
    must see exactly len(stream) bars, and must keep its equity-curve length in
    lock-step with bars seen.

    The nan_close stream now runs to completion for the z-score/RSI strategies
    too: they guard non-finite closes (runner -> strategy.on_bar -> HOLD) instead
    of feeding statistics.pstdev/fmean, so the replay no longer raises for them."""
    runner = _build_runner(name)
    stream = STREAMS[stream_name]()
    stats = _run(runner, stream)
    assert stats.bars_seen == len(stream)
    assert len(runner.equity_curve) == len(stream)
    # fills can never exceed orders+resting events; rejects bounded likewise
    assert stats.fills >= 0 and stats.orders_submitted >= 0


@pytest.mark.parametrize("name", STRATS)
def test_runner_empty_stream_is_a_clean_noop(name):
    runner = _build_runner(name)
    stats = _run(runner, [])
    assert stats.bars_seen == 0
    assert stats.orders_submitted == 0
    assert stats.fills == 0
    assert runner.position_qty == 0.0
    assert runner.equity_curve == []
    # equity falls back to cash == starting equity when no bars priced anything
    assert runner.equity == 100_000.0


@pytest.mark.parametrize("name", _FLAT_NO_ENTRY)
def test_runner_all_flat_stream_does_no_trades(name):
    """A perfectly flat market gives no actionable entries (except rsi-mr-ls, see
    above), so the runner submits no orders and ends flat with equity unchanged."""
    runner = _build_runner(name)
    _run(runner, STREAMS["all_flat"]())
    assert runner.position_qty == 0.0
    assert runner.stats.orders_submitted == 0
    assert runner.equity == pytest.approx(100_000.0)


def test_runner_rsi_mr_ls_flat_ends_flat():
    """After the wilder_rsi neutral-50 fix, a flat tape is zero-information: the
    long/short RSI-MR runner takes NO position and ends flat with finite equity."""
    runner = _build_runner("rsi-mr-ls")
    _run(runner, STREAMS["all_flat"]())
    assert runner.position_qty == 0.0
    assert runner.stats.shorts_opened == 0
    assert math.isfinite(runner.equity)


def test_runner_donchian_up_down_ends_flat():
    """Real round-trip: donchian enters on the up-leg, exits on the crash; the
    SIGNED position must return to 0 and at least one fill must have happened."""
    runner = _build_runner("donchian")
    runner.strategy = make("donchian", period=5, atr_period=3)
    _run(runner, STREAMS["up_then_crash"]())
    assert runner.position_qty == 0.0
    assert runner.stats.fills >= 2  # in and out


def test_runner_position_qty_matches_internal_positions():
    """Invariant: the public position_qty equals the signed qty held in the
    internal positions map (or 0 when flat), after a non-trivial replay."""
    runner = _build_runner("zscore")
    runner.strategy = make("zscore", lookback=20, entry_z=1.5)
    _run(runner, STREAMS["up_then_crash"]())
    pos = runner._positions.get("X")
    expected = pos.qty if pos else 0.0
    assert runner.position_qty == expected


def test_runner_equity_finite_on_well_formed_streams():
    """Over every NON-NaN stream the ending equity stays a finite positive number
    (cash + signed mark-to-market never explodes for these magnitudes)."""
    for stream_name in ("two", "zero_volume", "no_quote", "up_then_crash",
                        "negative_drift"):
        runner = _build_runner("supertrend")
        _run(runner, STREAMS[stream_name]())
        assert math.isfinite(runner.equity)
        assert runner.equity > 0.0


def test_runner_summary_is_self_consistent():
    runner = _build_runner("keltner")
    _run(runner, STREAMS["up_then_crash"]())
    summ = runner.summary()
    assert summ["symbol"] == "X"
    assert summ["strategy"] == "keltner"
    assert summ["bars_seen"] == len(STREAMS["up_then_crash"]())
    assert summ["fills"] >= 0
    # ending_equity in the summary matches the live property (rounded)
    assert summ["ending_equity"] == pytest.approx(round(runner.equity, 2))


def test_runner_to_result_roundtrips_equity_curve():
    runner = _build_runner("donchian")
    runner.strategy = make("donchian", period=5, atr_period=3)
    _run(runner, STREAMS["up_then_crash"]())
    res = runner.to_result()
    assert res.symbol == "X"
    assert res.strategy == "donchian"
    assert len(res.equity_curve) == len(STREAMS["up_then_crash"]())
    assert res.starting_equity == 100_000.0


def test_runner_huge_gaps_no_crash_and_bars_counted():
    """Extreme alternating overnight gaps (with real day-sized timestamp jumps so
    the session-start hook actually fires) must not break the runner."""
    runner = _build_runner("ofi")
    stats = _run(runner, STREAMS["huge_gaps"]())
    assert stats.bars_seen == len(STREAMS["huge_gaps"]())
    assert math.isfinite(runner.cash)


def test_runner_nan_close_does_not_raise():
    """The runner indexes bar['close'] and marks _last_price; for a comparison-
    only strategy a NaN/inf close propagates as NaN/inf in equity WITHOUT raising
    (graceful). We use donchian here; the z-score/RSI pstdev crash is pinned in
    test_runner_replay_never_crashes_and_counts_bars[<stats>-nan_close]."""
    runner = _build_runner("donchian")
    stats = _run(runner, STREAMS["nan_close"]())
    assert stats.bars_seen == len(STREAMS["nan_close"]())
    # final bar is a clean 100.0 close, so _last_price recovers to finite
    assert runner._last_price == 100.0
    assert math.isfinite(runner.equity)


def test_runner_session_start_hook_resets_ofi_window():
    """huge_gaps uses 1-day timestamp steps, so the runner's day-rollover detector
    fires strategy.on_session_start() on every bar after the first. For L1OFI that
    clears the rolling window, so its internal prev/window state stays bounded."""
    s = make("ofi", window=20)
    runner = _build_runner("ofi")
    runner.strategy = s
    _run(runner, STREAMS["huge_gaps"]())
    # window deque is bounded; after per-bar session resets it should be tiny
    assert len(s._e) <= s.window
    assert len(s._sz) <= s.window
