"""
Integration: the realistic fill model + Alpaca fee model flowing through the
backtester change PnL in the right direction vs the flat default.
"""

from alpca.backtest.engine import run_backtest
from alpca.data.bars import synthetic_bars
from alpca.execution.fees import AlpacaFeeModel
from alpca.execution.fills import FillModel
from alpca.strategies.breakout import DonchianBreakout
from alpca.strategies.registry import make


def _bars():
    return synthetic_bars("DEMO", n=500, seed=11, drift=0.0004, vol=0.013)


def test_default_is_unchanged_flat_behavior():
    # No fill_model/fee_model -> legacy flat path: entry fills at the NEXT bar's
    # open + slippage_bps (next-bar-open execution, no look-ahead).
    bars = _bars()
    a = run_backtest(make("donchian"), bars, slippage_bps=2.0, commission_bps=1.0)
    e = a.trades[0]
    assert e.entry_price > e.entry_ref
    assert abs((e.entry_price / e.entry_ref - 1) * 10_000 - 2.0) < 1e-6


def test_size_impact_makes_fills_worse():
    bars = _bars()
    flat = run_backtest(make("donchian"), bars,
                        fill_model=FillModel.flat(2.0))
    # same half-spread (2bps) but now WITH square-root size impact + volume cap
    impactful = run_backtest(make("donchian"), bars,
                             fill_model=FillModel(half_spread_bps=2.0,
                                                  impact_coef_bps=15.0,
                                                  participation_cap=1.0,
                                                  min_tick=0.0))
    fe = flat.trades[0]
    ie = impactful.trades[0]
    # impact only adds cost when there's volume context; synthetic bars have
    # volume, so the impactful entry should fill at a HIGHER price than flat.
    assert ie.entry_price >= fe.entry_price


def _one_round_trip_bars():
    """Donchian(period=5): flat channel, one breakout up, then a break down —
    exactly one entry and one exit, so there's no multi-trade compounding to
    confound a fee comparison."""
    bars = [_b(100, 100.5, 99.5, 100, i) for i in range(6)]   # channel
    bars.append(_b(105, 112, 104, 112, 6))                    # breakout (signal)
    bars.append(_b(120, 122, 119, 121, 7))                    # entry fills @ open 120
    bars.append(_b(121, 123, 120, 122, 8))                    # hold
    bars.append(_b(118, 119, 90, 95, 9))                      # break down (exit signal)
    bars.append(_b(94, 95, 92, 93, 10))                       # exit fills @ open 94
    return bars


def _b(o, h, l, c, ts):
    return {"open": o, "high": h, "low": l, "close": c, "volume": 1e7,
            "timestamp": ts, "symbol": "T"}


def test_sell_fees_reduce_proceeds():
    bars = _one_round_trip_bars()
    # baseline must be TRULY fee-free: flat(0) fill AND commission_bps=0 (else the
    # legacy 1bp default commission makes the "no-fee" run actually pay more).
    no_fee = run_backtest(DonchianBreakout(period=5, atr_period=3), bars,
                          fill_model=FillModel.flat(0.0), commission_bps=0.0)
    with_fee = run_backtest(DonchianBreakout(period=5, atr_period=3), bars,
                            fill_model=FillModel.flat(0.0), fee_model=AlpacaFeeModel())
    # exactly one round trip in each
    assert no_fee.n_trades == 1 and with_fee.n_trades == 1
    a, b = no_fee.trades[0], with_fee.trades[0]
    # identical entry + exit prices/qty (buys are fee-free, same starting cash)
    assert abs(a.qty - b.qty) < 1e-9
    assert abs(a.exit_price - b.exit_price) < 1e-9
    # with a single trade there is no compounding divergence, so the SEC/TAF sell
    # fee must lower ending equity by exactly that fee.
    fee = AlpacaFeeModel().regulatory(False, b.qty, b.exit_price)
    assert with_fee.ending_equity < no_fee.ending_equity
    assert abs((no_fee.ending_equity - with_fee.ending_equity) - fee) < 1e-6


def test_volume_cap_limits_position_size():
    # tiny-volume bars + a tight participation cap -> entry qty is capped
    bars = synthetic_bars("DEMO", n=400, seed=3, drift=0.0005, vol=0.012)
    for b in bars:
        b["volume"] = 100.0  # force scarce liquidity
    capped = run_backtest(make("donchian"), bars,
                          fill_model=FillModel(half_spread_bps=1.0, impact_coef_bps=0.0,
                                               participation_cap=0.10, min_tick=0.0))
    if capped.trades:
        # at most 10% of 100 shares = 10 shares may fill on entry
        assert capped.trades[0].qty <= 10.0 + 1e-9
