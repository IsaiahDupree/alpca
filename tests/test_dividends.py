"""
Dividend cash-flow crediting (raw/split-price path).

A held position crossing an ex-date is credited qty*amount in cash. Verified with
RAW-style synthetic bars where the price does NOT drop (we isolate the cash
credit), and the DividendSchedule window logic is unit-tested directly.
"""

from alpca.backtest.engine import run_backtest
from alpca.data.corporate_actions import Dividend, DividendSchedule
from alpca.strategies.breakout import DonchianBreakout


def _bar(o, h, l, c, ts):
    return {"open": o, "high": h, "low": l, "close": c, "volume": 1e7,
            "timestamp": ts, "symbol": "T"}


def test_schedule_window_is_half_open():
    sched = DividendSchedule.from_pairs([(100.0, 0.5), (200.0, 0.25), (300.0, 1.0)])
    assert len(sched) == 3
    # (prev, cur] — ex at exactly cur is included, at exactly prev is not
    assert sched.per_share_between(50, 100) == 0.5     # includes ex@100
    assert sched.per_share_between(100, 200) == 0.25   # excludes 100, includes 200
    assert sched.per_share_between(99, 300) == 0.5 + 0.25 + 1.0
    assert sched.per_share_between(300, 400) == 0.0


def _hold_through_bars():
    """Donchian(5) enters on the breakout, then holds across an ex-date bar."""
    bars = [_bar(100, 100.5, 99.5, 100, i) for i in range(6)]   # channel
    bars.append(_bar(105, 112, 104, 112, 6))                    # breakout signal
    bars.append(_bar(120, 122, 119, 121, 7))                    # entry fills @ open 120
    bars.append(_bar(121, 123, 120, 122, 8))                    # HOLD (ex-date here)
    bars.append(_bar(122, 124, 121, 123, 9))                    # HOLD
    bars.append(_bar(123, 125, 122, 124, 10))                   # HOLD (EOD liquidation)
    return bars


def test_dividend_credited_while_holding():
    bars = _hold_through_bars()
    # ex-date at ts=8 (a bar we hold through), $1.00/share
    sched = DividendSchedule.from_pairs([(8.0, 1.0)])

    no_div = run_backtest(DonchianBreakout(period=5, atr_period=3), bars,
                          slippage_bps=0.0, commission_bps=0.0)
    with_div = run_backtest(DonchianBreakout(period=5, atr_period=3), bars,
                            slippage_bps=0.0, commission_bps=0.0, dividends=sched)

    assert with_div.n_trades == no_div.n_trades >= 1
    qty = with_div.trades[0].qty
    # ending equity is higher by exactly qty * $1.00
    assert with_div.dividend_income > 0
    assert abs(with_div.dividend_income - qty * 1.0) < 1e-6
    assert abs((with_div.ending_equity - no_div.ending_equity) - qty * 1.0) < 1e-6
    assert with_div.summary()["dividend_income"] == round(qty * 1.0, 2)


def test_no_dividend_when_flat():
    # ex-date at ts=2, before any position is opened -> no credit
    bars = _hold_through_bars()
    sched = DividendSchedule.from_pairs([(2.0, 5.0)])
    res = run_backtest(DonchianBreakout(period=5, atr_period=3), bars,
                       slippage_bps=0.0, commission_bps=0.0, dividends=sched)
    assert res.dividend_income == 0.0


def test_dividend_income_defaults_zero():
    bars = _hold_through_bars()
    res = run_backtest(DonchianBreakout(period=5, atr_period=3), bars,
                       slippage_bps=0.0, commission_bps=0.0)
    assert res.dividend_income == 0.0
    assert res.summary()["dividend_income"] == 0.0
