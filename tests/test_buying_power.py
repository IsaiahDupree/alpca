"""Buying-power enforcement: a BUY may not cost more than available cash.

Tests isolate the buying-power gate by relaxing the other caps (notional +
concentration) so only the cash check can fire.
"""

from alpca.config import RiskConfig
from alpca.execution.order import Order, Side
from alpca.risk.risk_engine import RiskEngine


def _cfg(**kw):
    base = dict(max_order_notional=1e9, max_concentration_pct=1.0, **kw)
    return RiskConfig(**base)


def _buy(qty, price=100.0):
    o = Order(symbol="SPY", side=Side.BUY, qty=qty)
    o.mark_signal(price)
    return o


def test_buy_within_cash_allowed():
    eng = RiskEngine(_cfg())
    d = eng.check(_buy(50), equity=10_000, cash=10_000, ref_price=100.0)  # $5k <= $10k
    assert d.allowed, d.reason


def test_buy_over_cash_rejected():
    eng = RiskEngine(_cfg())
    d = eng.check(_buy(150), equity=100_000, cash=10_000, ref_price=100.0)  # $15k > $10k
    assert not d.allowed
    assert d.code == "INSUFFICIENT_BUYING_POWER"


def test_buying_power_not_checked_when_cash_absent():
    # backward-compatible: callers that don't supply cash get no BP gate
    eng = RiskEngine(_cfg())
    d = eng.check(_buy(150), equity=100_000, ref_price=100.0)
    assert d.allowed, d.reason


def test_buying_power_can_be_disabled():
    eng = RiskEngine(_cfg(enforce_buying_power=False))
    d = eng.check(_buy(150), equity=100_000, cash=10_000, ref_price=100.0)
    assert d.allowed, d.reason


def test_sell_not_blocked_by_cash():
    # a SELL that opens a short is allowed (no cash gate) WHEN shorting is enabled.
    eng = RiskEngine(_cfg(allow_short=True))
    o = Order(symbol="SPY", side=Side.SELL, qty=150)
    o.mark_signal(100.0)
    d = eng.check(o, equity=100_000, cash=0.0, ref_price=100.0)  # selling needs no cash
    assert d.allowed, d.reason
