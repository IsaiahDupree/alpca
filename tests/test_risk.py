from alpca.config import RiskConfig
from alpca.execution.order import Order, Side
from alpca.risk.risk_engine import Position, RiskEngine


def _order(symbol="SPY", side=Side.BUY, qty=1, price=500.0):
    o = Order(symbol=symbol, side=side, qty=qty)
    o.mark_signal(price)
    return o


def test_allows_normal_order():
    eng = RiskEngine(RiskConfig(), day_start_equity=100_000)
    d = eng.check(_order(qty=2), equity=100_000, positions={}, ref_price=500.0)
    assert d.allowed


def test_blocks_over_notional():
    eng = RiskEngine(RiskConfig(max_order_notional=10_000))
    d = eng.check(_order(qty=100), equity=100_000, positions={}, ref_price=500.0)
    assert not d.allowed
    assert d.code == "MAX_ORDER_NOTIONAL"


def test_forbidden_symbol():
    eng = RiskEngine(RiskConfig(), forbidden_symbols=["TSLA"])
    d = eng.check(_order(symbol="TSLA"), equity=100_000, positions={}, ref_price=500.0)
    assert not d.allowed
    assert d.code == "FORBIDDEN"


def test_daily_loss_halts():
    eng = RiskEngine(RiskConfig(daily_loss_pct=0.02), day_start_equity=100_000)
    d = eng.check(_order(), equity=97_000, positions={}, ref_price=500.0)
    assert not d.allowed
    assert d.code == "DAILY_LOSS"
    assert eng.halted
    d2 = eng.check(_order(), equity=100_000, positions={}, ref_price=500.0)
    assert d2.code == "HALTED"


def test_rate_limit():
    eng = RiskEngine(RiskConfig(max_orders_per_min=3))
    for _ in range(3):
        d = eng.check(_order(), equity=100_000, positions={}, ref_price=500.0)
        assert d.allowed
        eng.record_submission()
    d = eng.check(_order(), equity=100_000, positions={}, ref_price=500.0)
    assert not d.allowed
    assert d.code == "RATE_LIMIT"


def test_max_open_positions():
    eng = RiskEngine(RiskConfig(max_open_positions=2))
    positions = {
        "AAA": Position("AAA", 1, 100.0),
        "BBB": Position("BBB", 1, 100.0),
    }
    d = eng.check(_order(symbol="CCC"), equity=1_000_000, positions=positions, ref_price=100.0)
    assert not d.allowed
    assert d.code == "MAX_POSITIONS"


def test_concentration_cap():
    eng = RiskEngine(RiskConfig(max_concentration_pct=0.25))
    d = eng.check(_order(qty=60, price=500.0), equity=100_000, positions={}, ref_price=500.0)
    assert not d.allowed
    assert d.code == "CONCENTRATION"
