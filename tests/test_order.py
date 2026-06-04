import time

from alpca.execution.order import Fill, Order, OrderStatus, Side, new_client_order_id


def test_lifecycle_timestamps_and_latency():
    o = Order(symbol="SPY", side=Side.BUY, qty=10, strategy="orb")
    o.mark_signal(intended_price=500.0)
    time.sleep(0.01)
    o.mark_submit()
    time.sleep(0.01)
    o.mark_ack()
    time.sleep(0.01)
    o.add_fill(Fill(ts=time.time(), price=500.5, qty=10))

    assert o.status == OrderStatus.FILLED
    assert o.status.is_terminal
    assert o.signal_to_submit_ms >= 8
    assert o.submit_to_ack_ms >= 8
    assert o.ack_to_fill_ms >= 8
    assert o.signal_to_fill_ms >= o.submit_to_fill_ms >= o.ack_to_fill_ms


def test_partial_then_full_fill():
    o = Order(symbol="AAPL", side=Side.BUY, qty=10)
    o.mark_signal(100.0)
    o.mark_submit()
    o.add_fill(Fill(ts=time.time(), price=100.0, qty=4))
    assert o.status == OrderStatus.PARTIALLY_FILLED
    assert o.filled_qty == 4
    o.add_fill(Fill(ts=time.time(), price=101.0, qty=6))
    assert o.status == OrderStatus.FILLED
    assert o.filled_qty == 10
    assert abs(o.avg_fill_price - (100.0 * 4 + 101.0 * 6) / 10) < 1e-9


def test_slippage_sign_buy_vs_sell():
    buy = Order(symbol="X", side=Side.BUY, qty=1)
    buy.mark_signal(100.0)
    buy.add_fill(Fill(ts=time.time(), price=100.10, qty=1))  # paid more -> positive (worse)
    assert buy.slippage_bps > 0
    assert abs(buy.slippage_bps - 10.0) < 1e-6  # 0.10/100 = 10 bps

    sell = Order(symbol="X", side=Side.SELL, qty=1)
    sell.mark_signal(100.0)
    sell.add_fill(Fill(ts=time.time(), price=99.90, qty=1))  # received less -> positive (worse)
    assert sell.slippage_bps > 0
    assert abs(sell.slippage_bps - 10.0) < 1e-6


def test_client_order_id_cap():
    coid = new_client_order_id("a_very_long_strategy_name_indeed", seq=12345)
    assert len(coid) <= 48
