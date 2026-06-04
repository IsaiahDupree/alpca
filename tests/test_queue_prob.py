"""
Phase 3: probabilistic queue-position model (execution/queue_prob.py) + its
optional wiring into fill_limit and the OpenOrderBook.
"""

from alpca.execution.fills import FillModel
from alpca.execution.open_orders import OpenOrderBook
from alpca.execution.order import Order, OrderType, Side, TimeInForce
from alpca.execution.queue_prob import (
    PROB_FUNCS,
    LogProbQueueFunc,
    PowerProbQueueFunc,
    QueuePosition,
    SqrtProbQueueFunc,
)


# ---- probability functions -------------------------------------------------

def test_prob_funcs_bounds_and_edges():
    for name, f in PROB_FUNCS.items():
        assert 0.0 <= f(5, 5) <= 1.0
        assert f(10, 0) == 1.0          # nothing behind -> all cancels ahead
        assert f(0, 10) == 0.0          # nothing ahead -> none ahead
        assert abs(f(7, 7) - 0.5) < 1e-9, name   # symmetric


def test_prob_funcs_monotone():
    for f in (PowerProbQueueFunc(2), LogProbQueueFunc(), SqrtProbQueueFunc()):
        assert f(10, 5) > f(5, 5) > f(5, 10)     # ↑ in front, ↓ in back


def test_power_n_sharpens():
    # higher n weights a large front more strongly
    assert PowerProbQueueFunc(3)(10, 5) > PowerProbQueueFunc(1)(10, 5)


# ---- QueuePosition ---------------------------------------------------------

def test_queue_consumed_by_trades_then_fills():
    q = QueuePosition(front=100.0)
    assert q.advance(traded_qty=40) == 0.0      # still 60 ahead
    assert q.front == 60.0
    assert q.advance(traded_qty=60) == 0.0      # exactly exhausts the queue
    assert q.at_front
    assert q.advance(traded_qty=30) == 30.0     # now trades fill us


def test_front_of_queue_fills_immediately():
    q = QueuePosition(front=0.0)
    assert q.advance(traded_qty=25) == 25.0


def test_cancellations_advance_but_do_not_fill():
    q = QueuePosition(front=100.0, prob_func=PowerProbQueueFunc(2))
    # back=0 -> prob_ahead=1 -> the full reduction advances us, no fill
    filled = q.advance(traded_qty=0, depth_reduction=40, back=0)
    assert filled == 0.0
    assert abs(q.front - 60.0) < 1e-9


# ---- fill_limit integration ------------------------------------------------

def test_fill_limit_queue_waits_its_turn():
    fm = FillModel(half_spread_bps=0, impact_coef_bps=0, participation_cap=1.0, min_tick=0.0)
    q = QueuePosition(front=500.0)
    # bar trades through the limit with 300 volume: queue (500) not yet cleared
    r1 = fm.fill_limit(True, 100.0, 100.0, 100.5, 99.5, qty=10, bar_volume=300, queue_pos=q)
    assert r1.filled_qty == 0.0
    # another bar with 400 volume: 200 left of queue eaten, 200 spills to us -> fill 10
    r2 = fm.fill_limit(True, 100.0, 100.0, 100.5, 99.5, qty=10, bar_volume=400, queue_pos=q)
    assert r2.filled_qty == 10.0


def test_fill_limit_without_queue_is_legacy():
    fm = FillModel(half_spread_bps=0, impact_coef_bps=0, participation_cap=1.0, min_tick=0.0)
    r = fm.fill_limit(True, 100.0, 100.0, 100.5, 99.5, qty=10, bar_volume=300)
    assert r.filled_qty == 10.0      # no queue_pos -> fills (volume-cap path)


# ---- OpenOrderBook integration --------------------------------------------

def _limit_buy(qty=10):
    return Order(symbol="X", side=Side.BUY, qty=qty, order_type=OrderType.LIMIT,
                 limit_price=100.0, tif=TimeInForce.GTC, strategy="t")


def _bar(ts, *, bid_size=None):
    b = {"open": 100.0, "high": 100.5, "low": 99.5, "close": 100.0,
         "volume": 200.0, "timestamp": ts, "symbol": "X"}
    if bid_size is not None:
        b["bid_size"] = bid_size
        b["ask_size"] = bid_size
    return b


def test_book_queue_model_delays_fill_vs_proxy():
    # with a deep displayed queue ahead, the queue model should NOT fill on the
    # first bar, whereas the legacy proxy fills immediately.
    proxy = OpenOrderBook(FillModel(0, 0, 1.0, 0.0))
    proxy.add(_limit_buy(), 0)
    ev_proxy = proxy.on_bar(_bar(0, bid_size=1000), 0)
    assert any(e.kind in ("fill", "partial_fill") for e in ev_proxy)

    queued = OpenOrderBook(FillModel(0, 0, 1.0, 0.0), use_queue_model=True)
    queued.add(_limit_buy(), 0)
    ev_q = queued.on_bar(_bar(0, bid_size=1000), 0)     # 1000 ahead, 200 vol -> no fill
    assert not any(e.kind in ("fill", "partial_fill") for e in ev_q)
    assert len(queued) == 1                              # still resting


def test_book_queue_model_eventually_fills():
    book = OpenOrderBook(FillModel(0, 0, 1.0, 0.0), use_queue_model=True)
    book.add(_limit_buy(qty=10), 0)
    filled = False
    for i in range(20):                                  # 200 vol/bar clears 100 queue fast
        ev = book.on_bar(_bar(i, bid_size=100), 0)       # GTC, same session
        if any(e.kind == "fill" for e in ev):
            filled = True
            break
    assert filled
