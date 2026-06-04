import json
import time

from alpca.execution.order import Fill, Order, Side
from alpca.execution.order_event_log import EventType, OrderEventLog


def _make_filled_order(price=500.0):
    o = Order(symbol="SPY", side=Side.BUY, qty=1, strategy="t")
    o.mark_signal(price)
    o.mark_submit()
    o.mark_ack()
    o.add_fill(Fill(ts=time.time(), price=price, qty=1))
    return o


def test_event_count_and_chain(tmp_path):
    path = str(tmp_path / "events.jsonl")
    log = OrderEventLog(path)
    n = 10
    for _ in range(n):
        o = _make_filled_order()
        log.append(o, EventType.SIGNAL)
        log.append(o, EventType.SUBMIT)
        log.append(o, EventType.ACK)
        log.append(o, EventType.FILL)

    rows = log.read_all()
    assert len(rows) == n * 4
    chk = log.verify_chain()
    assert chk.ok
    assert chk.total == n * 4


def test_tamper_breaks_chain(tmp_path):
    path = str(tmp_path / "events.jsonl")
    log = OrderEventLog(path)
    for _ in range(5):
        o = _make_filled_order()
        log.append(o, EventType.SIGNAL)
        log.append(o, EventType.FILL)

    assert log.verify_chain().ok

    with open(path, "r") as fh:
        lines = fh.readlines()
    row = json.loads(lines[4])
    row["order"]["qty"] = 999
    lines[4] = json.dumps(row) + "\n"
    with open(path, "w") as fh:
        fh.writelines(lines)

    chk = log.verify_chain()
    assert not chk.ok
    assert chk.broken_at_seq is not None


def test_resume_continues_chain(tmp_path):
    path = str(tmp_path / "events.jsonl")
    log1 = OrderEventLog(path)
    o = _make_filled_order()
    log1.append(o, EventType.SIGNAL)
    log1.append(o, EventType.FILL)

    log2 = OrderEventLog(path)  # reopen — resume seq + last hash
    o2 = _make_filled_order()
    log2.append(o2, EventType.SIGNAL)
    log2.append(o2, EventType.FILL)

    chk = log2.verify_chain()
    assert chk.ok
    assert chk.total == 4
