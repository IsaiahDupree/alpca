"""
Per-session reset of intraday rolling state (Strategy.on_session_start).

The runner fires on_session_start() at the overnight UTC-day boundary so a strategy's
intraday window never straddles the close. L1OFI overrides it to clear its 20-bar OFI
accumulation; price-based strategies leave it a no-op on purpose.
"""

import asyncio

from alpca.config import RiskConfig
from alpca.data.feed import ReplayBarSource
from alpca.execution.adapters.sim import SimAdapter
from alpca.execution.router import ExecutionRouter
from alpca.risk.risk_engine import RiskEngine
from alpca.runtime.runner import LiveRunner
from alpca.strategies.base import Strategy, hold
from alpca.strategies.order_flow import L1OFI

DAY = 86400
T0 = 1_700_000_000  # a real epoch (2023-11-14), so int(ts//86400) is a real day


def _bar(px, ts, sym="T"):
    return {"open": px, "high": px + 0.5, "low": px - 0.5, "close": px,
            "volume": 1000, "timestamp": ts, "symbol": sym}


def _runner(strat):
    risk = RiskEngine(RiskConfig())
    router = ExecutionRouter(SimAdapter(seed=1, sleep=False), risk, None, fill_timeout_s=1.0)
    return LiveRunner(strat, "T", router, starting_equity=100_000)


class _Counter(Strategy):
    name = "counter"

    def __init__(self):
        super().__init__()
        self.sessions = 0

    def on_bar(self, bar):
        return hold()

    def on_session_start(self):
        self.sessions += 1


# ---- base hook ----
def test_base_on_session_start_is_noop():
    Strategy.on_session_start(object.__new__(_Counter))  # must not raise


# ---- L1OFI unit reset ----
def test_l1ofi_session_start_clears_window():
    s = L1OFI(window=5)
    q = dict(bid=100.0, ask=100.1, bid_size=200.0, ask_size=100.0)
    for i in range(4):
        s.on_bar({"close": 100 + i * 0.01, **q})
    assert s._prev is not None
    assert len(s._e) > 0
    s.on_session_start()
    assert s._prev is None
    assert len(s._e) == 0
    assert len(s._sz) == 0


# ---- L1OFI keeps position across the reset ----
def test_l1ofi_session_start_keeps_side():
    s = L1OFI(window=3)
    s._side = "LONG"
    s._in_position = True
    s.on_session_start()
    assert s._side == "LONG"        # position is NOT force-flattened
    assert s._in_position is True


# ---- runner fires the hook once per overnight boundary ----
def test_runner_fires_hook_once_per_day_boundary():
    strat = _Counter()
    bars = ([_bar(100, T0 + i * 60) for i in range(5)] +          # day 0
            [_bar(101, T0 + DAY + i * 60) for i in range(5)] +    # day 1
            [_bar(102, T0 + 2 * DAY + i * 60) for i in range(5)])  # day 2
    asyncio.run(_runner(strat).run(ReplayBarSource(bars)))
    assert strat.sessions == 2  # two boundaries across three days


def test_runner_no_hook_within_one_session():
    strat = _Counter()
    bars = [_bar(100, T0 + i * 60) for i in range(20)]  # all one UTC day
    asyncio.run(_runner(strat).run(ReplayBarSource(bars)))
    assert strat.sessions == 0


def test_runner_no_hook_on_synthetic_integer_ts():
    # synthetic int ts (0,1,2,...) all map to day 0 -> hook never fires; legacy
    # integer-ts tests must be unaffected.
    strat = _Counter()
    bars = [_bar(100, i) for i in range(50)]
    asyncio.run(_runner(strat).run(ReplayBarSource(bars)))
    assert strat.sessions == 0
