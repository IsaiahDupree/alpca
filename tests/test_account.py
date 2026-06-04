"""Unit tests for T+1 SettlementLedger and PdtGuard."""

from alpca.runtime.account import PdtGuard, SettlementLedger


# --------------------------------------------------------------- settlement
def test_sell_proceeds_settle_next_session():
    led = SettlementLedger(10_000.0)          # all settled at session 0
    assert led.available() == 10_000.0
    led.record_buy(4_000.0)                    # spend settled cash
    assert led.available() == 6_000.0
    led.record_sell(5_000.0, current_session=0)  # proceeds pending, settle T+1
    # still session 0: proceeds NOT yet available
    assert led.available() == 6_000.0
    assert led.pending_total == 5_000.0
    assert led.total == 11_000.0               # invariant: settled + pending
    # advance to session 1 (T+1): proceeds settle
    led.advance_to(1)
    assert led.available() == 11_000.0
    assert led.pending_total == 0.0


def test_multiple_sells_settle_on_their_own_dates():
    led = SettlementLedger(0.0)
    led.record_sell(1_000.0, current_session=0)   # settles at 1
    led.record_sell(2_000.0, current_session=1)   # settles at 2
    led.advance_to(1)
    assert led.available() == 1_000.0             # only the first matured
    led.advance_to(2)
    assert led.available() == 3_000.0


def test_settle_lag_configurable():
    led = SettlementLedger(0.0, settle_lag=2)     # T+2
    led.record_sell(1_000.0, current_session=0)
    led.advance_to(1)
    assert led.available() == 0.0                 # not yet (T+2)
    led.advance_to(2)
    assert led.available() == 1_000.0


# --------------------------------------------------------------- PDT
def test_pdt_inactive_above_threshold():
    g = PdtGuard()
    # open then close same symbol same session = a day trade, but equity >= 25k
    g.record_fill(0, "SPY", "BUY")
    ok, _ = g.check(0, "SPY", "SELL", equity=30_000)
    assert ok  # rule doesn't apply above $25k


def test_pdt_blocks_fourth_day_trade_under_threshold():
    g = PdtGuard(min_equity=25_000, max_day_trades=3, window_sessions=5)
    # 3 completed day trades in session 0 (each a buy then sell of distinct syms)
    for sym in ("AAA", "BBB", "CCC"):
        g.record_fill(0, sym, "BUY")
        g.record_fill(0, sym, "SELL")
    assert g.day_trade_count(0) == 3
    # a 4th would-be day trade under $25k is blocked
    g.record_fill(0, "DDD", "BUY")
    ok, reason = g.check(0, "DDD", "SELL", equity=20_000)
    assert not ok
    assert "PDT" in reason


def test_pdt_allows_non_daytrade_even_when_at_cap():
    g = PdtGuard()
    for sym in ("AAA", "BBB", "CCC"):
        g.record_fill(0, sym, "BUY")
        g.record_fill(0, sym, "SELL")
    # a fresh BUY (no opposite side this session) is NOT a day trade -> allowed
    ok, _ = g.check(0, "EEE", "BUY", equity=20_000)
    assert ok


def test_pdt_window_rolls_off_old_day_trades():
    g = PdtGuard(min_equity=25_000, max_day_trades=3, window_sessions=5)
    # 3 day trades back in session 0
    for sym in ("AAA", "BBB", "CCC"):
        g.record_fill(0, sym, "BUY")
        g.record_fill(0, sym, "SELL")
    # by session 5 the window (sessions 1..5) no longer counts session 0
    assert g.day_trade_count(5) == 0
    g.record_fill(5, "DDD", "BUY")
    ok, _ = g.check(5, "DDD", "SELL", equity=20_000)
    assert ok
