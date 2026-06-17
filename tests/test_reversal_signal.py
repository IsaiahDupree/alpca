"""Invariants for short_horizon_return_signal (used for the reversal factor, Case 53 — rejected as a
survivorship artifact, but the signal itself is a correct, reusable, no-lookahead trailing-return rank)."""

import numpy as np

from alpca.backtest.factor import short_horizon_return_signal


def test_trailing_return_value_and_no_current_bar_leak():
    # one symbol, prices 100,101,...; the 5-day trailing return at t uses price[t]/price[t-5]-1
    T = 12
    price = np.array([[100.0 * (1.01 ** i)] for i in range(T)])   # (T, 1)
    sig = short_horizon_return_signal(5)(list(range(T)), ["A"], price)
    assert np.all(np.isnan(sig[:5]))                              # not defined before `window` bars
    # at t=5: price[5]/price[0]-1 = 1.01^5 - 1
    assert abs(sig[5, 0] - (1.01 ** 5 - 1)) < 1e-9
    # the signal at t is built from price[t] and price[t-5] only — moving a FUTURE price doesn't change it
    price2 = price.copy(); price2[10, 0] *= 2.0
    sig2 = short_horizon_return_signal(5)(list(range(T)), ["A"], price2)
    assert abs(sig[5, 0] - sig2[5, 0]) < 1e-12                    # t=5 signal unaffected by t=10 change


def test_loser_ranks_below_winner():
    T = 10
    # col 0 = faller (loser), col 1 = riser (winner)
    price = np.zeros((T, 2))
    for i in range(T):
        price[i, 0] = 100.0 * (0.98 ** i)    # falling
        price[i, 1] = 100.0 * (1.02 ** i)    # rising
    sig = short_horizon_return_signal(3)(list(range(T)), ["LOSER", "WINNER"], price)
    t = T - 1
    assert sig[t, 0] < 0 < sig[t, 1]         # loser has negative trailing return, winner positive
    # reversal uses long_high=False -> longs the LOW signal (the loser), which is the intended trade
