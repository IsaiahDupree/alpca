"""Invariants for the live pairs-portfolio deploy layer."""

import math
import random

from alpca.live.pairs_portfolio import (
    compute_pairs_book, half_kelly_leverage, size_book)


def _coint_universe(n_days=420, seed=0):
    """Two cointegrated series (B drives A with a stationary mean-reverting spread) + noise names.
    The spread is pushed to a wide extreme at the end so the pair should fire an entry today."""
    rng = random.Random(seed)
    base = 1_600_000_000
    common = [0.0]
    for _ in range(n_days):
        common.append(common[-1] + rng.gauss(0, 0.01))     # shared stochastic trend
    spread = 0.0
    a_log, b_log = [], []
    for i in range(n_days):
        spread = 0.85 * spread + rng.gauss(0, 0.01)         # mean-reverting (half-life ~ a few days)
        b_log.append(4.6 + common[i + 1])
        a_log.append(4.6 + common[i + 1] + spread)
    a_log[-1] += 0.06                                       # shove the spread wide on the last bar -> entry
    bars = {}
    for name, logs in (("AAA", a_log), ("BBB", b_log)):
        bars[name] = [{"timestamp": base + i * 86400, "close": math.exp(logs[i])} for i in range(n_days)]
    for j in range(4):                                      # unrelated noise names
        p, rows = 100.0, []
        for i in range(n_days):
            p *= (1 + rng.gauss(0, 0.012))
            rows.append({"timestamp": base + i * 86400, "close": p})
        bars[f"N{j}"] = rows
    return bars


def test_finds_cointegrated_pair_and_fires_entry():
    bars = _coint_universe()
    book = compute_pairs_book(bars, train=400, top_n=6, lookback=40, entry_z=1.5,
                              max_half_life=40, min_half_life=2)
    pair_names = {(t.a, t.b) for t in book.targets}
    assert ("AAA", "BBB") in pair_names or ("BBB", "AAA") in pair_names   # the real pair is screened
    assert book.n_active >= 1                                              # at the extreme it fires
    assert abs(sum(abs(w) for w in book.weights.values()) - 1.0) < 1e-9    # gross-normalized to 1


def test_weights_are_equal_dollar_long_short():
    bars = _coint_universe()
    book = compute_pairs_book(bars, train=400, top_n=6, lookback=40, entry_z=1.5, max_half_life=40)
    if book.n_active:
        net = sum(book.weights.values())
        assert abs(net) < 0.4            # roughly dollar-neutral (legs offset; not exact w/ multiple pairs)
        assert any(w > 0 for w in book.weights.values()) and any(w < 0 for w in book.weights.values())


def test_hysteresis_holds_while_z_outside_exit_band():
    bars = _coint_universe()
    first = compute_pairs_book(bars, train=400, top_n=6, lookback=40, entry_z=1.5, exit_z=0.5,
                               max_half_life=40)
    # same data + prior state: a position entered at a wide z stays on while z is still outside
    # the (low) exit band — the no-churn hysteresis property.
    again = compute_pairs_book(bars, train=400, top_n=6, lookback=40, entry_z=1.5, exit_z=0.5,
                               max_half_life=40, prior_state=first.state)
    assert again.state == first.state        # nothing flips when the signal hasn't moved


def test_exit_logic_flattens_inside_band():
    # a short-spread (prev=-1) flattens once z falls to/below exit_z (unit test of the state machine)
    bars = _coint_universe()
    # prior says we're short-spread on the real pair; re-run with a HUGE exit_z so z<=exit_z triggers exit
    book = compute_pairs_book(bars, train=400, top_n=6, lookback=40, entry_z=1.5, exit_z=99.0,
                              max_half_life=40, prior_state={"AAA|BBB": -1})
    st = {f"{t.a}|{t.b}": t.state for t in book.targets}
    if "AAA|BBB" in st:
        assert st["AAA|BBB"] == 0            # z <= 99 -> exit to flat


def test_half_kelly_leverage_bounds():
    assert half_kelly_leverage(0.0, 0.05) == 0.0          # no edge -> no leverage
    assert half_kelly_leverage(-0.5, 0.05) == 0.0         # negative edge -> none
    assert half_kelly_leverage(1.0, 0.05, cap=2.0) == 2.0  # 0.5*1.0/0.05 = 10 -> capped
    assert 0 < half_kelly_leverage(0.29, 0.034, cap=5.0) < 5.0


def test_size_book_respects_vol_target_and_cap():
    bars = _coint_universe()
    book = compute_pairs_book(bars, train=400, top_n=6, lookback=40, entry_z=1.5, max_half_life=40)
    sized = size_book(book, basket_sharpe=0.29, ann_vol=0.034, target_vol=0.05, kelly_fraction=0.5, cap=1.0)
    g = sum(abs(w) for w in sized.values())
    assert g <= 1.0 + 1e-9                                  # never exceeds the cap
    flat = size_book(book, basket_sharpe=-0.5, ann_vol=0.034)   # negative edge -> zero book
    assert sum(abs(w) for w in flat.values()) == 0.0
