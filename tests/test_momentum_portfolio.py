"""Invariants for the live momentum-sleeve book (long winners + SPY index hedge, borrow-free)."""

import math
import random

from alpca.live.momentum_portfolio import compute_momentum_book, size_book, MomentumBook


def _universe(n_sym=20, n_days=400, seed=1):
    """Synthetic: symbol j has drift proportional to j, so high-j names are persistent WINNERS.
    vol-managed momentum should rank them top and the book should go long the high-j names."""
    rng = random.Random(seed)
    base = 1_600_000_000
    bars_by, spy = {}, []
    p_spy = 100.0
    for i in range(n_days):
        p_spy *= (1 + 0.0002 + rng.gauss(0, 0.008))
        spy.append({"timestamp": base + i * 86400, "close": p_spy})
    for j in range(n_sym):
        drift = 0.0004 * (j - n_sym / 2)        # high j = up-trenders (winners), low j = down
        p, bars = 50.0, []
        for i in range(n_days):
            p *= (1 + drift + rng.gauss(0, 0.012))
            bars.append({"timestamp": base + i * 86400, "close": max(p, 0.5)})
        bars_by[f"S{j:02d}"] = bars
    return bars_by, spy


def test_book_longs_winners_and_hedges_spy():
    bars, spy = _universe()
    book = compute_momentum_book(bars, spy, top_frac=0.2, lookback=120, skip=21, vol_window=60)
    assert book.n_winners >= 1
    assert abs(sum(book.longs.values()) - 1.0) < 1e-9          # long leg sums to +1
    assert book.spy_weight == -1.0                              # beta hedge present
    assert book.weights.get("SPY") == -1.0
    # winners should be the HIGH-index (up-trending) names, not the losers
    assert all(int(s[1:]) >= 10 for s in book.longs), book.longs


def test_no_spy_means_no_hedge():
    bars, _ = _universe()
    book = compute_momentum_book(bars, [], top_frac=0.2)
    assert book.spy_weight == 0.0 and "SPY" not in book.weights


def test_insufficient_data_returns_flat():
    bars, spy = _universe(n_days=60)
    book = compute_momentum_book(bars, spy)
    assert book.weights == {} and book.n_winners == 0


def test_size_book_scales_down_modest_edge():
    bars, spy = _universe()
    book = compute_momentum_book(bars, spy)
    sized = size_book(book, sleeve_sharpe=0.23, ann_vol=0.12, target_vol=0.04, kelly_fraction=0.5, cap=0.5)
    lev = max(abs(w) for w in sized.values()) / max(abs(w) for w in book.weights.values())
    assert 0 < lev <= 0.5                                       # capped, modest
    # vol-target: 0.04/0.12 ≈ 0.33 binds below half-Kelly (0.5*0.23/0.12≈0.96) and the 0.5 cap
    assert lev < 0.4


def test_size_book_zero_when_no_edge():
    bars, spy = _universe()
    book = compute_momentum_book(bars, spy)
    assert size_book(book, sleeve_sharpe=0.0, ann_vol=0.12) == {s: 0.0 for s in book.weights}
