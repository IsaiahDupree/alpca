"""Invariants for delisting_aware_walkforward — the survivorship-honest pairs walk-forward.

Key property: on a survivor-only universe (every name spans the full calendar) it must reproduce the
legacy `walkforward_pairs` exactly; and it must RUN (not collapse to 0 windows) when a name has only
partial history — the bug that the legacy global-intersection design has.
"""

import math
import random

from alpca.backtest.pairs import walkforward_pairs, delisting_aware_walkforward


def _cointegrated_universe(n_extra=6, n_days=900, seed=3, delist_at=None):
    """A few mean-reverting pairs (each = common factor + stationary spread) + optional one name whose
    bars STOP at `delist_at` (a delisting). Returns bars_by_sym."""
    rng = random.Random(seed)
    base = 1_600_000_000
    bars = {}
    # common random-walk factors, each shared by a pair -> the two legs cointegrate
    for g in range(4):
        f = 100.0
        path = []
        for i in range(n_days):
            f *= (1 + rng.gauss(0, 0.01))
            path.append(f)
        for leg in ("A", "B"):
            sp = 0.0
            rows = []
            for i in range(n_days):
                sp = 0.9 * sp + rng.gauss(0, 0.5)      # stationary spread -> mean-reverting
                px = path[i] * (1 + 0.02 * sp / 100.0) + (2.0 if leg == "B" else 0.0)
                rows.append({"timestamp": base + i * 86400, "close": max(px, 1.0)})
            bars[f"G{g}{leg}"] = rows
    # filler independent names
    for j in range(n_extra):
        p, rows = 50.0, []
        for i in range(n_days):
            p *= (1 + rng.gauss(0, 0.012))
            rows.append({"timestamp": base + i * 86400, "close": max(p, 1.0)})
        bars[f"F{j:02d}"] = rows
    if delist_at is not None:
        # a name that delists partway through (bars stop) — must not break the walk-forward
        p, rows = 40.0, []
        for i in range(delist_at):
            p *= (1 + rng.gauss(0, 0.013))
            rows.append({"timestamp": base + i * 86400, "close": max(p, 1.0)})
        bars["DEAD"] = rows
    return bars


def test_reproduces_legacy_on_survivor_universe():
    bars = _cointegrated_universe()
    legacy = walkforward_pairs(bars, train=252, test=63, top_n=6, max_adf=None)
    aware = delisting_aware_walkforward(bars, train=252, test=63, top_n=6, max_adf=None)
    assert aware.n_windows == legacy.n_windows and legacy.n_windows > 0
    assert math.isclose(aware.sharpe, legacy.sharpe, abs_tol=1e-6)


def test_runs_with_a_delisted_name_where_legacy_collapses():
    bars = _cointegrated_universe(delist_at=500)            # DEAD stops at bar 500
    legacy = walkforward_pairs(bars, train=252, test=63, top_n=6, max_adf=None)
    aware = delisting_aware_walkforward(bars, delisted_syms={"DEAD"}, train=252, test=63,
                                        top_n=6, max_adf=None)
    # legacy intersects timestamps -> the partial-history name truncates the whole calendar to <500 bars
    assert legacy.n_windows < aware.n_windows                # aware keeps trading on the union calendar
    assert aware.n_windows > 0 and len(aware.equity_curve) > 1
    assert all(math.isfinite(e) for e in aware.equity_curve)


def test_delisted_accounting_only_counts_listed_names():
    bars = _cointegrated_universe(delist_at=500)
    aware = delisting_aware_walkforward(bars, delisted_syms={"DEAD"}, train=252, test=63, top_n=6,
                                        max_adf=None)
    assert set(aware.delisted_names_traded) <= {"DEAD"}      # only the flagged name can be counted
    assert aware.delisted_leg_trades >= 0
