"""Invariants for cross_sectional_seasonality_signal — strict no-lookahead + ranks a seasonal synthetic.

The EDGE was rejected (Case 48: regime-unstable, the combiner lift was a partial-2026 artifact), but the
SIGNAL is a correct, reusable, no-lookahead factor — these lock in that it (a) never uses the current
month's own returns, and (b) ranks a name with a strong recurring January higher in January.
"""

import datetime
import math

import numpy as np

from alpca.backtest.factor import cross_sectional_seasonality_signal, _price_ret


def _bars(closes, start=1_600_000_000, step=86400):
    return [{"timestamp": start + i * step, "close": c} for i, c in enumerate(closes)]


def test_no_signal_until_a_prior_same_month_exists():
    """In the FIRST occurrence of a calendar month there is no prior-year data -> signal is NaN."""
    # ~3 months of daily bars starting Jan; first Jan days can't have a prior Jan
    closes = [100 * (1.001 ** i) for i in range(70)]
    bars = {"A": _bars(closes), "B": _bars([100 * (1.0005 ** i) for i in range(70)])}
    syms = sorted(bars)
    master = sorted({int(b["timestamp"]) for s in syms for b in bars[s]})
    price, _ = _price_ret(bars, syms, master)
    sig = cross_sectional_seasonality_signal(min_prior=5)(master, syms, price)
    assert np.all(np.isnan(sig[:25]))                       # no prior same-month data early on


def test_ranks_recurring_seasonal_name_higher_in_its_strong_month():
    """A name that jumps every January should, by the next January, carry a higher seasonality signal
    than a flat name — using ONLY prior Januaries (no look-ahead)."""
    rng = np.random.RandomState(0)
    days = 800
    base = datetime.datetime(2021, 1, 1, tzinfo=datetime.timezone.utc)
    ts = [int((base + datetime.timedelta(days=i)).timestamp()) for i in range(days)]
    seasonal, flat = [100.0], [100.0]
    for i in range(1, days):
        mon = datetime.datetime.fromtimestamp(ts[i], datetime.timezone.utc).month
        bump = 0.004 if mon == 1 else -0.0004      # strong every January, slight drift otherwise
        seasonal.append(seasonal[-1] * (1 + bump))
        flat.append(flat[-1] * (1 + 0.0))
    bars = {"SEAS": [{"timestamp": ts[i], "close": seasonal[i]} for i in range(days)],
            "FLAT": [{"timestamp": ts[i], "close": flat[i]} for i in range(days)]}
    syms = sorted(bars)
    master = sorted({int(b["timestamp"]) for s in syms for b in bars[s]})
    price, _ = _price_ret(bars, syms, master)
    sig = cross_sectional_seasonality_signal(min_prior=10)(master, syms, price)
    jcol = syms.index("SEAS"); fcol = syms.index("FLAT")
    # find a January day in 2022+ where the signal is defined for both
    jan_days = [t for t in range(len(master))
                if datetime.datetime.fromtimestamp(master[t], datetime.timezone.utc).month == 1
                and datetime.datetime.fromtimestamp(master[t], datetime.timezone.utc).year >= 2022
                and np.isfinite(sig[t, jcol]) and np.isfinite(sig[t, fcol])]
    assert jan_days, "no defined January signal found"
    t = jan_days[0]
    assert sig[t, jcol] > sig[t, fcol]                     # seasonal name ranks higher in January
