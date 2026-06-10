"""Invariants for the funding-rate data layer (alpca/data/funding.py). No network — the
live Kraken fetch is exercised by scripts/test_funding_tilt.py, not the unit suite."""

from alpca.data.funding import daily_funding


def _row(ts, rate):
    return {"ts": ts, "rate": rate}


def test_daily_funding_sums_by_utc_date():
    # 2021-01-01 00:00 and 08:00 UTC -> same date, summed; next day separate
    d0 = 1_609_459_200  # 2021-01-01T00:00:00Z
    rows = [_row(d0, 0.001), _row(d0 + 8 * 3600, 0.002), _row(d0 + 86400, -0.0005)]
    out = daily_funding(rows)
    assert abs(out["2021-01-01"] - 0.003) < 1e-12
    assert abs(out["2021-01-02"] - (-0.0005)) < 1e-12


def test_daily_funding_empty():
    assert daily_funding([]) == {}


def test_daily_funding_sign_preserved():
    d0 = 1_609_459_200
    out = daily_funding([_row(d0, -0.01), _row(d0 + 3600, -0.02)])
    assert out["2021-01-01"] < 0
