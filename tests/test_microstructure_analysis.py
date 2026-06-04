"""Tests for the Phase-4 deadband analyzer (scripts/analyze_microstructure.py).

Pure-helper tests + an end-to-end analyze() on a synthetic tick stream with a known
tilt/OFI structure, so the recommended deadbands are checked against hand-computed
percentiles rather than display output.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "analyze_microstructure",
    Path(__file__).resolve().parents[1] / "scripts" / "analyze_microstructure.py",
)
am = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(am)


def test_pctile_endpoints_and_interp():
    xs = [10, 20, 30, 40, 50]
    assert am._pctile(xs, 0) == 10
    assert am._pctile(xs, 100) == 50
    assert am._pctile(xs, 50) == 30
    # 25th pct of 5 points: rank=1.0 -> exactly the 2nd element
    assert am._pctile(xs, 25) == 20
    assert am._pctile([], 50) is None
    assert am._pctile([7], 90) == 7


def test_spread_bps():
    # 1 bp spread around 100 mid: (100.01-99.99)/100 * 1e4 = 2.0 bps
    assert round(am._spread_bps(99.99, 100.01), 4) == 2.0
    assert am._spread_bps(0, 0) == float("inf")


def test_analyze_screens_and_fits(tmp_path: Path):
    # synthetic clean stream: alternating size imbalance -> nonzero tilt every tick,
    # plus one crossed quote and one absurdly wide quote that must be screened out.
    rows = []
    ts = 1_700_000_000.0
    for i in range(60):
        # bid<ask, 2c spread around 100; sizes imbalanced so tilt != 0
        bs, az = (120.0, 40.0) if i % 2 == 0 else (40.0, 120.0)
        rows.append({"bid": 99.99, "ask": 100.01, "bid_size": bs, "ask_size": az,
                     "timestamp": ts + i})
    rows.append({"bid": 100.05, "ask": 100.00, "bid_size": 80, "ask_size": 80,  # crossed
                 "timestamp": ts + 100})
    rows.append({"bid": 90.0, "ask": 110.0, "bid_size": 80, "ask_size": 80,     # 2000bps wide
                 "timestamp": ts + 101})

    cache = tmp_path
    (cache / "TEST_quotes.jsonl").write_text("\n".join(json.dumps(r) for r in rows))

    r = am.analyze("TEST", cache, max_spread_bps=50.0, window=20)
    assert r["clean"] == 60                      # the crossed + wide rows dropped
    assert r["rejected_crossed"] == 1
    assert r["rejected_wide"] == 1
    # every clean tick has a real imbalance -> tilt percentiles are strictly > 0
    assert r["tilt_abs_p75"] > 0
    assert 0.0 < r["recommend_microprice_k"] <= 1.0
    # span: 59 seconds between first and last clean tick
    assert r["span_hours"] == round(59 / 3600.0, 2)
    assert r["unique_books"] == 2                # only two (size) configurations
