"""
Phase-4: lock the calibrated microprice deadband (k=0.5) and document the
deliberate decision to leave L1OFI's bar-level deadband heuristic.

k=0.5 is the 75th percentile of |microprice tilt| fit from 972k real IEX NBBO
quotes (SPY/QQQ/AAPL, 9 trading days x 3 regular-session windows) — see
scripts/analyze_microstructure.py + data/microstructure_deadbands.json. The tilt is
an instantaneous book quantity (identical on a tick or a bar's attached NBBO), so
the tick fit applies directly. L1OFI rolls over 20 BARS not 20 ticks, so its fit is
NOT transferable and intentionally stays at the heuristic 0.15.
"""

import json
from pathlib import Path

from alpca.strategies.microstructure import MicropriceGate, MicropriceTilt, gate
from alpca.strategies.order_flow import L1OFI
from alpca.strategies.base import Strategy


class _Dummy(Strategy):
    name = "dummy"

    def on_bar(self, bar):
        from alpca.strategies.base import hold
        return hold()


def test_microprice_gate_default_k_is_calibrated():
    assert MicropriceGate(_Dummy()).k == 0.5
    assert gate(_Dummy()).k == 0.5


def test_microprice_tilt_default_k_is_calibrated():
    assert MicropriceTilt().k == 0.5


def test_l1ofi_deadband_is_bar_level_calibrated():
    # bar-level OFI is now CALIBRATED from full-session contiguous qbars
    # (scripts/analyze_ofi_bars.py): entry = mean p90, exit = mean p50 of the real
    # 20-bar-window |normOFI|. NOT the tick-level 0.10 (~2x too low; different scale).
    s = L1OFI()
    assert s.entry == 0.19
    assert s.exit == 0.08


def test_ofi_deadbands_json_is_the_full_session_fit_if_present():
    p = Path(__file__).resolve().parents[1] / "data" / "ofi_deadbands.json"
    if not p.exists():
        return
    d = json.loads(p.read_text())
    assert d["window_bars"] == 20
    syms = {s["symbol"]: s for s in d["symbols"]}
    assert {"SPY", "QQQ", "AAPL"} <= set(syms)
    for sym, s in syms.items():
        assert s["n_ofi"] > 1000, f"{sym} bar-OFI fit thin ({s['n_ofi']})"
        assert 0.10 <= s["recommend_entry"] <= 0.30, f"{sym} entry={s['recommend_entry']}"


def test_deadbands_json_is_the_multiday_fit_if_present():
    # If the committed fit is present, it must be the real multi-day sample
    # (large clean tick counts + k recommendation near 0.5), not the thin seed.
    p = Path(__file__).resolve().parents[1] / "data" / "microstructure_deadbands.json"
    if not p.exists():
        return  # fit artifact not present in this checkout — nothing to assert
    d = json.loads(p.read_text())
    syms = {s["symbol"]: s for s in d["symbols"]}
    assert {"SPY", "QQQ", "AAPL"} <= set(syms)
    for sym, s in syms.items():
        assert s["clean"] > 100_000, f"{sym} fit is thin ({s['clean']} ticks) — re-sample"
        assert 0.3 <= s["recommend_microprice_k"] <= 0.8, f"{sym} k={s['recommend_microprice_k']}"
