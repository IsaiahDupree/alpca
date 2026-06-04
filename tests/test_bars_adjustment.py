"""
Bar adjustment plumbing (no network): verify the string->enum mapping and that
the requested adjustment is recorded. Live fetch is exercised by scripts, not
unit tests (no credentials in CI).
"""

import pytest

from alpca.data.bars import _ADJUSTMENTS, _to_alpaca_adjustment


def test_all_adjustments_map_to_enum():
    from alpaca.data.enums import Adjustment
    for name in _ADJUSTMENTS:
        a = _to_alpaca_adjustment(name)
        assert isinstance(a, Adjustment)
        assert a.value == name


def test_default_is_all():
    from alpaca.data.enums import Adjustment
    assert _to_alpaca_adjustment("all") == Adjustment.ALL
    assert _to_alpaca_adjustment(None) == Adjustment.ALL  # None -> default "all"


def test_raw_for_parity():
    from alpaca.data.enums import Adjustment
    assert _to_alpaca_adjustment("raw") == Adjustment.RAW


def test_bad_adjustment_rejected():
    with pytest.raises(ValueError):
        _to_alpaca_adjustment("adjusted-somehow")
