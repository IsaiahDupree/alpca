from alpca.execution.fills import FillModel


def test_flat_model_matches_legacy_bps():
    fm = FillModel.flat(5.0)
    buy = fm.fill(True, 100.0, 10)
    sell = fm.fill(False, 100.0, 10)
    # exactly +/-5 bps, no rounding, full fill
    assert abs(buy.price - 100.05) < 1e-9
    assert abs(sell.price - 99.95) < 1e-9
    assert buy.filled_qty == 10
    assert not buy.capped


def test_spread_is_side_aware_and_adverse():
    fm = FillModel(half_spread_bps=2.0, impact_coef_bps=0.0, participation_cap=1.0, min_tick=0.0)
    buy = fm.fill(True, 50.0, 1)
    sell = fm.fill(False, 50.0, 1)
    assert buy.price > 50.0           # buy pays up
    assert sell.price < 50.0          # sell receives less
    assert abs(buy.slippage_bps - 2.0) < 1e-9


def test_impact_grows_with_participation():
    fm = FillModel(half_spread_bps=1.0, impact_coef_bps=10.0, participation_cap=1.0, min_tick=0.0)
    small = fm.fill(True, 100.0, 100, bar_volume=100_000)    # 0.1% participation
    big = fm.fill(True, 100.0, 10_000, bar_volume=100_000)   # 10% participation
    assert big.slippage_bps > small.slippage_bps
    # sqrt law: 100x participation -> 10x impact term
    small_impact = small.slippage_bps - 1.0
    big_impact = big.slippage_bps - 1.0
    assert abs(big_impact / small_impact - 10.0) < 0.5


def test_volume_cap_produces_partial_fill():
    fm = FillModel(half_spread_bps=1.0, impact_coef_bps=0.0, participation_cap=0.10, min_tick=0.0)
    res = fm.fill(True, 100.0, 50_000, bar_volume=100_000)  # want 50k, cap 10% = 10k
    assert res.capped
    assert res.filled_qty == 10_000


def test_no_cap_when_participation_cap_one():
    fm = FillModel(half_spread_bps=1.0, impact_coef_bps=0.0, participation_cap=1.0, min_tick=0.0)
    res = fm.fill(True, 100.0, 50_000, bar_volume=100_000)
    assert not res.capped
    assert res.filled_qty == 50_000


def test_tick_rounding():
    fm = FillModel(half_spread_bps=3.3, impact_coef_bps=0.0, participation_cap=1.0, min_tick=0.01)
    res = fm.fill(True, 100.0, 1)
    # 100 * (1+0.00033) = 100.033 -> rounds to 100.03
    assert res.price == 100.03


def test_zero_qty_is_noop():
    fm = FillModel()
    res = fm.fill(True, 100.0, 0)
    assert res.filled_qty == 0.0
