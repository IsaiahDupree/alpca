from alpca.backtest.parity import run_parity
from alpca.data.bars import synthetic_bars


def test_parity_runs_and_quantifies_gap():
    bars = synthetic_bars("DEMO", n=400, seed=6, drift=0.0004, vol=0.013)
    rep = run_parity("donchian", bars, symbol="DEMO",
                     bt_slippage_bps=2.0, live_slippage_bps=4.0, sim_seed=6)

    # both paths acted on the same decisions
    assert rep.bt_n_trades >= 1
    assert rep.live_entries >= 1
    # entries should be in the same ballpark as backtest trades (decision parity)
    assert abs(rep.live_entries - rep.bt_n_trades) <= 2

    # live slippage was set worse than backtest assumed -> positive gap
    assert rep.live_realized_slippage_mean_bps is not None
    assert rep.slippage_gap_bps is not None
    assert rep.slippage_gap_bps > 0

    # report serializes cleanly
    d = rep.to_dict()
    assert d["backtest"]["assumed_slippage_bps"] == 2.0
    assert "return_gap" in d["gap"]
    assert isinstance(rep.render(), str) and "Parity" in rep.render()


def test_parity_worse_slippage_raises_realized_slippage():
    # keltner on this series produces ~24 entries — enough samples for the
    # realized-slippage mean to be statistically meaningful.
    bars = synthetic_bars("DEMO", n=600, seed=5, drift=0.0003, vol=0.013)
    cheap = run_parity("keltner", bars, bt_slippage_bps=2.0, live_slippage_bps=2.0, sim_seed=5)
    dear = run_parity("keltner", bars, bt_slippage_bps=2.0, live_slippage_bps=15.0, sim_seed=5)
    # Decisions are identical (deterministic strategy), so entries match...
    assert cheap.live_entries >= 10
    assert dear.live_entries == cheap.live_entries
    # ...and the only true monotonic invariant is that REALIZED slippage rises.
    # (Total return is NOT monotonic: worse fills shrink equity -> smaller later
    #  position sizes -> can cut losses on subsequent losers, so return can move
    #  either way. We therefore assert on slippage, the execution-quality metric.)
    assert dear.live_realized_slippage_mean_bps > cheap.live_realized_slippage_mean_bps + 5
    assert dear.slippage_gap_bps > cheap.slippage_gap_bps
