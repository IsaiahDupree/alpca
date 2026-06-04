from alpca.execution.fees import ZERO_FEES, AlpacaFeeModel


def test_buy_has_no_regulatory_fee():
    fm = AlpacaFeeModel()
    assert fm.fee(side_buy=True, qty=1000, price=100.0) == 0.0


def test_sell_charges_sec_and_taf():
    fm = AlpacaFeeModel()
    qty, price = 1000, 100.0   # $100k sell
    fee = fm.fee(side_buy=False, qty=qty, price=price)
    sec = 100_000 * 27.80e-6   # = 2.78
    taf = min(1000 * 0.000166, 8.30)  # = 0.166
    assert abs(fee - (sec + taf)) < 1e-9


def test_taf_is_capped():
    fm = AlpacaFeeModel()
    # 1,000,000 shares -> raw TAF 166.0 but capped at 8.30
    reg = fm.regulatory(side_buy=False, qty=1_000_000, price=1.0)
    sec = 1_000_000 * 1.0 * 27.80e-6
    assert abs(reg - (sec + 8.30)) < 1e-6


def test_zero_fee_model():
    assert ZERO_FEES.fee(side_buy=False, qty=1000, price=100.0) == 0.0


def test_commission_configurable():
    fm = AlpacaFeeModel(commission_per_share=0.005, commission_min=1.0)
    # 10 shares -> raw 0.05 -> floored to 1.0
    assert abs(fm.commission(10) - 1.0) < 1e-9
    # 1000 shares -> 5.0 (above floor)
    assert abs(fm.commission(1000) - 5.0) < 1e-9
    # default model: commission-free
    assert AlpacaFeeModel().commission(1000) == 0.0
