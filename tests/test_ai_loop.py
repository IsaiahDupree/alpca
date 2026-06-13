"""Offline tests for the AI research loop pieces: regime detector + constrained strategy generator."""

import random

from alpca.ai.regime import detect_regime, RegimeState
from alpca.ai.strategy_generator import (
    STRATEGY_SPACE, _clamp, heuristic_proposal, propose, run_proposal, ai_proposal)


def _series(trend, vol, n=90, base=1_600_000_000):
    rng = random.Random(0)
    p, bars = 100.0, []
    for i in range(n):
        p *= (1 + trend / n + rng.gauss(0, vol))
        bars.append({"timestamp": base + i * 86400, "close": p})
    return bars


def _universe(n_sym=12, n_days=320, seed=1):
    rng = random.Random(seed)
    base = 1_600_000_000
    u = {}
    for j in range(n_sym):
        p, bars = 100.0, []
        for i in range(n_days):
            p *= (1 + rng.gauss(0.0003, 0.012))
            bars.append({"timestamp": base + i * 86400, "close": p})
        u[f"S{j:02d}"] = bars
    return u


def test_detect_regime_labels():
    assert detect_regime(_series(0.30, 0.008)).label == "bull"
    assert detect_regime(_series(-0.30, 0.008)).label == "bear"
    assert detect_regime(_series(0.0, 0.004)).label == "chop"
    assert detect_regime(_series(0.1, 0.05)).label == "high_vol"
    r = detect_regime(_series(0.3, 0.008))
    assert isinstance(r, RegimeState) and r.n > 0 and "regime=" in r.as_prompt()


def test_heuristic_proposal_valid_for_every_regime():
    for reg in ("bull", "bear", "chop", "high_vol", "unknown"):
        c = heuristic_proposal(reg)
        assert c["strategy_type"] in STRATEGY_SPACE
        bounds = STRATEGY_SPACE[c["strategy_type"]]["params"]
        for k, (lo, hi) in bounds.items():
            assert lo <= c["params"][k] <= hi          # each param within its template's bounds


def test_clamp_bounds_params_per_template():
    c = _clamp("xsec_momentum", {"lookback": 9999, "hold": -5, "top_k": 1.7})
    b = STRATEGY_SPACE["xsec_momentum"]["params"]
    assert c["lookback"] == b["lookback"][1] and c["hold"] == b["hold"][0]
    a = _clamp("accruals", {"top_frac_pct": 99})
    assert a["top_frac_pct"] == STRATEGY_SPACE["accruals"]["params"]["top_frac_pct"][1]


def test_space_includes_fundamental_accruals():
    assert STRATEGY_SPACE["accruals"]["family"] == "fundamental"
    assert "top_frac_pct" in STRATEGY_SPACE["accruals"]["params"]


def test_propose_falls_back_to_heuristic_without_router():
    c = propose("regime=chop", "chop", router=None)
    assert c["source"] == "heuristic" and c["strategy_type"] in STRATEGY_SPACE


def test_ai_proposal_validates_and_rejects_junk():
    class GoodRouter:
        def think(self, p, **k):
            return 'sure: {"strategy_type":"xsec_reversal","params":{"lookback":40,"hold":8,"top_k":12},"rationale":"chop reverts"}'

    class JunkRouter:
        def think(self, p, **k):
            return '{"strategy_type":"make_money_fast","params":{}}'   # not in the space

    good = ai_proposal("regime=chop", GoodRouter())
    assert good["strategy_type"] == "xsec_reversal" and good["params"]["lookback"] == 40 and good["source"] == "ai"
    assert ai_proposal("regime=chop", JunkRouter()) is None          # invalid type -> rejected


def test_run_proposal_produces_gate_ready_result():
    main, fresh = _universe(seed=1), _universe(n_sym=8, seed=2)
    cfg = heuristic_proposal("bull")
    r = run_proposal(cfg, main, fresh)
    for key in ("sharpe", "oos_sharpe", "cost_2bps_sharpe", "fresh_holdout_sharpe", "per_year", "dsr"):
        assert key in r
    assert isinstance(r["fresh_holdout_sharpe"], float)
    assert isinstance(r["per_year"], dict)


def _funds(syms, base=1_600_000_000):
    import datetime
    out = {}
    for j, s in enumerate(syms):
        filed = datetime.datetime.fromtimestamp(base + 40 * 86400, datetime.timezone.utc).strftime("%Y-%m-%d")
        fyend = datetime.datetime.fromtimestamp(base + 5 * 86400, datetime.timezone.utc).strftime("%Y-%m-%d")
        out[s] = [{"fy_end": fyend, "filed": filed, "net_income": (j - len(syms) / 2) * 1e8,
                   "cfo": 0.0, "total_assets": 1e9}]
    return out


def test_run_proposal_handles_fundamental_accruals_template():
    main, fresh = _universe(seed=1), _universe(n_sym=8, seed=2)
    fm, ff = _funds(list(main)), _funds(list(fresh))
    cfg = {"strategy_type": "accruals", "params": {"top_frac_pct": 20}, "source": "heuristic"}
    r = run_proposal(cfg, main, fresh, fund_main=fm, fund_fresh=ff)
    assert r["strategy_type"] == "accruals" and "error" not in r
    assert isinstance(r["fresh_holdout_sharpe"], float) and isinstance(r["dsr"], float)


def test_run_proposal_fundamental_without_funds_flags_no_data():
    main = _universe(seed=1)
    cfg = {"strategy_type": "accruals", "params": {"top_frac_pct": 20}}
    r = run_proposal(cfg, main, {}, fund_main=None, fund_fresh=None)
    assert r.get("error") == "no_data"        # gracefully degrades, doesn't crash
