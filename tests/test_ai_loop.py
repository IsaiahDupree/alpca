"""Offline tests for the AI research loop pieces: regime detector + constrained strategy generator."""

import random

from alpca.ai.regime import detect_regime, RegimeState
from alpca.ai.strategy_generator import (
    PARAM_BOUNDS, STRATEGY_SPACE, _clamp, heuristic_proposal, propose, run_proposal, ai_proposal)


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
        for k, (lo, hi) in PARAM_BOUNDS.items():
            assert lo <= c["params"][k] <= hi


def test_clamp_bounds_params():
    c = _clamp({"lookback": 9999, "hold": -5, "top_k": 1.7})
    assert c["lookback"] == PARAM_BOUNDS["lookback"][1]
    assert c["hold"] == PARAM_BOUNDS["hold"][0]
    assert PARAM_BOUNDS["top_k"][0] <= c["top_k"] <= PARAM_BOUNDS["top_k"][1]


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
