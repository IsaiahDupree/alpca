"""Tests for the Haiku per-regime gate — deterministic rail + the model-can't-override-the-data veto."""

from alpca.ai.strategy_gate import classify_regime, falsification_gate, haiku_verdict, gate


def _spy(trend, vol, n=80, base=1_600_000_000):
    import random
    rng = random.Random(0)
    p, bars = 100.0, []
    drift = trend / n
    for i in range(n):
        p *= (1 + drift + rng.gauss(0, vol))
        bars.append({"timestamp": base + i * 86400, "close": p})
    return bars


class _MockRouter:
    """Stand-in for AIRouter.small — returns a scripted JSON verdict (no network/key)."""
    def __init__(self, verdict="GO", extra=""):
        self.verdict = verdict
        self.calls = []

    def small(self, prompt, **k):
        self.calls.append(prompt)
        return f'{{"verdict":"{self.verdict}","confidence":0.7,"rationale":"ok","biggest_risk":"x","regime_fit":"y"}}'


def test_classify_regime():
    assert classify_regime(_spy(0.30, 0.008)) == "bull"
    assert classify_regime(_spy(-0.30, 0.008)) == "bear"
    assert classify_regime(_spy(0.0, 0.004), trend_thr=0.04) == "chop"
    assert classify_regime(_spy(0.10, 0.05)) == "high_vol"     # high vol dominates
    assert classify_regime([{"close": 1.0}]) == "unknown"


def _good():   # clears every bar
    return {"fresh_holdout_sharpe": 0.6, "cost_2bps_sharpe": 0.7, "dsr": 0.95,
            "per_year": {2021: 0.8, 2022: 0.3, 2023: 0.9, 2024: 0.4, 2025: 0.5, 2026: 0.6}}


def _overfit():  # the EAR-PEAD/accruals failure mode: great in-sample, fresh holdout negative
    return {"fresh_holdout_sharpe": -0.5, "cost_2bps_sharpe": 0.7, "dsr": 0.95,
            "per_year": {2021: 1.8, 2022: 0.2, 2023: 1.4, 2024: 0.1, 2025: 0.8, 2026: 1.5}}


def test_falsification_gate_pass_and_fail():
    assert falsification_gate(_good()).passed is True
    g = falsification_gate(_overfit())
    assert g.passed is False
    assert g.checks["fresh_symbol_holdout"] is False
    assert any("fresh-symbol holdout" in r for r in g.reasons)


def test_haiku_verdict_parses_json():
    v = haiku_verdict(_good(), "bull", _MockRouter("GO"))
    assert v["verdict"] == "GO" and "rationale" in v


def test_gate_data_veto_overrides_haiku_go():
    # even if Haiku says GO, an overfit candidate (fails fresh-symbol holdout) must be NO-GO
    d = gate(_overfit(), _spy(0.2, 0.008), _MockRouter("GO"))
    assert d["falsification_pass"] is False
    assert d["decision"] == "NO-GO"            # the data rail vetoes the model


def test_gate_go_requires_both_rail_and_haiku():
    assert gate(_good(), _spy(0.2, 0.008), _MockRouter("GO"))["decision"] == "GO"
    # rail passes but Haiku flags a concern -> NO-GO (both must concur)
    assert gate(_good(), _spy(0.2, 0.008), _MockRouter("NO-GO"))["decision"] == "NO-GO"


def test_gate_failclosed_on_unparseable_haiku():
    class Bad:
        def small(self, p, **k):
            return "I think this looks promising but I'm not sure."
    d = gate(_good(), _spy(0.2, 0.008), Bad())
    assert d["haiku"]["verdict"] == "NO-GO" and d["decision"] == "NO-GO"   # fail-closed
