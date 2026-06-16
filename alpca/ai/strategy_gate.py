"""
The Haiku per-regime gate, behind the falsification test (docs/SYSTEM_MAP.md §6).

A candidate strategy's test battery is judged in two layers:
  1. falsification_gate(...)  — DETERMINISTIC hard rail with VETO power. Encodes the lessons paid for
     across 23 case studies: it must clear the out-of-universe (fresh-symbol) holdout, be regime-robust
     (out-of-regime), survive the cost wall, and clear the Deflated Sharpe bar. Code decides this, not a model.
  2. haiku_verdict(...)       — Haiku reads the SAME results + the current market regime and returns a
     structured GO / NO-GO with a one-line rationale and the biggest risk. It adds judgement and a per-
     regime read; it CANNOT override the deterministic veto.

Final decision (gate(...)) = falsification_pass AND haiku_go. The model can suggest, but only the data
validates — so an AI-driven research loop built on this is safe by construction.
"""

from __future__ import annotations

import json
import math
import re
import statistics
from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ---- 1. deterministic regime classifier (cheap, no AI) ----
def classify_regime(spy_bars: List[dict], *, lookback: int = 60, ppy: float = 252.0,
                    trend_thr: float = 0.04, vol_thr: float = 0.25) -> str:
    """bull / bear / chop / high_vol from SPY trend + realized vol over the trailing `lookback`."""
    cl = [float(b["close"]) for b in spy_bars if b.get("close")]
    if len(cl) < lookback + 1:
        return "unknown"
    window = cl[-lookback - 1:]
    ret = window[-1] / window[0] - 1.0
    rets = [window[i] / window[i - 1] - 1.0 for i in range(1, len(window)) if window[i - 1] > 0]
    vol = statistics.pstdev(rets) * math.sqrt(ppy) if len(rets) > 1 else 0.0
    if vol > vol_thr:
        return "high_vol"
    if ret > trend_thr:
        return "bull"
    if ret < -trend_thr:
        return "bear"
    return "chop"


# ---- 2. deterministic falsification gate (the hard rail) ----
@dataclass
class GateResult:
    passed: bool
    checks: Dict[str, bool]
    reasons: List[str] = field(default_factory=list)


def falsification_gate(result: Dict, *, min_fresh: float = 0.2, min_regime_frac: float = 0.6,
                       min_cost_sharpe: float = 0.2, min_dsr: float = 0.9) -> GateResult:
    """The bar a candidate must clear, encoding the session's hard-won lessons. `result` keys:
      fresh_holdout_sharpe (out-of-universe), per_year {yr: sharpe} (out-of-regime),
      cost_2bps_sharpe (cost wall), dsr (deflated significance)."""
    py = result.get("per_year", {}) or {}
    n = len(py)
    pos_frac = (sum(1 for s in py.values() if s > 0) / n) if n else 0.0
    checks = {
        "fresh_symbol_holdout": result.get("fresh_holdout_sharpe", -9) > min_fresh,
        "regime_robust": n >= 3 and pos_frac >= min_regime_frac,
        "survives_cost": result.get("cost_2bps_sharpe", -9) > min_cost_sharpe,
        "deflated_significant": result.get("dsr", 0.0) >= min_dsr,
    }
    reasons = []
    if not checks["fresh_symbol_holdout"]:
        reasons.append(f"fails fresh-symbol holdout ({result.get('fresh_holdout_sharpe')} <= {min_fresh}) "
                       "— does not generalize to unseen symbols")
    if not checks["regime_robust"]:
        reasons.append(f"not regime-robust (positive {pos_frac:.0%} of {n} years < {min_regime_frac:.0%})")
    if not checks["survives_cost"]:
        reasons.append(f"dies to the cost wall (2bps Sharpe {result.get('cost_2bps_sharpe')} <= {min_cost_sharpe})")
    if not checks["deflated_significant"]:
        reasons.append(f"below the DSR bar ({result.get('dsr')} < {min_dsr})")
    return GateResult(passed=all(checks.values()), checks=checks, reasons=reasons)


# ---- 3. Haiku verdict (judgement + per-regime read; no veto over the rail) ----
_GATE_SYSTEM = (
    "You are a skeptical quant reviewer for a market-neutral research loop. You receive a candidate "
    "strategy's out-of-sample test results and the current market regime. Reply ONLY with compact JSON: "
    '{"verdict":"GO"|"NO-GO","confidence":0..1,"rationale":"<=20 words","biggest_risk":"<=12 words",'
    '"regime_fit":"<=12 words"}. Be harsh: an edge must generalize to UNSEEN symbols and survive '
    "realistic costs across regimes; in-sample shine is worthless.")


def _haiku_prompt(result: Dict, regime: str, gate: GateResult) -> str:
    return (f"Current regime: {regime}\n"
            f"Deterministic falsification gate: {'PASS' if gate.passed else 'FAIL'} "
            f"({'; '.join(gate.reasons) if gate.reasons else 'all checks passed'})\n"
            f"Results: {json.dumps(result, default=str)}\n"
            "Give your JSON verdict.")


def haiku_verdict(result: Dict, regime: str, router, gate: Optional[GateResult] = None) -> Dict:
    gate = gate or falsification_gate(result)
    raw = router.small(_haiku_prompt(result, regime, gate), system=_GATE_SYSTEM, max_tokens=300)
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    try:
        v = json.loads(m.group(0)) if m else {}
    except ValueError:
        v = {}
    v.setdefault("verdict", "NO-GO")          # fail-closed if the model reply is unparseable
    v.setdefault("rationale", raw[:120])
    return v


# ---- 4. combined gate: deterministic rail AND Haiku must both say GO ----
def gate(result: Dict, spy_bars: List[dict], router, *,
         candidate_returns: Optional[Dict[int, float]] = None,
         book_returns: Optional[Dict[int, float]] = None) -> Dict:
    """Two-gate decision: (1) falsification_gate — is it a REAL edge? (2) Haiku concurs. When
    `candidate_returns` + `book_returns` are supplied, a THIRD gate runs — the second-leg gate
    (`leg_gate.evaluate_leg_candidate`): does it DIVERSIFY the deployed book (positive, uncorrelated,
    lifts robustly)? An edge that is real but dilutes the book is NO-GO as a leg (momentum, Case 47)."""
    regime = classify_regime(spy_bars)
    fg = falsification_gate(result)
    hv = haiku_verdict(result, regime, router, gate=fg)
    haiku_go = str(hv.get("verdict", "")).upper() == "GO"

    leg = None
    leg_ok = True
    if candidate_returns is not None and book_returns is not None:
        from alpca.backtest.leg_gate import evaluate_leg_candidate
        lv = evaluate_leg_candidate(candidate_returns, book_returns)
        leg = {"passed": lv.passed, "checks": lv.checks, "reasons": lv.reasons,
               "rho": lv.rho, "lift": lv.lift, "combined_sharpe": lv.combined_sharpe}
        leg_ok = lv.passed

    return {
        "regime": regime,
        "falsification_pass": fg.passed,
        "falsification_checks": fg.checks,
        "falsification_reasons": fg.reasons,
        "haiku": hv,
        "leg_gate": leg,
        # GO only if the data clears the bar AND Haiku concurs AND (if requested) it diversifies the book.
        "decision": "GO" if (fg.passed and haiku_go and leg_ok) else "NO-GO",
    }
