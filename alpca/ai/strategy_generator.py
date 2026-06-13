"""
Strategy generator for the AI research loop — SAFE by construction.

The model never writes or runs code. It only SELECTS and CONFIGURES a strategy from a constrained
space of KNOWN, already-tested backtest templates (param ranges clamped). The deterministic harness
then runs that config and the falsification gate judges it. So the worst the model can do is propose
a config that the harness rejects — it cannot execute arbitrary code or bypass the data.

  propose(regime, router)   -> a validated config {strategy_type, params, rationale, source}
  run_proposal(config, ...)  -> a gate-ready result dict (sharpe, oos, per_year, cost_2bps,
                                fresh_holdout via a disjoint universe, dsr, turnover)
"""

from __future__ import annotations

import json
import re
import time
from typing import Dict, List, Optional

from alpca.backtest.cross_sectional import backtest_cross_sectional_momentum
from alpca.backtest.evaluation import deflated_sharpe_ratio, sharpe_of

PPY = 252.0
PARAM_BOUNDS = {"lookback": (20, 250), "hold": (5, 60), "top_k": (3, 15)}


def _run_xsec(bars, p, cost_bps, reverse):
    return backtest_cross_sectional_momentum(
        bars, lookback=int(p["lookback"]), hold=int(p["hold"]), top_k=int(p["top_k"]),
        bottom_k=int(p["top_k"]), cost_bps=cost_bps, periods_per_year=PPY,
        market_neutral=True, reverse=reverse)


STRATEGY_SPACE = {
    "xsec_momentum": {"reverse": False, "desc": "long recent winners / short losers (cross-sectional momentum)"},
    "xsec_reversal": {"reverse": True, "desc": "long recent losers / short winners (short-horizon reversal)"},
}

# a-priori regime -> template mapping (the deterministic fallback / cold-start)
_HEURISTIC = {
    "bull":     ("xsec_momentum", {"lookback": 120, "hold": 20, "top_k": 10}),
    "bear":     ("xsec_reversal", {"lookback": 20, "hold": 5, "top_k": 10}),
    "chop":     ("xsec_reversal", {"lookback": 60, "hold": 10, "top_k": 10}),
    "high_vol": ("xsec_momentum", {"lookback": 250, "hold": 60, "top_k": 5}),
    "unknown":  ("xsec_momentum", {"lookback": 120, "hold": 20, "top_k": 10}),
}


def _clamp(params: Dict) -> Dict:
    out = {}
    for k, (lo, hi) in PARAM_BOUNDS.items():
        v = params.get(k, lo)
        try:
            out[k] = int(min(hi, max(lo, float(v))))
        except (TypeError, ValueError):
            out[k] = lo
    return out


def heuristic_proposal(regime: str) -> Dict:
    st, p = _HEURISTIC.get(regime, _HEURISTIC["unknown"])
    return {"strategy_type": st, "params": dict(p), "rationale": f"a-priori {regime} template",
            "source": "heuristic"}


def ai_proposal(regime_prompt: str, router, tried: Optional[List[str]] = None) -> Optional[Dict]:
    """Ask the medium model to pick + configure a template from the constrained space. Returns a
    validated config, or None if the reply can't be validated (caller falls back to heuristic)."""
    space = "\n".join(f'  - "{k}": {v["desc"]}' for k, v in STRATEGY_SPACE.items())
    bounds = ", ".join(f"{k} in [{lo},{hi}]" for k, (lo, hi) in PARAM_BOUNDS.items())
    sys = ("You are a quant proposing ONE market-neutral cross-sectional strategy config from a fixed "
           "menu. You may ONLY choose a listed strategy_type and integer params within bounds. Reply "
           "ONLY with JSON: {\"strategy_type\":\"...\",\"params\":{\"lookback\":N,\"hold\":N,\"top_k\":N},"
           "\"rationale\":\"<=18 words\"}.")
    prompt = (f"Market {regime_prompt}.\nMenu:\n{space}\nParam bounds: {bounds}.\n"
              f"Already tried this run: {tried or 'none'}.\n"
              "Pick the config most likely to be a REAL market-neutral edge in this regime "
              "(must generalize to unseen symbols and survive costs). JSON only.")
    raw = router.think(prompt, system=sys, max_tokens=200)
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except ValueError:
        return None
    st = d.get("strategy_type")
    if st not in STRATEGY_SPACE:
        return None
    return {"strategy_type": st, "params": _clamp(d.get("params", {})),
            "rationale": str(d.get("rationale", ""))[:120], "source": "ai"}


def propose(regime_prompt: str, regime_label: str, router=None, tried: Optional[List[str]] = None) -> Dict:
    """AI proposal if a router is available, else the deterministic heuristic."""
    if router is not None and router.available().get("openai"):
        try:
            ai = ai_proposal(regime_prompt, router, tried)
            if ai:
                return ai
        except Exception:
            pass
    return heuristic_proposal(regime_label)


def _common_dates(bars_by_sym) -> List[int]:
    sets = [set(int(b["timestamp"]) for b in v) for v in bars_by_sym.values() if v]
    return sorted(set.intersection(*sets)) if sets else []


def _per_year(equity: List[float], dates: List[int]) -> Dict[int, float]:
    # equity has len == len(dates); returns sit between consecutive points (aligned with dates[1:])
    by: Dict[int, List[float]] = {}
    for i in range(1, min(len(equity), len(dates))):
        if equity[i - 1] > 0:
            by.setdefault(time.gmtime(dates[i]).tm_year, []).append(equity[i] / equity[i - 1] - 1.0)
    out = {}
    for y, r in by.items():
        if len(r) >= 30:
            eq = [1.0]
            for x in r:
                eq.append(eq[-1] * (1 + x))
            out[y] = round(sharpe_of(eq, PPY), 2)
    return out


def run_proposal(config: Dict, bars_main: Dict, bars_fresh: Dict, *, cost_bps: float = 2.0,
                 n_trials: int = 40) -> Dict:
    """Run a proposed config through the harness on the MAIN universe + a DISJOINT fresh universe,
    returning the result dict the falsification gate consumes."""
    rev = STRATEGY_SPACE[config["strategy_type"]]["reverse"]
    p = config["params"]
    main = _run_xsec(bars_main, p, cost_bps, rev)
    eq = main.equity_curve
    dates = _common_dates(bars_main)
    sp = int(len(eq) * 0.7)
    is_sh = sharpe_of(eq[:sp], PPY) if sp > 5 else 0.0
    oos_sh = sharpe_of(eq[sp:], PPY) if len(eq) - sp > 5 else 0.0
    fresh = _run_xsec(bars_fresh, p, cost_bps, rev) if bars_fresh else None
    return {
        "strategy_type": config["strategy_type"], "params": p, "source": config.get("source"),
        "rationale": config.get("rationale"),
        "sharpe": round(main.sharpe, 3), "is_sharpe": round(is_sh, 3), "oos_sharpe": round(oos_sh, 3),
        "cost_2bps_sharpe": round(main.sharpe, 3),          # the run is already at cost_bps=2.0
        "fresh_holdout_sharpe": round(fresh.sharpe, 3) if fresh else None,
        "per_year": _per_year(eq, dates),
        "dsr": round(deflated_sharpe_ratio(eq, n_trials=n_trials, sharpe_variance=1e-4), 3),
        "max_drawdown": round(main.max_drawdown, 3), "n_rebalances": main.n_rebalances,
    }
