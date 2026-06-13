"""
Strategy generator for the AI research loop — SAFE by construction.

The model never writes or runs code. It only SELECTS and CONFIGURES a strategy from a constrained
space of KNOWN, already-tested backtest templates (per-template param ranges clamped). The
deterministic harness then runs that config and the falsification gate judges it. So the worst the
model can do is propose a config the harness rejects — it cannot execute arbitrary code or bypass data.

Families:
  xsec        — price-only cross-sectional (momentum / reversal); needs bars only.
  fundamental — accruals (Sloan earnings-quality, EDGAR); needs bars + annual fundamentals.

  propose(...)      -> a validated config {strategy_type, params, rationale, source}
  run_proposal(...) -> a gate-ready result dict (sharpe, oos, per_year, cost_2bps, fresh_holdout, dsr)
"""

from __future__ import annotations

import json
import re
import time
from typing import Dict, List, Optional

from alpca.backtest.accruals import backtest_accruals
from alpca.backtest.cross_sectional import backtest_cross_sectional_momentum
from alpca.backtest.evaluation import deflated_sharpe_ratio, sharpe_of
from alpca.backtest.value import backtest_value_composite

PPY = 252.0

# Each template: family, its own integer param bounds, and a one-line description for the model.
STRATEGY_SPACE: Dict[str, Dict] = {
    "xsec_momentum": {"family": "xsec", "reverse": False,
                      "params": {"lookback": (20, 250), "hold": (5, 60), "top_k": (3, 15)},
                      "desc": "long recent winners / short losers (cross-sectional momentum)"},
    "xsec_reversal": {"family": "xsec", "reverse": True,
                      "params": {"lookback": (20, 250), "hold": (5, 60), "top_k": (3, 15)},
                      "desc": "long recent losers / short winners (short-horizon reversal)"},
    "accruals":      {"family": "fundamental",
                      "params": {"top_frac_pct": (10, 33)},
                      "desc": "long low-accrual / short high-accrual (Sloan earnings-quality; "
                              "fundamental, annual rebalance, ~free turnover, diversifies price edges)"},
    "value_composite": {"family": "fundamental",
                        "params": {"top_frac_pct": (10, 33), "rebalance_days": (10, 63)},
                        "desc": "long cheap / short expensive on an E/P + FCF/P + B/P composite "
                                "(the value premium; fundamental, low turnover, diversifies momentum)"},
}

# a-priori regime -> template (the deterministic fallback / cold-start)
_HEURISTIC = {
    "bull":     ("xsec_momentum", {"lookback": 120, "hold": 20, "top_k": 10}),
    "bear":     ("xsec_reversal", {"lookback": 20, "hold": 5, "top_k": 10}),
    "chop":     ("xsec_reversal", {"lookback": 60, "hold": 10, "top_k": 10}),
    "high_vol": ("accruals",      {"top_frac_pct": 10}),     # de-risk to a fundamental, low-turnover edge
    "unknown":  ("xsec_momentum", {"lookback": 120, "hold": 20, "top_k": 10}),
}


def _clamp(strategy_type: str, params: Dict) -> Dict:
    bounds = STRATEGY_SPACE[strategy_type]["params"]
    out = {}
    for k, (lo, hi) in bounds.items():
        try:
            out[k] = int(min(hi, max(lo, float(params.get(k, lo)))))
        except (TypeError, ValueError):
            out[k] = lo
    return out


def heuristic_proposal(regime: str) -> Dict:
    st, p = _HEURISTIC.get(regime, _HEURISTIC["unknown"])
    return {"strategy_type": st, "params": dict(p), "rationale": f"a-priori {regime} template",
            "source": "heuristic"}


def _menu() -> str:
    lines = []
    for k, v in STRATEGY_SPACE.items():
        ps = ", ".join(f"{pk} in [{lo},{hi}]" for pk, (lo, hi) in v["params"].items())
        lines.append(f'  - "{k}" ({v["family"]}): {v["desc"]} | params: {ps}')
    return "\n".join(lines)


def ai_proposal(regime_prompt: str, router, tried: Optional[List[str]] = None) -> Optional[Dict]:
    sys = ("You are a quant proposing ONE market-neutral strategy config from a fixed menu. You may "
           "ONLY choose a listed strategy_type and integer params within its bounds. Prefer edges that "
           "GENERALIZE to unseen symbols and survive costs; fundamental/low-turnover edges diversify a "
           "price-heavy book. Reply ONLY with JSON: "
           '{"strategy_type":"...","params":{...},"rationale":"<=18 words"}.')
    prompt = (f"Market {regime_prompt}.\nMenu:\n{_menu()}\n"
              f"Already tried this run: {tried or 'none'}.\nPick the config most likely to be a REAL "
              "market-neutral edge in this regime. JSON only.")
    raw = router.think(prompt, system=sys, max_tokens=220)
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
    return {"strategy_type": st, "params": _clamp(st, d.get("params", {})),
            "rationale": str(d.get("rationale", ""))[:120], "source": "ai"}


def propose(regime_prompt: str, regime_label: str, router=None, tried: Optional[List[str]] = None) -> Dict:
    if router is not None and router.available().get("openai"):
        try:
            ai = ai_proposal(regime_prompt, router, tried)
            if ai:
                return ai
        except Exception:
            pass
    return heuristic_proposal(regime_label)


# ---- harness runners ----
def _run(config: Dict, bars: Dict, funds: Optional[Dict], cost_bps: float):
    spec = STRATEGY_SPACE[config["strategy_type"]]
    p = config["params"]
    if spec["family"] == "xsec":
        return backtest_cross_sectional_momentum(
            bars, lookback=p["lookback"], hold=p["hold"], top_k=p["top_k"], bottom_k=p["top_k"],
            cost_bps=cost_bps, periods_per_year=PPY, market_neutral=True, reverse=spec["reverse"])
    if spec["family"] == "fundamental":
        if not funds:
            return None
        if config["strategy_type"] == "value_composite":
            return backtest_value_composite(bars, funds, top_frac=p["top_frac_pct"] / 100.0,
                                            rebalance_days=int(p.get("rebalance_days", 21)),
                                            reverse=False, cost_bps=cost_bps, periods_per_year=PPY)
        return backtest_accruals(bars, funds, top_frac=p["top_frac_pct"] / 100.0,
                                 reverse=False, cost_bps=cost_bps, periods_per_year=PPY)
    return None


def _common_dates(bars_by_sym) -> List[int]:
    sets = [set(int(b["timestamp"]) for b in v) for v in bars_by_sym.values() if v]
    return sorted(set.intersection(*sets)) if sets else []


def _per_year(res, bars) -> Dict[int, float]:
    eq = res.equity_curve
    dates = getattr(res, "dates", None) or _common_dates(bars)
    by: Dict[int, List[float]] = {}
    for i in range(1, min(len(eq), len(dates) + 1)):
        if eq[i - 1] > 0:
            by.setdefault(time.gmtime(dates[i - 1]).tm_year, []).append(eq[i] / eq[i - 1] - 1.0)
    out = {}
    for y, r in by.items():
        if len(r) >= 30:
            e = [1.0]
            for x in r:
                e.append(e[-1] * (1 + x))
            out[y] = round(sharpe_of(e, PPY), 2)
    return out


def run_proposal(config: Dict, bars_main: Dict, bars_fresh: Dict, *, fund_main: Optional[Dict] = None,
                 fund_fresh: Optional[Dict] = None, cost_bps: float = 2.0, n_trials: int = 40) -> Dict:
    """Run a proposed config through the harness on the MAIN universe + a DISJOINT fresh universe,
    returning the result dict the falsification gate consumes. Fundamental templates use the funds."""
    main = _run(config, bars_main, fund_main, cost_bps)
    if main is None:
        return {"strategy_type": config["strategy_type"], "params": config["params"], "error": "no_data",
                "sharpe": 0.0, "oos_sharpe": 0.0, "cost_2bps_sharpe": 0.0, "fresh_holdout_sharpe": None,
                "per_year": {}, "dsr": 0.0}
    eq = main.equity_curve
    sp = int(len(eq) * 0.7)
    is_sh = sharpe_of(eq[:sp], PPY) if sp > 5 else 0.0
    oos_sh = sharpe_of(eq[sp:], PPY) if len(eq) - sp > 5 else 0.0
    fresh = _run(config, bars_fresh, fund_fresh, cost_bps) if bars_fresh else None
    return {
        "strategy_type": config["strategy_type"], "params": config["params"],
        "source": config.get("source"), "rationale": config.get("rationale"),
        "sharpe": round(main.sharpe, 3), "is_sharpe": round(is_sh, 3), "oos_sharpe": round(oos_sh, 3),
        "cost_2bps_sharpe": round(main.sharpe, 3),
        "fresh_holdout_sharpe": round(fresh.sharpe, 3) if fresh else None,
        "per_year": _per_year(main, bars_main),
        "dsr": round(deflated_sharpe_ratio(eq, n_trials=n_trials, sharpe_variance=1e-4), 3),
        "max_drawdown": round(main.max_drawdown, 3),
    }
