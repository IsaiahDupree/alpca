"""
Fit the offline fill model + latency preset to REAL paper fills.

Given a list of CalibrationRecords (real fills), estimate:
  - half_spread_bps : the spread component you pay on every marketable fill —
                      the median realized slippage at the SMALLEST trade size
                      (where market impact is negligible).
  - impact_coef_bps : the square-root impact coefficient, fit by least squares on
                      slippage = half_spread + coef * sqrt(participation) — but
                      ONLY if the records span a range of participation; with
                      uniform tiny size it is left at the prior (and flagged).
  - latency preset  : submit->ack and ack->fill percentiles for SimAdapter.

The output is a CalibrationResult you can turn into a FillModel
(`to_fill_model()`); its measured SimAdapter latency preset is exposed as the
`.latency` attribute (a LatencyPreset). It also writes a JSON artifact (`save()`)
the runner/backtester can load.

Pure + dependency-free (no numpy) so it runs anywhere and is fully unit-tested.
"""

from __future__ import annotations

import json
import math
import os
import statistics
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

from alpca.calibration.records import CalibrationRecord


def _percentile(sorted_vals: List[float], q: float) -> Optional[float]:
    if not sorted_vals:
        return None
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    idx = q * (len(sorted_vals) - 1)
    lo, hi = math.floor(idx), math.ceil(idx)
    if lo == hi:
        return sorted_vals[int(idx)]
    frac = idx - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


@dataclass
class LatencyPreset:
    submit_latency_ms: float
    ack_latency_ms: float
    fill_latency_ms: float
    jitter_ms: float

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class CalibrationResult:
    n_records: int
    n_buys: int
    n_sells: int
    half_spread_bps: float
    impact_coef_bps: float
    impact_fitted: bool          # True only if participation varied enough to fit
    slippage_p50_bps: Optional[float]
    slippage_p95_bps: Optional[float]
    latency: Optional[LatencyPreset]
    notes: List[str] = field(default_factory=list)
    # --- Phase 1 extras (all optional; legacy fields above are unchanged) ---
    half_spread_measured: bool = False     # half_spread came from real quotes, not the bottom-quartile proxy
    eta: Optional[float] = None            # Almgren impact: slippage ≈ c + eta·σ·participation^beta
    beta: Optional[float] = None           # participation exponent (sqrt law => 0.5)
    beta_fitted: bool = False
    impact_coef_low_vol: Optional[float] = None   # sqrt-impact coef in the low-σ regime
    impact_coef_high_vol: Optional[float] = None  # ...and the high-σ regime
    vol_threshold: Optional[float] = None         # σ split between the two regimes

    def to_fill_model(self, *, participation_cap: float = 0.10, min_tick: float = 0.01):
        from alpca.execution.fills import FillModel
        return FillModel(half_spread_bps=max(0.0, self.half_spread_bps),
                         impact_coef_bps=max(0.0, self.impact_coef_bps),
                         participation_cap=participation_cap, min_tick=min_tick)

    def to_dict(self) -> Dict:
        d = asdict(self)
        if self.latency is not None:
            d["latency"] = self.latency.to_dict()
        return d

    def save(self, path: str = "data/calibration.json") -> str:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2, default=str)
        return path


def _ols_two_param(xs: List[float], ys: List[float]) -> Optional[tuple]:
    """Least-squares fit y = a + b*x. Returns (a, b) or None if degenerate."""
    n = len(xs)
    if n < 2:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 1e-12:
        return None
    sxy = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
    b = sxy / sxx
    a = my - b * mx
    return a, b


def _nelder_mead(f, x0: List[float], *, step: float = 0.25,
                 max_iter: int = 600, tol: float = 1e-10):
    """Minimal dependency-free Nelder-Mead simplex minimizer (scipy is not
    installed). Returns (best_x, best_f). Bounds are handled by the objective
    (clamp + penalty), keeping this routine generic."""
    n = len(x0)
    simplex = [list(x0)]
    for i in range(n):
        x = list(x0)
        x[i] = x[i] + step if x[i] == 0 else x[i] * (1 + step)
        simplex.append(x)
    fv = [f(x) for x in simplex]
    for _ in range(max_iter):
        order = sorted(range(n + 1), key=lambda k: fv[k])
        simplex = [simplex[k] for k in order]
        fv = [fv[k] for k in order]
        if abs(fv[-1] - fv[0]) <= tol * (abs(fv[0]) + abs(fv[-1]) + 1e-12):
            break
        cen = [sum(simplex[k][j] for k in range(n)) / n for j in range(n)]
        worst = simplex[-1]
        xr = [cen[j] + (cen[j] - worst[j]) for j in range(n)]
        fr = f(xr)
        if fv[0] <= fr < fv[-2]:
            simplex[-1], fv[-1] = xr, fr
        elif fr < fv[0]:
            xe = [cen[j] + 2.0 * (cen[j] - worst[j]) for j in range(n)]
            fe = f(xe)
            simplex[-1], fv[-1] = (xe, fe) if fe < fr else (xr, fr)
        else:
            xc = [cen[j] + 0.5 * (worst[j] - cen[j]) for j in range(n)]
            fc = f(xc)
            if fc < fv[-1]:
                simplex[-1], fv[-1] = xc, fc
            else:
                best = simplex[0]
                for k in range(1, n + 1):
                    simplex[k] = [best[j] + 0.5 * (simplex[k][j] - best[j]) for j in range(n)]
                    fv[k] = f(simplex[k])
    bi = min(range(n + 1), key=lambda k: fv[k])
    return simplex[bi], fv[bi]


def fit_nonlinear_impact(records: List[CalibrationRecord], *,
                         half_spread_hint: float = 0.0,
                         min_points: int = 4,
                         participation_spread_threshold: float = 1.5) -> Optional[Dict]:
    """
    Fit the Almgren-style impact curve  slippage ≈ c + η·σ·participation^β  by
    minimizing SSE with Nelder-Mead (β bounded to [0.2, 1.0]; c, η ≥ 0). σ is each
    record's realized_vol (falls back to 1.0, in which case η absorbs the vol
    scale). Returns {c, eta, beta} or None when there are too few points / too
    narrow a participation range to bend the curve (the exact case the legacy
    single-slope OLS handles poorly). Reimplemented clean from public Almgren math.
    """
    pts = [(r.participation, r.slippage_bps, (r.realized_vol or 1.0))
           for r in records
           if r.participation and r.participation > 0 and r.slippage_bps is not None]
    if len(pts) < min_points:
        return None
    parts = [p for p, _, _ in pts]
    if min(parts) <= 0 or (max(parts) / min(parts)) < participation_spread_threshold:
        return None

    def obj(params):
        c, eta, beta = params
        pen = 0.0
        bc = min(1.0, max(0.2, beta))
        cc = max(0.0, c)
        ec = max(0.0, eta)
        pen += (beta - bc) ** 2 * 1e6 + (min(0.0, c)) ** 2 * 1e6 + (min(0.0, eta)) ** 2 * 1e6
        sse = 0.0
        for x, y, sig in pts:
            pred = cc + ec * sig * (x ** bc)
            sse += (y - pred) ** 2
        return sse + pen

    ys = [y for _, y, _ in pts]
    resid = max(0.0, statistics.median(ys) - half_spread_hint)
    xref = statistics.median([(x ** 0.5) * s for x, _, s in pts]) or 1.0
    eta0 = resid / xref if xref > 0 else 1.0
    best, _ = _nelder_mead(obj, [max(0.0, half_spread_hint), max(1e-6, eta0), 0.5])
    c, eta, beta = best
    return {"c": max(0.0, c), "eta": max(0.0, eta), "beta": min(1.0, max(0.2, beta))}


def _regime_split_impact(records: List[CalibrationRecord], *,
                         participation_spread_threshold: float = 1.5,
                         min_per_regime: int = 4) -> Optional[tuple]:
    """Stratify records by realized_vol (median split) and fit the legacy sqrt-
    impact coef per regime. Returns (coef_low, coef_high, vol_threshold) — either
    coef may be None if its regime is too small/narrow. None if not splittable."""
    rs = [r for r in records
          if r.realized_vol is not None and r.participation and r.participation > 0
          and r.slippage_bps is not None]
    if len(rs) < 2 * min_per_regime:
        return None
    thr = _percentile(sorted(r.realized_vol for r in rs), 0.50)
    if thr is None:
        return None

    def coef(group):
        if len(group) < min_per_regime:
            return None
        parts = [r.participation for r in group]
        if min(parts) <= 0 or (max(parts) / min(parts)) < participation_spread_threshold:
            return None
        fit = _ols_two_param([math.sqrt(r.participation) for r in group],
                             [r.slippage_bps for r in group])
        return max(0.0, fit[1]) if fit else None

    low = coef([r for r in rs if r.realized_vol <= thr])
    high = coef([r for r in rs if r.realized_vol > thr])
    if low is None and high is None:
        return None
    return low, high, thr


def calibrate(records: List[CalibrationRecord], *,
              min_records: int = 6,
              participation_spread_threshold: float = 1.5) -> CalibrationResult:
    """
    Fit a CalibrationResult from real fills.

    `min_records`: below this, the fit is reported but flagged low-confidence.
    `participation_spread_threshold`: impact is only fit when max/min participation
    exceeds this ratio (otherwise sqrt(participation) is ~constant and the slope
    is meaningless).
    """
    notes: List[str] = []
    valid = [r for r in records if r.slippage_bps is not None]
    if not valid:
        return CalibrationResult(0, 0, 0, half_spread_bps=1.0, impact_coef_bps=8.0,
                                 impact_fitted=False, slippage_p50_bps=None,
                                 slippage_p95_bps=None, latency=None,
                                 notes=["no valid records; returning priors"])

    n = len(valid)
    n_buys = sum(1 for r in valid if r.side == "BUY")
    n_sells = n - n_buys
    if n < min_records:
        notes.append(f"only {n} records (< {min_records}); fit is low-confidence")

    slips = sorted(r.slippage_bps for r in valid)
    p50 = _percentile(slips, 0.50)
    p95 = _percentile(slips, 0.95)

    # half-spread: use the records with the smallest participation (impact ~0).
    with_part = [r for r in valid if r.participation is not None]
    impact_coef = 8.0
    impact_fitted = False
    if with_part:
        parts = [r.participation for r in with_part]
        pmin, pmax = min(parts), max(parts)
        # half-spread from the bottom-quartile-participation fills
        cut = _percentile(sorted(parts), 0.25) or pmin
        small = [r.slippage_bps for r in with_part if r.participation <= cut + 1e-12]
        half_spread = statistics.median(small) if small else statistics.median(slips)
        # impact: fit slippage = half_spread + coef*sqrt(participation)
        if pmin > 0 and (pmax / pmin) >= participation_spread_threshold and len(with_part) >= 4:
            xs = [math.sqrt(r.participation) for r in with_part]
            ys = [r.slippage_bps for r in with_part]
            fit = _ols_two_param(xs, ys)
            if fit is not None:
                a, b = fit
                impact_coef = max(0.0, b)
                impact_fitted = True
                notes.append(f"impact fitted via sqrt-participation OLS (intercept {a:.2f} bps)")
        else:
            notes.append("participation range too narrow to fit impact; kept prior coef")
    else:
        half_spread = statistics.median(slips)
        notes.append("no volume context on records; impact not fittable, kept prior coef")

    # Spread decomposition (tcapy): if records carry a real NBBO, MEASURE the
    # half-spread directly rather than inferring it from bottom-quartile fills.
    half_spread_measured = False
    quoted = [r for r in valid if r.quoted_half_spread_bps is not None]
    if len(quoted) >= 4:
        half_spread = statistics.median(r.quoted_half_spread_bps for r in quoted)
        half_spread_measured = True
        notes.append(f"half-spread measured from {len(quoted)} real quotes "
                     f"({half_spread:.2f} bps)")

    half_spread = max(0.0, half_spread)

    # Nonlinear (eta, beta) Almgren fit + vol-regime split — Phase 1 extras. These
    # are ADDITIVE outputs; half_spread_bps/impact_coef_bps/impact_fitted above keep
    # their legacy meaning so existing consumers are unchanged.
    eta = beta = None
    beta_fitted = False
    nl = fit_nonlinear_impact(valid, half_spread_hint=half_spread)
    if nl is not None:
        eta, beta = nl["eta"], nl["beta"]
        beta_fitted = True
        notes.append(f"nonlinear impact fit: eta={eta:.3f} beta={beta:.3f}")

    impact_low = impact_high = vol_thr = None
    rs = _regime_split_impact(valid)
    if rs is not None:
        impact_low, impact_high, vol_thr = rs
        notes.append(f"vol-regime impact split @ sigma={vol_thr:.3f}: "
                     f"low={impact_low} high={impact_high}")

    # latency preset from measured submit->ack / ack->fill
    lat = _fit_latency(valid)
    if lat is None:
        notes.append("no latency captured on records; latency preset unavailable")

    return CalibrationResult(
        n_records=n, n_buys=n_buys, n_sells=n_sells,
        half_spread_bps=round(half_spread, 3),
        impact_coef_bps=round(impact_coef, 3),
        impact_fitted=impact_fitted,
        slippage_p50_bps=None if p50 is None else round(p50, 3),
        slippage_p95_bps=None if p95 is None else round(p95, 3),
        latency=lat, notes=notes,
        half_spread_measured=half_spread_measured,
        eta=None if eta is None else round(eta, 4),
        beta=None if beta is None else round(beta, 4),
        beta_fitted=beta_fitted,
        impact_coef_low_vol=None if impact_low is None else round(impact_low, 3),
        impact_coef_high_vol=None if impact_high is None else round(impact_high, 3),
        vol_threshold=None if vol_thr is None else round(vol_thr, 4),
    )


def _fit_latency(records: List[CalibrationRecord]) -> Optional[LatencyPreset]:
    sub_ack = sorted(r.submit_to_ack_ms for r in records if r.submit_to_ack_ms is not None)
    ack_fill = sorted(r.ack_to_fill_ms for r in records if r.ack_to_fill_ms is not None)
    if not sub_ack and not ack_fill:
        return None
    # split submit->ack into a nominal submit + ack leg (50/50) for the preset
    sa_p50 = _percentile(sub_ack, 0.50) or 0.0
    af_p50 = _percentile(ack_fill, 0.50) or 0.0
    sa_p95 = _percentile(sub_ack, 0.95) or sa_p50
    jitter = max(0.0, (sa_p95 - sa_p50))
    return LatencyPreset(
        submit_latency_ms=round(sa_p50 / 2.0, 1),
        ack_latency_ms=round(sa_p50 / 2.0, 1),
        fill_latency_ms=round(af_p50, 1),
        jitter_ms=round(jitter, 1),
    )
