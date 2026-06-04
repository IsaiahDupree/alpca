"""
Cash dividends — schedule + cash-flow crediting.

IMPORTANT adjustment policy (avoids double-counting):
  - With **dividend-adjusted** bars (`adjustment="all"` or `"dividend"`), the
    historical prices are back-adjusted so the ex-date price drop is removed and
    total return is already captured in price continuity. DO NOT also credit
    dividend cash — that double-counts.
  - With **raw** (or split-only) bars, the price DROPS by ~the dividend on the
    ex-date, so a holder's mark-to-market falls; crediting the dividend cash
    restores the correct total return. This is the live-trading reality (Alpaca
    pays the cash into the account; quotes are raw).

So: use a DividendSchedule with raw/split bars, NOT with dividend-adjusted bars.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Optional

from alpca.config import AlpacaConfig


@dataclass
class Dividend:
    ex_date_ts: float   # epoch seconds of the ex-dividend date (00:00 UTC of that day)
    amount: float       # cash dividend per share


class DividendSchedule:
    """Ordered set of cash dividends; answers 'how much per share between t0 and t1'."""

    def __init__(self, dividends: Optional[List[Dividend]] = None) -> None:
        self._divs = sorted(dividends or [], key=lambda d: d.ex_date_ts)

    def __len__(self) -> int:
        return len(self._divs)

    @property
    def dividends(self) -> List[Dividend]:
        return list(self._divs)

    def per_share_between(self, prev_ts: float, cur_ts: float) -> float:
        """Sum of per-share dividends with ex-date in the half-open (prev_ts, cur_ts].

        Half-open on the low side so crossing an ex-date credits exactly once as
        the backtest advances bar to bar, regardless of bar granularity.
        """
        total = 0.0
        for d in self._divs:
            if prev_ts < d.ex_date_ts <= cur_ts:
                total += d.amount
        return total

    @classmethod
    def from_pairs(cls, pairs) -> "DividendSchedule":
        """Build from (ex_date_epoch_seconds, amount_per_share) tuples."""
        return cls([Dividend(float(ts), float(amt)) for ts, amt in pairs])


def _to_epoch(value) -> float:
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, datetime):
        return value.timestamp()
    s = str(value).strip()
    # corporate-action ex_date is typically a YYYY-MM-DD date string
    for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%SZ"):
        try:
            dt = datetime.strptime(s, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.timestamp()
        except ValueError:
            continue
    return 0.0


def fetch_alpaca_dividends(
    config: AlpacaConfig,
    symbol: str,
    *,
    start: datetime,
    end: datetime,
) -> DividendSchedule:
    """
    Fetch cash dividends for `symbol` from Alpaca's corporate-actions endpoint.

    Returns a DividendSchedule keyed by ex-date. Use this together with RAW bars
    (`fetch_alpaca_bars(adjustment="raw")`) so the price drop on ex-date is offset
    by the credited cash. Requires credentials.
    """
    config.require_credentials()
    from alpaca.data.historical.corporate_actions import CorporateActionsClient
    from alpaca.data.requests import CorporateActionsRequest
    from alpaca.data.enums import CorporateActionsType

    client = CorporateActionsClient(config.api_key, config.secret_key)
    req = CorporateActionsRequest(
        symbols=[symbol],
        types=[CorporateActionsType.CASH_DIVIDEND],
        start=start.date() if isinstance(start, datetime) else start,
        end=end.date() if isinstance(end, datetime) else end,
    )
    resp = client.get_corporate_actions(req)

    # The response groups actions by type; cash dividends carry ex_date + rate.
    raw = getattr(resp, "data", resp)
    items = []
    if isinstance(raw, dict):
        for key in ("cash_dividends", "cash_dividend", "CASH_DIVIDEND"):
            if key in raw:
                items = raw[key]
                break
    divs: List[Dividend] = []
    for it in items or []:
        ex = getattr(it, "ex_date", None) or (it.get("ex_date") if isinstance(it, dict) else None)
        rate = getattr(it, "rate", None)
        if rate is None and isinstance(it, dict):
            rate = it.get("rate")
        sym = getattr(it, "symbol", None) or (it.get("symbol") if isinstance(it, dict) else symbol)
        if ex is not None and rate is not None and (sym is None or sym == symbol):
            divs.append(Dividend(_to_epoch(ex), float(rate)))
    return DividendSchedule(divs)
