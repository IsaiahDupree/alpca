"""
Calibration records — one row per REAL paper fill, persisted as JSONL.

Each record captures everything needed to fit the offline fill model to reality:
the price the strategy INTENDED (the mid at signal time), the price actually
FILLED, the side/size, the bar/quote volume context, and the measured lifecycle
latencies. The fitter (alpca.calibration.fit) turns a list of these into a
calibrated FillModel + SimAdapter latency preset.

Realized slippage is computed the same way as Order.slippage_bps: signed in bps
vs the intended price, positive = worse (paid more on a buy / received less on a
sell).
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from typing import List, Optional


@dataclass
class CalibrationRecord:
    symbol: str
    side: str                 # "BUY" | "SELL"
    qty: float
    intended_price: float     # mid (or marking price) at signal time
    fill_price: float         # realized average fill
    bar_volume: Optional[float] = None   # contemporaneous volume, for impact fit
    # measured lifecycle latencies (ms); any may be None if not captured
    signal_to_submit_ms: Optional[float] = None
    submit_to_ack_ms: Optional[float] = None
    ack_to_fill_ms: Optional[float] = None
    signal_to_fill_ms: Optional[float] = None
    ts: float = 0.0           # epoch seconds of the fill
    broker_order_id: Optional[str] = None
    # quote-at-signal + regime context (added for the Almgren impact fit + tcapy
    # spread decomposition). All Optional/None so old JSONL rows still load.
    realized_vol: Optional[float] = None   # annualized σ at signal time
    bid: Optional[float] = None
    ask: Optional[float] = None
    bid_size: Optional[float] = None
    ask_size: Optional[float] = None
    quote_ts: Optional[float] = None

    @property
    def slippage_bps(self) -> Optional[float]:
        """Signed adverse slippage vs intended price, in bps (positive = worse)."""
        if self.intended_price is None or self.fill_price is None or self.intended_price <= 0:
            return None
        diff = self.fill_price - self.intended_price
        if self.side == "SELL":
            diff = -diff
        return (diff / self.intended_price) * 10_000.0

    @property
    def participation(self) -> Optional[float]:
        if not self.bar_volume or self.bar_volume <= 0:
            return None
        return self.qty / self.bar_volume

    @property
    def quoted_half_spread_bps(self) -> Optional[float]:
        """Half the quoted spread in bps from the captured NBBO (the spread cost you
        pay regardless of size). None unless both bid and ask were captured."""
        if not self.bid or not self.ask or self.ask <= self.bid:
            return None
        mid = (self.bid + self.ask) / 2.0
        if mid <= 0:
            return None
        return ((self.ask - self.bid) / 2.0 / mid) * 10_000.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["slippage_bps"] = self.slippage_bps
        d["participation"] = self.participation
        d["quoted_half_spread_bps"] = self.quoted_half_spread_bps
        return d


class CalibrationStore:
    """Append-only JSONL store of CalibrationRecords."""

    def __init__(self, path: str = "data/calibration_fills.jsonl") -> None:
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    def append(self, rec: CalibrationRecord) -> None:
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec.to_dict(), default=str) + "\n")

    def read_all(self) -> List[CalibrationRecord]:
        out: List[CalibrationRecord] = []
        if not os.path.exists(self.path):
            return out
        with open(self.path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                # drop derived fields not in the dataclass
                row.pop("slippage_bps", None)
                row.pop("participation", None)
                row.pop("quoted_half_spread_bps", None)
                out.append(CalibrationRecord(**row))
        return out

    def count(self) -> int:
        return len(self.read_all())
