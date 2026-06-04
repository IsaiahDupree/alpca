"""
Append-only, hash-chained order event ledger.

Every meaningful order-lifecycle transition is appended as one JSONL line:

    {seq, ts, event, order, prev_hash, hash}

`hash` = SHA-256 over canonical JSON of (seq, ts, event, order, prev_hash), so
tampering with any past line breaks the chain at the next line — verify_chain()
reports the first broken seq.

This is the durable system-of-record for the latency metrics: each line snapshots
the order (including lifecycle timestamps + computed latencies), so a full
signal->order->fill timeline can be reconstructed and audited after the fact.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Iterator, List, Optional

from alpca.execution.order import Order


class EventType(str, Enum):
    SIGNAL = "SIGNAL"
    SUBMIT = "SUBMIT"
    ACK = "ACK"
    PARTIAL_FILL = "PARTIAL_FILL"
    FILL = "FILL"
    CANCEL = "CANCEL"
    REJECT = "REJECT"
    RISK_BLOCK = "RISK_BLOCK"
    EXPIRE = "EXPIRE"      # a DAY order that rested past its session
    REPLACE = "REPLACE"    # cancel-replace: the superseded order
    TRIGGER = "TRIGGER"    # a STOP/STOP_LIMIT whose stop price was touched


def _canonical(obj: Dict[str, Any]) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")


def _hash(seq: int, ts: float, event: str, order: Dict[str, Any], prev_hash: str) -> str:
    payload = {"seq": seq, "ts": ts, "event": event, "order": order, "prev_hash": prev_hash}
    return hashlib.sha256(_canonical(payload)).hexdigest()


@dataclass
class ChainCheck:
    ok: bool
    total: int
    broken_at_seq: Optional[int] = None


class OrderEventLog:
    """Thread-safe append-only ledger backed by a single JSONL file."""

    GENESIS = "0" * 64

    def __init__(self, path: str = "data/order_events.jsonl") -> None:
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._lock = threading.Lock()
        self._seq, self._last_hash = self._scan_tail()

    def _scan_tail(self) -> tuple[int, str]:
        """Resume seq + last hash from an existing file so the chain continues."""
        seq, last_hash = -1, self.GENESIS
        if not os.path.exists(self.path):
            return seq, last_hash
        with open(self.path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "seq" in row and "hash" in row:
                    seq = row["seq"]
                    last_hash = row["hash"]
        return seq, last_hash

    def append(self, order: Order, event: EventType, ts: Optional[float] = None) -> Dict[str, Any]:
        ts = ts if ts is not None else time.time()
        with self._lock:
            self._seq += 1
            seq = self._seq
            order_snap = order.to_dict()
            h = _hash(seq, ts, event.value, order_snap, self._last_hash)
            row = {
                "seq": seq,
                "ts": ts,
                "event": event.value,
                "order": order_snap,
                "prev_hash": self._last_hash,
                "hash": h,
            }
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, default=str) + "\n")
            self._last_hash = h
            return row

    def read_all(self) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        if not os.path.exists(self.path):
            return rows
        with open(self.path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    def iter_events(self, event: Optional[EventType] = None) -> Iterator[Dict[str, Any]]:
        for row in self.read_all():
            if event is None or row.get("event") == event.value:
                yield row

    def verify_chain(self) -> ChainCheck:
        rows = self.read_all()
        prev = self.GENESIS
        chained = [r for r in rows if "seq" in r and "hash" in r]
        for r in chained:
            expected = _hash(r["seq"], r["ts"], r["event"], r["order"], prev)
            if expected != r["hash"] or r.get("prev_hash") != prev:
                return ChainCheck(ok=False, total=len(chained), broken_at_seq=r["seq"])
            prev = r["hash"]
        return ChainCheck(ok=True, total=len(chained))
