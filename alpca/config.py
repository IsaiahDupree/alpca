"""
Central configuration for Alpca — loaded once from environment.

Safety posture: PAPER by default. Live trading requires an explicit, separate
opt-in (ALPACA_PAPER=0 AND ALPACA_LIVE_CONFIRMED=I_UNDERSTAND). Nothing in this
package points at the live endpoint unless both are set.

Never logs secret values. `describe()` reports only whether keys are set + length.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

try:  # optional: load .env if python-dotenv is present
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover - dotenv is optional at runtime
    pass


PAPER_TRADING_URL = "https://paper-api.alpaca.markets"
LIVE_TRADING_URL = "https://api.alpaca.markets"


def _get_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _get_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _get_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() not in ("0", "false", "no")


@dataclass(frozen=True)
class RiskConfig:
    max_order_notional: float = 50_000.0
    daily_loss_pct: float = 0.02
    max_concentration_pct: float = 0.25
    max_open_positions: int = 20
    max_orders_per_min: int = 60
    # When True, a BUY whose notional exceeds available cash is rejected
    # (cash-account semantics — no margin). Mirrors Alpaca rejecting an order
    # that exceeds buying power.
    enforce_buying_power: bool = True
    # Short-selling. Default False = long/flat only (a SELL may only reduce or
    # close a long, never open/extend a short — the historical behavior). Set
    # True to allow shorts; short positions then accrue a daily borrow fee.
    allow_short: bool = False
    short_borrow_apr: float = 0.03   # annual borrow rate on short notional (~3%)

    @classmethod
    def from_env(cls) -> "RiskConfig":
        return cls(
            max_order_notional=_get_float("RISK_MAX_ORDER_NOTIONAL", 50_000.0),
            daily_loss_pct=_get_float("RISK_DAILY_LOSS_PCT", 0.02),
            max_concentration_pct=_get_float("RISK_MAX_CONCENTRATION_PCT", 0.25),
            max_open_positions=_get_int("RISK_MAX_OPEN_POSITIONS", 20),
            max_orders_per_min=_get_int("RISK_MAX_ORDERS_PER_MIN", 60),
            enforce_buying_power=_get_bool("RISK_ENFORCE_BUYING_POWER", True),
            allow_short=_get_bool("RISK_ALLOW_SHORT", False),
            short_borrow_apr=_get_float("RISK_SHORT_BORROW_APR", 0.03),
        )


@dataclass(frozen=True)
class AlpacaConfig:
    api_key: Optional[str]
    secret_key: Optional[str]
    paper: bool
    data_feed: str  # "iex" | "sip"
    risk: RiskConfig = field(default_factory=RiskConfig)

    @classmethod
    def from_env(cls) -> "AlpacaConfig":
        # Accept both Alpca-native and legacy alpaca-trade-api env names.
        api_key = os.getenv("ALPACA_API_KEY") or os.getenv("APCA_API_KEY_ID")
        secret_key = os.getenv("ALPACA_SECRET_KEY") or os.getenv("APCA_API_SECRET_KEY")
        paper_raw = os.getenv("ALPACA_PAPER", "1").strip().lower()
        paper = paper_raw not in ("0", "false", "no")
        # If the configured base URL clearly points at paper, force paper on.
        base = (os.getenv("ALPACA_BASE_URL") or os.getenv("ALPACA_TRADING_BASE_URL") or "").lower()
        if "paper-api" in base:
            paper = True
        data_feed = (os.getenv("ALPACA_DATA_FEED", "iex") or "iex").strip().lower()
        return cls(
            api_key=api_key,
            secret_key=secret_key,
            paper=paper,
            data_feed=data_feed,
            risk=RiskConfig.from_env(),
        )

    @property
    def trading_base_url(self) -> str:
        if self.paper:
            return os.getenv("ALPACA_TRADING_BASE_URL") or os.getenv("ALPACA_BASE_URL") or PAPER_TRADING_URL
        return LIVE_TRADING_URL

    @property
    def live_confirmed(self) -> bool:
        return os.getenv("ALPACA_LIVE_CONFIRMED", "") == "I_UNDERSTAND"

    @property
    def has_credentials(self) -> bool:
        return bool(self.api_key) and bool(self.secret_key)

    def require_credentials(self) -> None:
        if not self.has_credentials:
            raise RuntimeError(
                "Alpaca credentials missing. Set ALPACA_API_KEY and ALPACA_SECRET_KEY "
                "(see .env.example). Never hardcode them in source."
            )
        if not self.paper and not self.live_confirmed:
            raise RuntimeError(
                "Refusing to run against LIVE Alpaca. Set ALPACA_PAPER=1 for paper, or "
                "ALPACA_LIVE_CONFIRMED=I_UNDERSTAND to deliberately enable live."
            )

    def describe(self) -> str:
        """Human-readable status with NO secret values (lengths only)."""
        ak = f"set(len={len(self.api_key)})" if self.api_key else "MISSING"
        sk = f"set(len={len(self.secret_key)})" if self.secret_key else "MISSING"
        mode = "PAPER" if self.paper else ("LIVE (confirmed)" if self.live_confirmed else "LIVE (BLOCKED)")
        return (
            f"AlpacaConfig(mode={mode}, base_url={self.trading_base_url}, "
            f"data_feed={self.data_feed}, api_key={ak}, secret_key={sk})"
        )


def load_config() -> AlpacaConfig:
    return AlpacaConfig.from_env()
