"""
Download a crypto-basket of daily bars (Alpaca crypto API) into a cache for the
harness/discovery to test — a genuinely DIFFERENT asset class (24/7, less efficient,
historically more trend-persistent than equities). Filenames sanitize "BTC/USD" -> "BTCUSD"
so the existing discover_universe / truth_table scripts can read them.

Run:
  .venv/bin/python scripts/download_crypto.py --years 4 --out "/Volumes/My Passport/AlpcaData/crypto"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DEFAULT = ("BTC/USD,ETH/USD,LTC/USD,BCH/USD,LINK/USD,UNI/USD,AAVE/USD,DOGE/USD,SOL/USD,"
           "AVAX/USD,DOT/USD,SHIB/USD,MKR/USD,GRT/USD,CRV/USD,XTZ/USD,SUSHI/USD,YFI/USD,BAT/USD,DOGE/USD")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default=DEFAULT)
    ap.add_argument("--timeframe", default="1day")
    ap.add_argument("--years", type=float, default=4.0)
    ap.add_argument("--out", default="data/crypto")
    args = ap.parse_args()

    from alpca.config import load_config
    from alpca.data.bars import fetch_alpaca_crypto_bars

    cfg = load_config()
    if not cfg.has_credentials:
        print("[fail] no Alpaca credentials", file=sys.stderr)
        return 1
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    days = int(args.years * 365)
    syms = sorted(set(s.strip().upper() for s in args.symbols.split(",") if s.strip()))
    print(f"[ok] {len(syms)} crypto symbols, {args.timeframe}, ~{args.years}y -> {out}")
    total = 0
    for sym in syms:
        try:
            bars = fetch_alpaca_crypto_bars(cfg, sym, timeframe=args.timeframe, days=days)
        except Exception as e:
            print(f"  {sym}: FAIL {type(e).__name__}: {e}")
            continue
        if not bars:
            print(f"  {sym}: no data (not on Alpaca?)")
            continue
        fn = sym.replace("/", "") + f"_{args.timeframe}_bars.jsonl"
        with (out / fn).open("w") as f:
            for b in bars:
                f.write(json.dumps(b) + "\n")
        total += len(bars)
        print(f"  {sym}: {len(bars)} bars -> {fn}")
    print(f"\n[done] {total} bars across crypto basket -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
