#!/usr/bin/env python3
"""Download 5m OHLCV candles from Binance public klines API (no key needed).

Stores one CSV per symbol with simple column names:
    datetime, open, high, low, close, volume
Resumable via a `since` cursor and weekly chunking.
"""
import argparse
import os
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

BASE_URL = "https://api.binance.com/api/v3/klines"
LIMIT = 1000
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT"]
COLUMNS = ["open", "high", "low", "close", "volume"]
DEFAULT_OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dataset", "candles")


def fetch_chunk(pair: str, start_ms: int, end_ms: int, session: requests.Session):
    for attempt in range(5):
        try:
            r = session.get(
                BASE_URL,
                params={"symbol": pair, "interval": "5m", "startTime": start_ms,
                        "endTime": end_ms, "limit": LIMIT},
                timeout=30,
            )
            if r.status_code == 200:
                return r.json()
            if r.status_code == 429:
                time.sleep(int(r.headers.get("Retry-After", 5)))
                continue
        except requests.RequestException:
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"Failed to fetch klines for {pair} at {start_ms}")


def download_pair(pair: str, start: datetime, end: datetime, out_dir: str) -> str:
    path = os.path.join(out_dir, f"{pair}_5m.csv")
    existing = None
    if os.path.exists(path):
        existing = pd.read_csv(path, parse_dates=["datetime"])
        last_dt = existing["datetime"].iloc[-1].to_pydatetime()
        if last_dt >= end - timedelta(minutes=5) and last_dt <= end:
            print(f"  {pair}: up to date ({len(existing):,} candles)")
            return path
        start_cursor = last_dt + timedelta(minutes=5)
    else:
        existing = None
        start_cursor = start

    session = requests.Session()
    cur = int(start_cursor.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    rows = []
    while cur < end_ms:
        data = fetch_chunk(pair, cur, end_ms, session)
        if not data:
            break
        rows.extend(data)
        cur = data[-1][0] + 60_000
        time.sleep(0.05)

    if not rows:
        print(f"  {pair}: no new candles")
        return path

    df = pd.DataFrame(rows, columns=[
        "open_time", "open", "high", "low", "close", "volume", "close_time",
        "quote_volume", "trades", "taker_base", "taker_quote", "ignore"])
    df["datetime"] = pd.to_datetime(df["open_time"], unit="ms", utc=True).dt.tz_localize(None)
    df = df[["datetime"] + COLUMNS].astype({c: float for c in COLUMNS})
    df = df.reset_index(drop=True)

    if existing is not None:
        df = pd.concat([existing, df], ignore_index=True)
        df = df.drop_duplicates(subset="datetime", keep="last").sort_values("datetime")
    df.to_csv(path, index=False)
    print(f"  {pair}: {len(df):,} candles -> {path}")
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--months", type=float, default=8.0, help="total history length")
    ap.add_argument("--pairs", nargs="*", default=SYMBOLS)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    end = datetime.now(timezone.utc).replace(second=0, microsecond=0, tzinfo=None)
    end = end.replace(minute=(end.minute // 5) * 5) - timedelta(minutes=5)
    start = end - timedelta(days=int(args.months * 30.44))

    for pair in args.pairs:
        print(f"⬇️  {pair}")
        download_pair(pair, start, end, args.out)
    print("✅ Download complete")


if __name__ == "__main__":
    main()
