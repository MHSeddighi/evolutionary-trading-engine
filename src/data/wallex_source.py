"""Phase 2 data source: Wallex klines -> phase-1 CSV candle cache.

Contract identical to CSVSource (Source.read_pair -> CANDLE_COLUMNS frame), so
FeatureBuilder consumes it unchanged.

Wallex has no 5m resolution (valid: 1, 15, 60, 240, 720, 1D...), so:

    GET /v1/udf/history?resolution=1   (1-minute candles)
    -> filtered to 1-minute bars sorted by candle-start timestamp
    -> bucketed to 5m UTC boundaries (ts // 300 * 300, open/high/low/close/sum volume)
    -> merged into an incremental CSV cache dataset/candles_live/{PAIR}_5m.csv

Candle timestamps = candle START (open time), matching the Binance-backed
phase-1 CSVs from scripts/download_data.py. Optional backfill fetches enough
1m history to cover the 2016-bar (7-day) feature warm-up window.
"""
import logging
import os
import threading
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from src.data.source import CANDLE_COLUMNS, Source
from src.execution.wallex import WallexClient

logger = logging.getLogger(__name__)

BAR_SECONDS = 300          # 5m
RESOLUTION = "1"           # fetch 1m, aggregate to 5m


def resample_1m_to_5m(df: pd.DataFrame) -> pd.DataFrame:
    """1m candle-open timestamps -> aligned 5m buckets, continuous via gap-fill.

    Wallex 1m klines are sparse (no-trade minutes are absent), so 5m buckets
    are reindexed onto a dense grid and flat-filled with the previous close
    (volume 0). This keeps the rolling feature windows aligned like the
    continuous research candles did.
    """
    df = df.sort_values("datetime").reset_index(drop=True)
    df["bucket"] = df["datetime"].values.astype("datetime64[s]").astype("int64") \
                   // BAR_SECONDS * BAR_SECONDS
    df["bucket"] = pd.to_datetime(df["bucket"], unit="s")
    gg = df.groupby("bucket")
    agg = gg.agg(open=("open", "first"), high=("high", "max"),
                 low=("low", "min"), close=("close", "last"),
                 volume=("volume", "sum")).reset_index()
    agg = agg.rename(columns={"index" if "index" in agg.columns else
                              ("bucket" if "bucket" in agg.columns else agg.columns[0]):
                              "datetime"})
    if agg.empty:
        return pd.DataFrame(columns=CANDLE_COLUMNS)
    dense = pd.DataFrame({"datetime": pd.date_range(
        agg["datetime"].min(), agg["datetime"].max(), freq=f"{BAR_SECONDS}s")})
    agg = dense.merge(agg, on="datetime", how="left")
    agg[["open", "close"]] = agg[["open", "close"]].ffill()
    agg["high"] = agg["high"].fillna(agg["open"])
    agg["low"] = agg["low"].fillna(agg["open"])
    agg["volume"] = agg["volume"].fillna(0.0)
    fill_mask = agg["open"].isna()
    if fill_mask.any():  # fully empty dense head
        agg = agg[~fill_mask].reset_index(drop=True)
    return agg[CANDLE_COLUMNS]


def udf_to_frame(payload: Dict[str, list]) -> pd.DataFrame:
    """{t,o,h,l,c,v} udf arrays -> 1m candle frame filtered to exact 1m bars."""
    t = payload.get("t") or []
    if not t:
        return pd.DataFrame(columns=CANDLE_COLUMNS)
    n = len(t)
    df = pd.DataFrame({
        "datetime": pd.to_datetime(np.asarray(t[:n], dtype="int64"), unit="s"),
        "open": pd.to_numeric(payload.get("o", [])[:n], errors="coerce"),
        "high": pd.to_numeric(payload.get("h", [])[:n], errors="coerce"),
        "low": pd.to_numeric(payload.get("l", [])[:n], errors="coerce"),
        "close": pd.to_numeric(payload.get("c", [])[:n], errors="coerce"),
        "volume": pd.to_numeric(payload.get("v", [])[:n], errors="coerce"),
    })
    # candle-start alignment: keep only timestamps whose seconds % 60 == 0
    df = df[df["datetime"].dt.second == 0]
    df = df.dropna().sort_values("datetime").drop_duplicates("datetime")
    return df.reset_index(drop=True)[CANDLE_COLUMNS]


class WallexSource(Source):
    """Wallex-backed candle cache source (phase 2 live data)."""

    def __init__(self, client: WallexClient, cache_dir: str = "dataset/candles_live",
                 timeframe: str = "5m", warmup_bars: int = 2016,
                 backfill_on_start: bool = True):
        self.client = client
        self.cache_dir = Path(cache_dir)
        self.timeframe = timeframe
        self.warmup_bars = warmup_bars
        self.backfill_on_start = backfill_on_start
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._locks: Dict[str, threading.Lock] = {}
        self._backfilled: set = set()

    # ---------------- Source contract ----------------

    def read_pair(self, pair: str, **kwargs) -> pd.DataFrame:
        """Cached 5m candles (auto-backfill/refresh unless disabled in kwargs)."""
        refresh = kwargs.get("refresh", True)
        if refresh:
            self.ensure_history(pair)
        return self._read_cache(pair)

    def list_pairs(self) -> list:
        return sorted({p.name.rsplit("_", 1)[0] for p in self.cache_dir.glob("*_*.csv")})

    # ---------------- cache mechanics ----------------

    def _path(self, pair: str) -> Path:
        return self.cache_dir / f"{pair}_{self.timeframe}.csv"

    def _lock(self, pair: str) -> threading.Lock:
        if pair not in self._locks:
            self._locks[pair] = threading.Lock()
        return self._locks[pair]

    def _read_cache(self, pair: str) -> pd.DataFrame:
        path = self._path(pair)
        if not path.exists():
            return pd.DataFrame(columns=CANDLE_COLUMNS)
        df = pd.read_csv(path)
        df["datetime"] = pd.to_datetime(df["datetime"])
        return df[CANDLE_COLUMNS].sort_values("datetime").reset_index(drop=True)

    def _write_cache(self, pair: str, df: pd.DataFrame):
        df = df[CANDLE_COLUMNS].sort_values("datetime").drop_duplicates("datetime")
        df.to_csv(self._path(pair), index=False)

    def _append(self, pair: str, new_5m: pd.DataFrame) -> int:
        if new_5m.empty:
            return 0
        with self._lock(pair):
            existing = self._read_cache(pair)
            if not existing.empty and not new_5m.empty:
                new_5m = new_5m[new_5m["datetime"] > existing["datetime"].max()]
            if new_5m.empty:
                return 0
            self._write_cache(pair, pd.concat([existing, new_5m], ignore_index=True))
            return len(new_5m)

    # ---------------- fetching ----------------

    def _fetch_window(self, pair: str, start_ts: int, end_ts: int) -> pd.DataFrame:
        """Fetch 1m candles in time chunks (UDF windows can be capped server-side)."""
        chunks = []
        step = 6 * 3600  # udf caps per-request range; 6h of 1m ≈ 360 candles
        cur = (start_ts // 60) * 60  # align to minute so udf buckets come back aligned
        end_ts = (end_ts // 60) * 60
        while cur < end_ts:
            nxt = min(cur + step, end_ts)
            payload = self.client.get_klines(pair, RESOLUTION, cur, nxt)
            df = udf_to_frame(payload)
            if not df.empty:
                chunks.append(df)
            cur = nxt
            time.sleep(0.15)  # gentle pacing across symbols
        if not chunks:
            return pd.DataFrame(columns=CANDLE_COLUMNS)
        one_min = pd.concat(chunks, ignore_index=True)
        one_min = one_min.drop_duplicates("datetime").sort_values("datetime")
        return one_min

    def ensure_history(self, pair: str) -> dict:
        """Guarantee >= warmup_bars cached candles; backfill or increment as needed."""
        with self._lock(pair):
            cache = self._read_cache(pair)
        now = int(time.time())
        last = None if cache.empty else cache["datetime"].max()
        need_backfill = cache.empty or len(cache) < self.warmup_bars
        if need_backfill:
            start_ts = now - int(self.warmup_bars * BAR_SECONDS * 1.4)  # 40% headroom
            if not cache.empty:
                start_ts = min(start_ts, int(cache["datetime"].min().timestamp()))
            first = cache.empty
            one_min = self._fetch_window(pair, start_ts, now)
            bars = resample_1m_to_5m(one_min)
            self._append(pair, bars)
            self._backfilled.add(pair)
            return {"backfilled": True, "was_empty": first, "bars_added": len(bars)}
        # incremental: fill the gap since the last cached bar
        gap_start = int(last.timestamp()) - 5 * 60
        one_min = self._fetch_window(pair, gap_start, now)
        bars = resample_1m_to_5m(one_min)
        added = self._append(pair, bars)
        return {"backfilled": False, "bars_added": added}

    def refresh_symbols(self, pairs, max_workers: int = 4) -> Dict[str, dict]:
        """Threaded ensure_history for many symbols (latency budget control)."""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        out = {}
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            fut = {pool.submit(self.ensure_history, p): p for p in pairs}
            for f in as_completed(fut):
                p = fut[f]
                try:
                    out[p] = f.result()
                except Exception as e:
                    logger.warning("refresh failed for %s: %s", p, e)
                    out[p] = {"error": str(e)}
        return out
