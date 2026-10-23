"""Storage facade: candles persisted in the dockerized DB, CSV as fallback.

LiveStackSource implements the Source contract used by LiveEvaluator/WallexSource:

    choose_store()   -> DatabaseSource if WALLEX_DSN reachable else CSVSource-like
    ensure_history() -> fetch from Wallex, upsert into the ACTIVE store (DB or CSV)
    read_pair()      -> read from the ACTIVE store (identical interface)

Migration helpers keep both stores consistent when switching:
    csv_to_db(), db_to_csv().
"""
import logging
import threading
import time
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from src.data.database_source import DatabaseSource, db_available
from src.data.source import CANDLE_COLUMNS
from src.data.wallex_source import WallexSource, resample_1m_to_5m, udf_to_frame

logger = logging.getLogger(__name__)

BAR_SECONDS = 300


class LiveCandleStore:
    """Wallex fetch layer backed by DB-first storage with CSV fallback.

    The DB is re-probed after a cooldown if it disappears mid-run; every read
    and write degrades to the CSV mirror instead of killing the trading loop.
    """

    DB_RETRY_SECONDS = 300

    def __init__(self, client, cache_dir: str = "dataset/candles_live",
                 dsn: Optional[str] = None, warmup_bars: Optional[int] = None,
                 force_csv: bool = False):
        self.client = client
        if warmup_bars is None:
            from src.live.evaluator import default_warmup_bars
            warmup_bars = default_warmup_bars()
        self.warmup_bars = int(warmup_bars)
        self.force_csv = force_csv
        self.dsn = dsn
        self.db: Optional[DatabaseSource] = None
        self._db_retry_at = 0.0
        if not force_csv and db_available(dsn):
            try:
                self.db = DatabaseSource(dsn)
                logger.info("Candle store: TimescaleDB (docker) ACTIVE")
            except Exception as e:
                logger.warning("DB init failed (%s) — falling back to CSV", e)
        if self.db is None:
            logger.info("Candle store: CSV cache ACTIVE (%s)", cache_dir)
        self.cache_dir = Path(cache_dir)
        self._locks: Dict[str, threading.Lock] = {}
        logger.info("Feature warm-up requirement: %d bars (~%.1f days)",
                    self.warmup_bars, self.warmup_bars * BAR_SECONDS / 86400)

    # ---------------- storage selection ----------------

    @property
    def is_db(self) -> bool:
        return self.db is not None

    def _db_failed(self, err: Exception):
        if self.db is not None:
            logger.error("Candle DB failed (%s) — switching to CSV mirror; "
                         "will re-probe in %ds", err, self.DB_RETRY_SECONDS)
        self.db = None
        self._db_retry_at = time.time() + self.DB_RETRY_SECONDS

    def _maybe_retry_db(self):
        if self.db is not None or self.force_csv:
            return
        if time.time() < self._db_retry_at:
            return
        self._db_retry_at = time.time() + self.DB_RETRY_SECONDS
        if db_available(self.dsn, timeout=2.0):
            try:
                self.db = DatabaseSource(self.dsn)
                logger.info("Candle DB reconnected — back to DB store")
            except Exception as e:
                self._db_failed(e)

    def _read(self, pair: str) -> pd.DataFrame:
        self._maybe_retry_db()
        if self.db is not None:
            try:
                return self.db.read_pair(pair)
            except Exception as e:
                self._db_failed(e)
        path = self.cache_dir / f"{pair}_5m.csv"
        if not path.exists():
            return pd.DataFrame(columns=CANDLE_COLUMNS)
        df = pd.read_csv(path)
        df["datetime"] = pd.to_datetime(df["datetime"])
        return df[CANDLE_COLUMNS].sort_values("datetime").reset_index(drop=True)

    def _append(self, pair: str, bars: pd.DataFrame) -> int:
        new = bars[CANDLE_COLUMNS].dropna().drop_duplicates("datetime")
        if new.empty:
            return 0
        if self.db is not None:
            try:
                return self.db.upsert_candles(pair, new)
            except Exception as e:
                self._db_failed(e)
        with self._lock(pair):
            path = self.cache_dir / f"{pair}_5m.csv"
            if not path.exists():
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                new.sort_values("datetime").to_csv(path, index=False)
                return len(new)
            existing = pd.read_csv(path)
            existing["datetime"] = pd.to_datetime(existing["datetime"])
            # Upsert semantics: accept any bar not already present (covers
            # internal gaps, not just appends after the last row).
            have = set(existing["datetime"])
            fresh = new[~new["datetime"].isin(have)].sort_values("datetime")
            if fresh.empty:
                return 0
            pd.concat([existing[CANDLE_COLUMNS], fresh], ignore_index=True) \
                .drop_duplicates("datetime").sort_values("datetime") \
                .to_csv(path, index=False)
            return len(fresh)

    def _lock(self, pair: str) -> threading.Lock:
        if pair not in self._locks:
            self._locks[pair] = threading.Lock()
        return self._locks[pair]

    # ---------------- Source contract for LiveEvaluator ----------------

    def read_pair(self, pair: str, **kwargs) -> pd.DataFrame:
        return self._read(pair)

    def list_pairs(self):
        if self.db is not None:
            return self.db.list_pairs()
        return sorted({p.name.rsplit("_", 1)[0]
                       for p in self.cache_dir.glob("*_5m.csv")})

    # ---------------- fetching (shared Wallex logic) ----------------

    def _fetch_window(self, pair: str, start_ts: int, end_ts: int) -> pd.DataFrame:
        chunks, step, cur = [], 6 * 3600, (start_ts // 60) * 60
        end_ts = (end_ts // 60) * 60
        total = max(1, int((end_ts - cur) // step) + 1)
        i = 0
        while cur < end_ts:
            nxt = min(cur + step, end_ts)
            payload = self.client.get_klines(pair, "1", cur, nxt)
            df = udf_to_frame(payload)
            if not df.empty:
                chunks.append(df)
            cur = nxt
            i += 1
            if i % 40 == 0:
                logger.info("backfill %s: %d/%d chunks (%.0f%%)",
                            pair, i, total, 100.0 * i / total)
            time.sleep(0.05)
        if not chunks:
            return pd.DataFrame(columns=CANDLE_COLUMNS)
        one_min = pd.concat(chunks, ignore_index=True) \
                   .drop_duplicates("datetime").sort_values("datetime")
        return one_min

    def ensure_history(self, pair: str) -> dict:
        """Guarantee >= warmup_bars and fill every gap up to the latest minute.

        The warm-up is feature-driven (60 days for the 30d horizon block), not
        a fixed 7 days: without it the long-horizon rules can never fire live.
        """
        with self._lock(pair):
            cache = self._read(pair)
        now = int(time.time())
        if cache.empty or len(cache) < self.warmup_bars:
            # 5% calendar headroom for exchange no-trade gaps
            start_ts = now - int(self.warmup_bars * BAR_SECONDS * 1.05)
            if not cache.empty:
                start_ts = min(start_ts,
                               int(cache["datetime"].min().value // 10**9))
            logger.info("%s: backfill %d bars requested (have %d)",
                        pair, self.warmup_bars, len(cache))
            bars = resample_1m_to_5m(self._fetch_window(pair, start_ts, now))
            added = self._append(pair, bars)
            # preserve a CSV mirror of the DB store for portability
            if self.db is not None:
                self._mirror_to_csv(pair)
            return {"backfilled": True, "was_empty": cache.empty,
                    "bars_added": added, "have": len(cache) + added}
        last = cache["datetime"].max()
        gap_start = int(last.value // 10**9) - 5 * 60
        if now - gap_start > 60 * 60:
            logger.info("%s: filling %.1fh gap since %s", pair,
                        (now - gap_start) / 3600, last)
        bars = resample_1m_to_5m(self._fetch_window(pair, gap_start, now))
        added = self._append(pair, bars)
        if self.db is not None and added:
            self._mirror_to_csv(pair)
        return {"backfilled": False, "bars_added": added, "have": len(cache) + added}

    def refresh_symbols(self, pairs, max_workers: int = 4) -> Dict[str, dict]:
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

    # ---------------- mirroring / migration ----------------

    def _mirror_to_csv(self, pair: str):
        if self.db is None:
            return
        df = self.db.read_pair(pair)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.cache_dir / f"{pair}_5m.csv", index=False)

    def mirror_all_to_csv(self):
        for sym in self.list_pairs():
            self._mirror_to_csv(sym)

    def csv_to_db(self, cache_dir: str = "dataset/candles_live"):
        if self.db is None:
            logger.warning("DB store inactive — start docker compose up -d candles-db")
            return {}
        return DatabaseSource(self.db.dsn).migrate_csvs(cache_dir=cache_dir)
