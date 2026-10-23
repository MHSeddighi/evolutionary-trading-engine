"""Phase 2 candle store: dockerized TimescaleDB (postgres + time-series extension).

Candles live in table `candles`:

    (symbol text, datetime timestamptz, open/high/low/close/volume double precision)
    PRIMARY KEY (symbol, datetime), hypertable partitioned县委 by datetime.

DSN from env WALLEX_DSN, default:
    postgresql://trader:trader@localhost:5434/tradingbot
Start the image with:  docker compose up -d candles-db

Bulk upsert (ON CONFLICT) keeps refresh idempotent; read_pair returns the exact
CSVSource-style frame (datetime,open,high,low,close,volume, UTC) so
FeatureBuilder / LiveEvaluator work unchanged. CSV caches remain a fallback:
DatabaseSource.migrate_csvs() imports existing dataset/candles_live/*.csv.
"""
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from src.data.source import CANDLE_COLUMNS, Source

logger = logging.getLogger(__name__)

DEFAULT_DSN = "postgresql://trader:trader@localhost:5434/tradingbot"
TABLE = "candles"


def db_available(dsn: Optional[str] = None, timeout: float = 3.0) -> bool:
    dsn = dsn or os.environ.get("WALLEX_DSN", DEFAULT_DSN)
    try:
        import psycopg
        with psycopg.connect(dsn, connect_timeout=timeout):
            return True
    except Exception as e:
        logger.warning("Candle DB not reachable (%s): %s", dsn, e)
        return False


class DatabaseSource(Source):
    """Candle storage in the TimescaleDB container (phase-2 store of record)."""

    def __init__(self, dsn: Optional[str] = None):
        self.dsn = dsn or os.environ.get("WALLEX_DSN", DEFAULT_DSN)
        self.ensure_schema()

    # ---------------- schema ----------------

    def ensure_schema(self):
        import psycopg
        with psycopg.connect(self.dsn, autocommit=True) as conn:
            conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {TABLE} (
                    symbol   text NOT NULL,
                    datetime timestamptz NOT NULL,
                    open  double precision NOT NULL,
                    high  double precision NOT NULL,
                    low   double precision NOT NULL,
                    close double precision NOT NULL,
                    volume double precision NOT NULL DEFAULT 0,
                    PRIMARY KEY (symbol, datetime)
                )""")
            is_hypertable = 0
            try:
                is_hypertable = conn.execute(
                    "SELECT count(*) FROM timescaledb_information.hypertables "
                    "WHERE hypertable_name = %s", (TABLE,)).fetchone()[0]
            except Exception:
                pass  # plain postgres image: no timescale extension
            if not is_hypertable:
                try:
                    conn.execute("CREATE EXTENSION IF NOT EXISTS timescaledb")
                    conn.execute(
                        f"SELECT create_hypertable('{TABLE}', 'datetime', "
                        "if_not_exists => TRUE, migrate_data => TRUE)")
                    logger.info("TimescaleDB hypertable enabled")
                except Exception as e:
                    logger.info("Plain postgres table (no timescale ext): %s", e)

    # ---------------- Source contract ----------------

    def read_pair(self, pair: str, **kwargs) -> pd.DataFrame:
        import psycopg
        with psycopg.connect(self.dsn) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"SELECT datetime, open, high, low, close, volume "
                    f"FROM {TABLE} WHERE symbol = %s ORDER BY datetime",
                    (pair,))
                df = pd.DataFrame(cur.fetchall(),
                                  columns=CANDLE_COLUMNS)
        if df.empty:
            return pd.DataFrame(columns=CANDLE_COLUMNS)
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True) \
                          .dt.tz_convert(None)
        for c in ("open", "high", "low", "close", "volume"):
            df[c] = df[c].astype(float)
        return df[CANDLE_COLUMNS].sort_values("datetime").reset_index(drop=True)

    def list_pairs(self) -> List[str]:
        import psycopg
        with psycopg.connect(self.dsn) as conn:
            rows = conn.execute(
                f"SELECT DISTINCT symbol FROM {TABLE} ORDER BY symbol").fetchall()
        return [r[0] for r in rows]

    # ---------------- writes ----------------

    def upsert_candles(self, symbol: str, df: pd.DataFrame) -> int:
        """Bulk upsert (conflict = no-op). `df` uses CANDLE_COLUMNS, naive UTC."""
        if df.empty:
            return 0
        df = df[CANDLE_COLUMNS].dropna().drop_duplicates("datetime")
        rows = list(df.itertuples(index=False, name=None))
        ts = [(symbol, r[0], r[1], r[2], r[3], r[4], r[5]) for r in rows]
        return self._copy_upsert(ts)

    def _copy_upsert(self, rows) -> int:
        """Efficient upsert: COPY into temp table, INSERT ... ON CONFLICT."""
        import psycopg
        for attempt in range(2):
            try:
                with psycopg.connect(self.dsn, autocommit=True) as conn:
                    # note: ON COMMIT DROP would drop instantly under autocommit;
                    # temp tables die with the session anyway
                    conn.execute(f"CREATE TEMP TABLE _stage (LIKE {TABLE} "
                                 "INCLUDING DEFAULTS)")
                    with conn.cursor() as cur:
                        with cur.copy(f"COPY _stage (symbol, datetime, open, "
                                      "high, low, close, volume) FROM STDIN") as cp:
                            for r in rows:
                                cp.write_row(r)
                    with conn.cursor() as cur:
                        cur.execute(
                            f"INSERT INTO {TABLE} SELECT * FROM _stage "
                            "ON CONFLICT (symbol, datetime) DO NOTHING")
                        return cur.rowcount
            except psycopg.errors.UndefinedTable:
                self.ensure_schema()
        return 0

    def latest_bar(self, symbol: str) -> Optional[pd.Timestamp]:
        import psycopg
        with psycopg.connect(self.dsn) as conn:
            r = conn.execute(f"SELECT max(datetime) FROM {TABLE} WHERE symbol=%s",
                             (symbol,)).fetchone()
        return r[0] if r and r[0] else None

    def delete_symbol(self, symbol: str):
        import psycopg
        with psycopg.connect(self.dsn, autocommit=True) as conn:
            conn.execute(f"DELETE FROM {TABLE} WHERE symbol = %s", (symbol,))

    # ---------------- CSV migration ----------------

    def migrate_csvs(self, cache_dir: str = "dataset/candles_live",
                     research_dir: str = "dataset/candles") -> Dict[str, int]:
        """Import existing CSV candle caches (live + phase-1 research) into DB."""
        out = {}
        for root in (cache_dir, research_dir):
            path = Path(root)
            if not path.exists():
                continue
            for f in sorted(path.glob("*_5m.csv")):
                sym = f.name.rsplit("_", 1)[0]
                df = pd.read_csv(f)
                df["datetime"] = pd.to_datetime(df["datetime"])
                out[sym] = self.upsert_candles(sym, df)
                logger.info("migrated %s: %d rows from %s", sym, out[sym], f)
        return out
