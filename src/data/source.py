"""Data sources for candle / prebuilt datasets.

Phase 1 (research) uses CSVSource reading {PAIR}_{timeframe}.csv from dataset/candles.
Phase 2 (live) swaps in DatabaseSource reading candle tables from the Wallex-side
database. Both share the Dataset assembly contract below.
"""
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

CANDLE_COLUMNS = ["datetime", "open", "high", "low", "close", "volume"]


class Source:
    def read_pair(self, pair: str, **kwargs) -> pd.DataFrame:  # pragma: no cover
        raise NotImplementedError

    def list_pairs(self) -> List[str]:
        raise NotImplementedError


class CSVSource(Source):
    """Phase 1 source: one CSV per pair with columns datetime,open,high,low,close,volume."""

    def __init__(self, root: str = "dataset/candles", timeframe: str = "5m"):
        self.root = Path(root)
        self.timeframe = timeframe

    def read_pair(self, pair: str, **kwargs) -> pd.DataFrame:
        df = pd.read_csv(self.root / f"{pair}_{self.timeframe}.csv")
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df[CANDLE_COLUMNS].sort_values("datetime").reset_index(drop=True)
        return df

    def list_pairs(self) -> List[str]:
        return sorted({p.name.rsplit("_", 1)[0] for p in self.root.glob("*_*.csv")})


class DatabaseSource(Source):
    """Phase 2 source stub: Wallex-side candle tables (e.g. table per pair).

    Implemented in phase 2; contract identical to CSVSource so every downstream
    module works unchanged.
    """

    def __init__(self, connection=None, timeframe: str = "5m"):
        self.connection = connection
        self.timeframe = timeframe

    def read_pair(self, pair: str, **kwargs) -> pd.DataFrame:
        raise NotImplementedError("DatabaseSource arrives in phase 2")

    def list_pairs(self) -> List[str]:
        raise NotImplementedError("DatabaseSource arrives in phase 2")
