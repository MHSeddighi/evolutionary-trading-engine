"""Assemble the pooled all-symbols arrays consumed by everything downstream.

The contract (identical to the old project, now symbol-pooled):
  features : Dict[str, np.ndarray]   feature name -> column (all symbols concatenated)
  feature_cols : List[str]
  labels   : Dict[str, np.ndarray]   label_open_next, label_close_288/288 min/max, label_max_before_min
  meta     : Dict[str, np.ndarray]   datetime, symbol (int), _symbol_bar_index

Rows are sorted by (datetime, symbol); _symbol_bar_index is per-symbol sequential so
rolling windows, labels and the backtester never cross symbol boundaries.
"""
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

LABEL_COLS = [
    "label_open_next", "label_close_288", "label_min_288", "label_max_288", "label_max_before_min",
]
META_COLS = ["datetime", "symbol"]
EXTRA_COLS = ["candle_open", "candle_high", "candle_low", "candle_close", "candle_volume"]  # raw candles, not features
LABEL_WINDOW = 288


def assemble(built: pd.DataFrame) -> Tuple[Dict[str, np.ndarray], List[str], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Take the builder's output (feature cols + label cols + meta) and produce arrays."""
    df = built.sort_values(["datetime", "symbol"]).reset_index(drop=True)
    df["_symbol_bar_index"] = df.groupby("symbol").cumcount().astype(np.int32)
    # integer symbol codes (backtester expects numeric codes)
    sym_codes, sym_uniques = pd.factorize(df["symbol"], sort=True)
    df["symbol"] = sym_codes.astype(np.int32) + 1

    feature_cols = [c for c in df.columns
                    if c not in LABEL_COLS + META_COLS + EXTRA_COLS + ["_symbol_bar_index"]]
    df[feature_cols] = df[feature_cols].fillna(0.0).astype(np.float32)

    features: Dict[str, np.ndarray] = {c: df[c].to_numpy() for c in feature_cols}
    labels: Dict[str, np.ndarray] = {c: df[c].to_numpy(np.float32 if c != "label_max_before_min" else np.int8)
                                     for c in LABEL_COLS if c in df.columns}
    meta: Dict[str, np.ndarray] = {
        "_symbol_bar_index": df["_symbol_bar_index"].to_numpy(np.int32),
        "symbol": df["symbol"].to_numpy(np.int32),
        "symbol_names": np.asarray(sym_uniques),
        "datetime": df["datetime"].to_numpy(),
    }
    return features, feature_cols, labels, meta
