"""Forward-looking labels — computed from completed candles, per symbol.

Definitions (identical to the legacy dataset):
    label_open_next     open price of the next candle (entry price)
    label_close_288     close price 288 candles later (time exit)
    label_min_288       min of lows over candles t+1 .. t+288
    label_max_288       max of highs over candles t+1 .. t+288
    label_max_before_min  1 if the window max occurred before the min (TP-first for longs)
"""
import numpy as np
import pandas as pd
import numba


@numba.njit(cache=True)
def _max_before_min(highs, lows, window):
    # Window must match label_max_288 / label_min_288 exactly: bars i+1..i+window
    # (the signal bar itself is already known at entry and must NOT arbitrate
    # the TP/SL ordering — it caused an off-by-one look-ahead bias).
    n = len(highs)
    out = np.full(n, -1, dtype=np.int8)
    for i in range(n):
        if i + window > n:
            continue
        first = i + 1
        max_v, max_i, min_v, min_i = highs[first], first, lows[first], first
        for j in range(first + 1, i + window + 1):
            h, l = highs[j], lows[j]
            if h > max_v:
                max_v, max_i = h, j
            if l < min_v:
                min_v, min_i = l, j
        out[i] = 1 if max_i < min_i else 0
    return out


def build_labels(df, window=288):
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    w = window

    out = {}
    out["label_open_next"] = o.shift(-1).astype(float)
    out["label_close_288"] = c.shift(-w).astype(float)
    out["label_max_288"] = h.rolling(w, min_periods=w).max().shift(-w).astype(float)
    out["label_min_288"] = l.rolling(w, min_periods=w).min().shift(-w).astype(float)
    mbm = _max_before_min(h.to_numpy(np.float64), l.to_numpy(np.float64), w)
    out["label_max_before_min"] = pd.Series(mbm, index=df.index)

    df = df.copy()
    for k, v in out.items():
        df[k] = np.asarray(v)
    return df
