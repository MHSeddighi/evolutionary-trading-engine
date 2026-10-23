"""Feature builder: raw candles -> 80 causal features + labels, pooled dataset.

Pipeline per symbol:
    load candles (CSVSource in phase 1 / DatabaseSource in phase 2)
    -> optional resample to a coarser timeframe (reduces candle count, windows scale)
    -> compute all registry features (shared primitives, per-symbol isolation)
    -> compute labels from the same candles
    -> drop warm-up bars (max lookback) and forward-undefined label rows
    -> add BTC market-context features aligned by timestamp (no future BTC candles)

Final outputs (all symbols merged side-by-side, sorted by datetime):
    dataset/features/train.csv + train.parquet
    dataset/features/test.csv  + test.parquet
    dataset/features/feature_scaler.pkl   (min/max fitted on TRAIN only, reused in phase 2)

Every feature column is normalized to the 0..1 range using min/max statistics
fitted on the TRAIN split only (test rows may deviate slightly beyond 1 — that
is real distribution shift and must stay visible, not hidden).
"""
import os
from typing import Dict, Optional

import numpy as np
import pandas as pd

from src.data.source import CSVSource, DatabaseSource, Source
from src.features import registry
from src.features.registry import FEATURES
from src.features.labels import build_labels

# warm-up: the widest horizon block needs 2H bars (H-bar signal standardized
# over an H-bar window) → 2 × 8640 (30d) = 17280 five-minute bars = 60 days.
# Computed from the active horizon list so it always covers every window.
DEFAULT_HORIZONS = [3, 12, 48, 288, 2016, 8640]   # 15m, 1h, 4h, 24h, 7d, 30d in 5m bars
BTC_CONTEXT = {
    "btc_logret_1": 1,
    "btc_logret_12": 12,
    "btc_logret_48": 48,
    "btc_logret_288": 288,
}
BTC_STATIC = ["btc_realized_vol_288", "btc_volatility_ratio_48_288"]


class FeatureBuilder:
    def __init__(self, cfg, source: Optional[Source] = None):
        self.cfg = cfg
        self.source = source or self._make_source()
        self.scale = self._timeframe_scale(cfg.timeframe, cfg.resample_to or cfg.timeframe)
        horizons = getattr(cfg, "horizons", DEFAULT_HORIZONS)
        # horizon registry must be (re)registered before every feature compute
        registry.register_horizon_features(horizons)
        # 2H bars of warm-up (H = widest horizon) covers every feature window
        self.max_window = int(np.ceil(2 * max(horizons) * self.scale))

    @staticmethod
    def _make_source():
        from config import load_config
        cfg = load_config()
        return CSVSource(cfg.candles_dir, cfg.timeframe)

    @staticmethod
    def _timeframe_scale(base: str, resample: str) -> int:
        def minutes(tf: str) -> int:
            unit, size = tf[-1], tf[:-1] or "1"
            return {"m": 1, "h": 60, "d": 1440}[unit] * max(1, int(size))
        scale = minutes(resample) // minutes(base)
        if scale <= 0:
            raise ValueError(f"resample_to must be >= {base}")
        return scale

    # ---------------- final outputs (train/test, csv+parquet) -----------------

    def finalize(self, force: bool = False, normalize: bool = True) -> Dict[str, pd.DataFrame]:
        """Build -> split -> purge -> (normalize) -> save csv+parquet for both splits.

        normalize=False returns RAW feature values (no min/max transform). The
        discretizer/live pipeline consumes raw values, so research fitting on
        raw scale keeps quantile bins comparable with live computed features.
        """
        out = self._feature_dir()
        suffix = "_norm" if normalize else ""
        paths = {s: os.path.join(out, f"{s}{suffix}.parquet") for s in ("train", "test")}
        if not force and all(os.path.exists(p) for p in paths.values()):
            print(f"   📦 Final feature files already present in {out}")
            return {s: pd.read_parquet(paths[s]) for s in paths}

        if not force:
            raise RuntimeError(
                f"Final feature files {paths['train']} / {paths['test']} are missing.\n"
                "The research pipeline ONLY reads these two files.\n"
                "To (re)create them explicitly run the one-off bootstrap:\n"
                "  .venv/bin/python scripts/download_data.py   # fetch candles\n"
                "  .venv/bin/python -c \"from config import load_config; from src.features.builder import FeatureBuilder; \\\n"
                "     FeatureBuilder(load_config()).build_from_candles(use_cache=True)\"\n"
                "Or pass force=True directly (it also targets the candle bootstrap).")
        print("   🏗️  building pooled features from raw candles (one-off bootstrap)...")
        pooled = self._build_pooled(use_cache=True)
        boundary = pd.Timestamp(self.cfg.train_test_split)
        train = pooled[pooled["datetime"] < boundary].reset_index(drop=True)
        test = pooled[pooled["datetime"] >= boundary].reset_index(drop=True)

        # Label-window purge: the 288-bar forward label of the last training bars
        # of each symbol is realised with TEST-period candles. Drop those rows
        # (plus a 1-bar embargo) so no training label touches the test period.
        purge_bars = 288 * max(1, self.scale)
        n_before = len(train)
        train = self._purge_tail_per_symbol(train, purge_bars)
        print(f"   🧹 Purged {n_before - len(train):,} train rows within "
              f"{purge_bars}-bar label window of the split boundary")

        print(f"   ✂️  split @ {boundary.date()}: train {len(train):,} | test {len(test):,}")

        feature_cols = [c for c in train.columns
                        if not c.startswith(("candle_", "label_")) and c not in ("datetime", "symbol")]
        if normalize:
            out_frames, artifact = self._normalize(train, test, feature_cols)
        else:
            out_frames = {"train": train, "test": test}
            artifact = {
                "feature_cols": feature_cols,
                "scale": "raw",
                "fit_on": "train-only",
                "boundary": self.cfg.train_test_split,
            }

        os.makedirs(out, exist_ok=True)
        for split in ("train", "test"):
            te = out_frames[split]
            te.to_parquet(paths[split])
            te.to_csv(paths[split].replace(".parquet", ".csv"), index=False)
            print(f"   💾 {split}: {len(te):,} rows -> {paths[split]} + .csv")
        return out_frames

    @staticmethod
    def _purge_tail_per_symbol(df: pd.DataFrame, bars: int) -> pd.DataFrame:
        """Drop the last `bars` rows of every symbol (order preserved)."""
        if bars <= 0 or df.empty:
            return df
        size = df.groupby("symbol")["symbol"].transform("size")
        cum = df.groupby("symbol").cumcount()
        return df[cum < (size - bars)].reset_index(drop=True)

    def _normalize(self, train, test, feature_cols):
        """Min-max to [0,1], fit on train only; NaNs are preserved as NaN."""
        t_start = train[feature_cols].to_numpy(dtype=np.float64)
        t_test = test[feature_cols].to_numpy(dtype=np.float64)
        with np.errstate(all="ignore"):  # all-NaN columns would warn in nan-range
            fmin = np.nanmin(t_start, axis=0)
            fmax = np.nanmax(t_start, axis=0)
        rng = np.where((fmax - fmin) > 0, fmax - fmin, 1.0)   # constant columns -> 0.0
        tr = (t_start - fmin) / rng
        te = (t_test - fmin) / rng
        out = {
            "train": self._inject(train, feature_cols, tr),
            "test": self._inject(test, feature_cols, te),
        }
        artifact = {
            "feature_cols": feature_cols,
            "min": {c: float(m) for c, m in zip(feature_cols, fmin)},
            "max": {c: float(m) for c, m in zip(feature_cols, fmax)},
            "fit_on": "train-only",
            "boundary": self.cfg.train_test_split,
        }
        report = pd.DataFrame({
            "feature": feature_cols,
            "train_min": fmin, "train_max": fmax,
            "test_min": np.nanmin(t_test, axis=0), "test_max": np.nanmax(t_test, axis=0),
        })
        report.to_csv(os.path.join(self._feature_dir(), "normalization_report.csv"), index=False)
        return out, artifact

    @staticmethod
    def _inject(df, feature_cols, values):
        df = df.copy()
        for i, c in enumerate(feature_cols):
            col = values[:, i]
            col[np.isnan(col)] = 0.0        # documented policy: warm-up/undefined -> 0
            df[c] = col.astype(np.float32)
        return df

    # ---------------- pooled build (internal cache) -----------------

    def build_raw_pooled(self, use_cache: bool = True) -> pd.DataFrame:
        """Public raw-scale pooled frame (no normalization, no file overwrite).

        Used by scripts/package_fit.py: the live discretizer must be fitted on
        the same RAW feature scale the live evaluator computes. The research
        train/test parquet files may carry normalized values, so reusing them
        would silently misalign live bin edges.
        """
        return self._build_pooled(use_cache=use_cache)

    @staticmethod
    def split_raw_pooled(pooled: pd.DataFrame, cfg) -> pd.DataFrame:
        train = pooled[pooled["datetime"] < pd.Timestamp(cfg.train_test_split)]
        train = train.reset_index(drop=True)
        return FeatureBuilder._purge_tail_per_symbol(train, 288)

    def _build_pooled(self, use_cache=True) -> pd.DataFrame:
        cache_path = self._raw_cache_path()
        if use_cache and os.path.exists(cache_path):
            print(f"   📦 Loaded raw features from cache: {cache_path}")
            return pd.read_parquet(cache_path)

        pairs = self.cfg.symbols
        btc_pair = self.cfg.btc_pair
        # BTC context must be computed on the FULL history so its own 7d warm-up
        # already happened long before the first trimmed row
        btc_full = self._symbol_features(btc_pair, trim=False)
        btc_feat = self._btc_context_source(btc_full)
        btc_df = self._trim(btc_full)

        parts = []
        for pair in pairs:
            print(f"   🧮 Building features: {pair}")
            # attach context BEFORE trimming so BTC-corr/28d windows keep their warm-up
            full = btc_full if pair == btc_pair else self._symbol_features(pair, trim=False)
            df = self._trim(self._attach_btc_context(full, btc_feat, pair))
            df["symbol"] = pair
            parts.append(df)

        pooled = pd.concat(parts, ignore_index=True)

        # rows preceding BTC's first valid context bar (e.g. symbols starting
        # later than BTC) get NaN context -> drop them, they are a handful
        n0 = len(pooled)
        pooled = pooled.dropna(subset=["btc_logret_288"]).reset_index(drop=True)
        if len(pooled) < n0:
            print(f"   🧹 Dropped {n0 - len(pooled):,} rows without BTC context")

        # rename raw candle columns -> feature columns that overlap nothing else
        for col in ("open", "high", "low", "close", "volume"):
            pooled = pooled.rename(columns={col: f"candle_{col}"})

        if use_cache:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            pooled.to_parquet(cache_path)
            print(f"   💾 Cached raw features: {cache_path}")
        return pooled

    def _symbol_features(self, pair: str, trim: bool = True) -> pd.DataFrame:
        candles = self.source.read_pair(pair)
        if self.scale > 1:
            candles = self._resample(candles)
        if len(candles) <= self.max_window:
            raise ValueError(f"{pair}: only {len(candles)} candles, need > warm-up {self.max_window}")

        candles = build_labels(candles.reset_index(drop=True))
        p = registry.Primitives(candles["open"], candles["high"], candles["low"],
                                candles["close"], candles["volume"], k=self.scale)
        for name, (_, fn) in registry.FEATURES.items():
            candles[name] = fn(p)
        if not trim:
            return candles.reset_index(drop=True)
        # drop warm-up bars where candles insufficient for any 7-day feature window
        return self._trim(candles)

    def _trim(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.iloc[self.max_window:].dropna(
            subset=["label_min_288", "label_max_288"]).reset_index(drop=True)

    def _resample(self, candles: pd.DataFrame) -> pd.DataFrame:
        candles = candles.set_index("datetime")
        agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
        out = candles.resample(self.cfg.resample_to).agg(agg).dropna().reset_index()
        return out

    # ---------------- BTC market context -----------------

    def _btc_context_source(self, btc_df: pd.DataFrame) -> pd.DataFrame:
        """BTC features indexed by datetime — reused for every other symbol."""
        p = registry.Primitives(btc_df["open"], btc_df["high"], btc_df["low"],
                                btc_df["close"], btc_df["volume"], k=self.scale)
        ctx = pd.DataFrame(index=btc_df["datetime"])
        for name, w in BTC_CONTEXT.items():
            ctx[name] = p.logret(w).to_numpy()
        ctx["btc_realized_vol_288"] = p.rv(288).to_numpy()
        ctx["btc_volatility_ratio_48_288"] = (p.rv(48) / p.rv(288).where(p.rv(288) > registry.EPS)).to_numpy()
        ctx.index.name = "datetime"
        return ctx.reset_index()

    def _attach_btc_context(self, df: pd.DataFrame, ctx: pd.DataFrame, pair: str) -> pd.DataFrame:
        idx = df.set_index("datetime")
        j = idx.join(ctx.set_index("datetime"), how="left")
        if j["btc_logret_48"].isna().all():
            raise ValueError("BTC context join failed — timestamp grids mismatch")
        # strict-timestamp join: only completed BTC candles at or before each decision bar

        j["symbol_relative_return_288"] = j["logret_288"] - j["btc_logret_288"]

        if pair == self.cfg.btc_pair:
            # BTC vs BTC: definitions are exact — relative return 0, correlation 1
            j["symbol_btc_corr_288"] = 1.0
            j["symbol_relative_return_288"] = 0.0
            return j.reset_index().drop(columns=["btc_logret_1"])
        p = registry.Primitives(j["open"], j["high"], j["low"], j["close"], j["volume"], k=self.scale)
        sym_lr1 = p.logret(1)
        j["symbol_btc_corr_288"] = sym_lr1.rolling(p.w(288), min_periods=p.w(288)).corr(j["btc_logret_1"])
        # btc_logret_1 is an internal input for symbol_btc_corr, not a candidate feature
        return j.reset_index().drop(columns=["btc_logret_1"])

    # ---------------- paths -----------------

    def _feature_dir(self) -> str:
        from config import load_config
        return load_config().feature_cache_dir

    def _raw_cache_path(self) -> str:
        from config import load_config
        cfg = load_config()
        tf = cfg.resample_to or cfg.timeframe
        subset = ",".join(sorted(self.cfg.symbols))
        return os.path.join(cfg.feature_cache_dir, f".raw_pooled_{cfg.feature_engine}_{tf}_{subset}.parquet")


def run(cfg, split: str = None, force: bool = False,
        normalize: bool = True) -> pd.DataFrame:
    """Public entrypoint: final split (features + labels + meta).

    split=None -> merged (train+test); 'train'/'test' -> the final files.
    normalize=False -> RAW feature values (live-parity scale for the discretizer)."""
    builder = FeatureBuilder(cfg)
    if split is None:
        norm = builder.finalize(force=force, normalize=normalize)
        merged = pd.concat([norm["train"], norm["test"]], ignore_index=True)
        return merged
    norm = builder.finalize(force=force, normalize=normalize)
    out = norm[split]
    print(f"   ✂️  split={split}: {len(out):,} rows (boundary {cfg.train_test_split})")
    return out
