"""Live signal evaluation: cached 5m candles -> features -> discretize -> rules.

Reuses the phase-1 feature registry and the persisted quantile bins so the
live feature codes are computed *identically* to the research pipeline:

    source.read_pair(pair)                      # 5m cache (WallexSource)
    -> registry.Primitives + FEATURES           # same 80 causal features
    -> BTC market-context attach                # same join, strict timestamps
    -> FeatureDiscretizer.transform             # persisted bin edges from phase 1
    -> eval ONLY the last CLOSED bar            # no lookahead, ever
    -> LiveRuleEvaluator.evaluate_row           # triggered ruleset rules

Partial (in-progress) 5m buckets are excluded from evaluation; they only
appear in the candle cache once fully closed.
"""
import logging
import time
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from src.features import registry
from src.features.registry import FEATURES
from src.live.fit_package import LiveRuleEvaluator

logger = logging.getLogger(__name__)

BAR_SECONDS = 300


def default_warmup_bars(horizons=None) -> int:
    """Minimum cached bars for every feature to be defined at the decision bar.

    The widest feature is the 30d momentum z-score: a z-score of H-bar returns
    over a window of H bars needs 2*H bars of history (H = 8640 five-minute
    bars). Add one 24h label-window worth of headroom so context joins and
    rolling corr/vol features are always defined.
    """
    if horizons is None:
        try:
            from config import load_config
            horizons = load_config().horizons
        except Exception:
            horizons = [8640]
    return int(np.ceil(2 * max(horizons))) + 288


def _btc_config() -> Tuple[list, list]:
    """Mirror the phase-1 builder's BTC-context definition dynamically so the
    live feature set always matches whatever phase 1 currently fits."""
    from src.features.builder import BTC_CONTEXT, BTC_STATIC
    return list(BTC_CONTEXT.items()), list(BTC_STATIC)


def btc_context_derived(p) -> Dict[str, pd.Series]:
    """BTC static context features — computed exactly like
    src/features/builder.py does (rv(288) + rv(48)/rv(288) ratio)."""
    eps = getattr(registry, "EPS", 1e-12)
    ctx = {"btc_realized_vol_288": p.rv(288)}
    ctx["btc_volatility_ratio_48_288"] = \
        p.rv(48) / p.rv(288).where(p.rv(288) > eps)
    return ctx


def computable_feature_names() -> Set[str]:
    """All feature names live evaluation can produce (registry + BTC context)."""
    from src.features.builder import BTC_CONTEXT, BTC_STATIC
    names = set(FEATURES) - {"symbol"}
    names |= set(BTC_CONTEXT) | set(BTC_STATIC)
    names |= {"symbol_relative_return_288", "symbol_btc_corr_288"}
    return names


def _last_closed_candle_time(now: Optional[float] = None) -> pd.Timestamp:
    """Start time of the most recently CLOSED 5m bucket (UTC)."""
    now = now or time.time()
    return pd.Timestamp((int(now) // BAR_SECONDS * BAR_SECONDS) - BAR_SECONDS,
                        unit="s")


class LiveEvaluator:
    """Per-symbol feature+rule evaluation driven by the candle cache."""

    def __init__(self, disc, source, btc_pair: str = "BTCUSDT",
                 require_bars: Optional[int] = None):
        self.disc = disc
        self.source = source
        self.btc_pair = btc_pair
        # Register the horizon block BEFORE anything asks for feature names or
        # computes features. Research registers it in FeatureBuilder.__init__;
        # live must do the same or every h_* condition silently disappears.
        try:
            from config import load_config
            self.cfg = load_config()
        except Exception:
            self.cfg = None
        horizons = getattr(self.cfg, "horizons", None) or [8640]
        registry.register_horizon_features(horizons)
        self.require_bars = int(require_bars or default_warmup_bars(horizons))
        self.rule_engine: Optional[LiveRuleEvaluator] = None
        self._skipped_features: Set[str] = set()

    def set_rules(self, ruleset: Dict):
        registry.register_horizon_features(
            getattr(self.cfg, "horizons", None) or [8640])
        computable = computable_feature_names()
        portable = [r for r in ruleset["rules"] if not
                    any(c.split("]", 1)[0].strip("[]") == "symbol"
                        for c in r["conditions"])]
        for rule in ruleset["rules"]:
            if rule not in portable:
                logger.warning("Rule skipped: symbol-conditional rules are "
                               "not portable to live symbols: %s",
                               rule["conditions"])
        portable = []
        for rule in ruleset["rules"]:
            missing = [c for c in rule["conditions"]
                       if c.split("]", 1)[0].strip("[]") not in computable]
            if missing:
                logger.warning("Rule skipped: features not computable live "
                               "%s: %s", missing, rule["conditions"])
                continue
            portable.append(rule)
        trimmed = {**ruleset, "rules": portable}
        self.rule_engine = LiveRuleEvaluator(self.disc, trimmed)
        for rule, reason in self.rule_engine.skipped_rules:
            logger.warning("Rule skipped for live deployment (%s): %s",
                           reason, rule.get("conditions"))
        logger.info("Rules portable to live: %d/%d",
                    len(self.rule_engine.active_rules), len(ruleset["rules"]))

    # ---------------- feature computation ----------------

    def _btc_context(self, btc_candles: pd.DataFrame) -> pd.DataFrame:
        """BTC context features indexed by datetime (mirrors research builder)."""
        btc_windows, _ = _btc_config()
        p = registry.Primitives(btc_candles["open"], btc_candles["high"],
                                btc_candles["low"], btc_candles["close"],
                                btc_candles["volume"], k=1)
        ctx = pd.DataFrame(index=btc_candles["datetime"])
        for name, w in btc_windows:
            ctx[name] = p.logret(w).to_numpy()
        for name, s in btc_context_derived(p).items():
            ctx[name] = s.to_numpy()
        ctx.index.name = "datetime"
        return ctx.reset_index()

    def _attach_btc(self, df: pd.DataFrame, ctx: pd.DataFrame,
                    pair: Optional[str] = None) -> pd.DataFrame:
        idx = df.set_index("datetime")
        j = idx.join(ctx.set_index("datetime"), how="left")
        if j["btc_logret_48"].isna().all():
            raise ValueError("BTC context join failed — timestamp grids mismatch")
        if pair == self.btc_pair:
            # Mirror the research builder exactly: BTC vs BTC is degenerate.
            j["symbol_relative_return_288"] = 0.0
            j["symbol_btc_corr_288"] = 1.0
            return j.reset_index().drop(columns=["btc_logret_1"])
        j["symbol_relative_return_288"] = j["logret_288"] - j["btc_logret_288"]
        p = registry.Primitives(j["open"], j["high"], j["low"],
                                j["close"], j["volume"], k=1)
        sym_lr1 = p.logret(1)
        j["symbol_btc_corr_288"] = sym_lr1.rolling(p.w(288),
                                                  min_periods=p.w(288)).corr(
            j["btc_logret_1"])
        return j.reset_index().drop(columns=["btc_logret_1"])

    def features_df(self, pair: str, tail_bars: Optional[int] = None) -> pd.DataFrame:
        """Full feature frame for cached CLOSE bars (partial bucket excluded).

        tail_bars lets the offline replay request enough rows to cover BOTH the
        feature warm-up and the replay window. Live evaluation uses the default
        (warm-up + 24h), which is all last-bar decisions need.
        """
        candles = self.source.read_pair(pair, refresh=False)
        if len(candles) < self.require_bars:
            raise ValueError(
                f"{pair}: need >= {self.require_bars} cached candles for the "
                f"30d feature warm-up, have {len(candles)} — run a backfill "
                "with a larger --warmup-bars")
        cutoff = _last_closed_candle_time()
        candles = candles[candles["datetime"] <= cutoff]
        lookback = int(tail_bars or (self.require_bars + 288))
        lookback = max(lookback, self.require_bars)
        if len(candles) < lookback:
            raise ValueError(
                f"{pair}: requested tail_bars={lookback} but only "
                f"{len(candles)} closed bars are cached (need warm-up + replay)")
        candles = candles.tail(lookback).reset_index(drop=True)
        p = registry.Primitives(candles["open"], candles["high"],
                                candles["low"], candles["close"],
                                candles["volume"], k=1)
        for name, (_, fn) in FEATURES.items():
            if name != "symbol":
                try:
                    candles[name] = fn(p)
                except Exception as e:
                    if name not in self._skipped_features:
                        logger.warning("Feature '%s' failed live computation: %s",
                                       name, e)
                        self._skipped_features.add(name)
        # BTC context is required for BTC itself too (btc_logret_* features),
        # not just the alt symbols.
        if pair == self.btc_pair:
            btc_candles = candles
        else:
            btc_candles = self.source.read_pair(self.btc_pair, refresh=False)
            btc_candles = btc_candles[btc_candles["datetime"] <= cutoff]
            if len(btc_candles) < lookback:
                raise ValueError(self.btc_pair + " cache too small for replay "
                                 f"lookback ({len(btc_candles)} < {lookback})")
            btc_candles = btc_candles.tail(lookback).reset_index(drop=True)
        candles = self._attach_btc(candles, self._btc_context(btc_candles), pair)
        return candles

    def codes_matrix(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Vectorized discretized codes for every row (uses persisted bins)."""
        out = {}
        for info in self.disc.feature_info:
            col = info["col"]
            if col == "symbol" or col not in df.columns:
                continue
            vals = pd.to_numeric(df[col], errors="coerce").to_numpy()
            bins = self.disc.bins.get(col)
            if bins is not None:
                codes = np.digitize(np.nan_to_num(vals, nan=0.0), bins[:-1]) - 1
                codes = np.clip(codes, 0, info["num_classes"] - 1)
                # undefined features FAIL CLOSED: code -1 means "no match"
                # (row_codes_row drops codes < 0 -> rules requiring the
                # feature cannot fire until the rolling windows are defined)
                codes[~np.isfinite(vals)] = -1
            else:
                codes = np.where(np.isfinite(vals), vals, -1).astype(int)
            out[col] = codes
        return out

    def row_codes_row(self, df: pd.DataFrame, i: int) -> Optional[Dict[str, int]]:
        if not hasattr(self, "_codes_cache") or self._codes_cache_df is not df:
            self._codes_cache = self.codes_matrix(df)
            self._codes_cache_df = df
        codes = {col: int(arr[i]) for col, arr in self._codes_cache.items()
                 if arr[i] >= 0}
        return codes or None

    def last_bar_codes(self, pair: str) -> Optional[
            Tuple[pd.Timestamp, Dict[str, int], float]]:
        """(bar_datetime, codes, close) for the last CLOSED bar; None if behind."""
        cutoff = _last_closed_candle_time()
        df = self.features_df(pair)
        row = df.iloc[-1]
        if pd.isna(row["datetime"]) or row["datetime"] > cutoff:
            return None  # latest cached bucket not closed yet
        codes = self.row_codes_row(df, len(df) - 1)
        if codes is None:
            return None
        return pd.Timestamp(row["datetime"]), codes, float(row["close"])

    # ---------------- signal emission ----------------

    def evaluate_frame(self, pair: str, df: pd.DataFrame) -> List[Dict]:
        """Trigger rules on the last row of an already-computed feature frame.

        Used by the two-direction shadow runner so the expensive feature
        computation is shared between the long and short evaluators.
        """
        if self.rule_engine is None:
            raise RuntimeError("LiveEvaluator.set_rules() not called")
        if df is None or df.empty:
            return []
        i = len(df) - 1
        codes = self.row_codes_row(df, i)
        if codes is None:
            return []
        row = df.iloc[i]
        triggered = self.rule_engine.evaluate_row(codes, pair)
        for t in triggered:
            t["symbol"] = pair
            t["bar_datetime"] = str(pd.Timestamp(row["datetime"]))
            t["close"] = float(row["close"])
        return triggered

    def evaluate_symbol(self, pair: str) -> List[Dict]:
        """Triggered rules (with explanations) for pair's last closed bar."""
        if self.rule_engine is None:
            raise RuntimeError("LiveEvaluator.set_rules() not called")
        bar = self.last_bar_codes(pair)
        if bar is None:
            return []
        bar_time, codes, close = bar
        triggered = self.rule_engine.evaluate_row(codes, pair)
        for t in triggered:
            t["symbol"] = pair
            t["bar_datetime"] = str(bar_time)
            t["close"] = close
        return triggered
