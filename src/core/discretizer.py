from typing import Dict, List, Optional, Tuple, Literal
import numpy as np


from typing import Dict as _Dict, List as _List, Optional as _Optional, Literal as _Literal


# ---------------------------------------------------------------------------
# Adaptive per-feature discretization (2026-10 redesign).
#
# Bin COUNT and bin BOUNDARIES are separate decisions:
#   - Boundary generation stays train-only quantile binning (baseline).
#   - The bin count is chosen per feature from distribution shape, the expected
#     feature->outcome relationship shape and out-of-sample support, NOT from
#     feature importance alone. Refinement experiments may raise a feature's
#     count (up to 10) by extending ADAPTIVE_BIN_COUNTS after validation.
#
# Support floor: if quantile edges of the requested count coincide (small or
# degenerate distributions), the count is automatically reduced until
# (count - 1) unique interior edges exist; a feature that cannot support even
# 3 ordered bins collapses to one bin. No edges are ever silently duplicated.
# ---------------------------------------------------------------------------

# Ordered human-readable band labels for adaptive bin counts (2..10).
BAND_LABELS = {
    2:  ["Low", "High"],
    3:  ["Low", "Medium", "High"],
    4:  ["Very Low", "Low", "High", "Very High"],
    5:  ["Very Low", "Low", "Medium", "High", "Very High"],
    6:  ["Very Low", "Low", "Low-Mid", "Mid-High", "High", "Very High"],
    7:  ["Extremely Low", "Very Low", "Low", "Medium", "High", "Very High", "Extremely High"],
    8:  ["Extremely Low", "Very Low", "Low", "Low-Mid", "Mid-High", "High", "Very High", "Extremely High"],
    9:  ["Bottom", "Very Low", "Low", "Low-Mid", "Medium", "Mid-High", "High", "Very High", "Top"],
    10: ["Bottom", "Very Low", "Low", "Low-Mid", "Medium", "Mid-High", "High", "Very High", "Very High+", "Top"],
}

# Continuous-class default: signed features previously got 10 decile classes,
# which produced low-support partitions; the ordered default is now 5.
DEFAULT_CONTINUOUS_CLASSES = 5

MIN_CONTINUOUS_CLASSES = 3

# Per-feature initial class counts (rationale grid in the audit doc).
ADAPTIVE_BIN_COUNTS: Dict[str, int] = {
    # regime / threshold features
    "momentum_sign_agreement_12_48_288": 3,
    "range_compression_flag": 2,          # binary by construction
    "inside_bar_flag": 2,                 # binary by construction
    # horizon block
    "h_momz_15m": 4, "h_momz_1h": 4, "h_momz_4h": 4, "h_momz_24h": 4,
    "h_momz_7d": 4, "h_momz_30d": 4,
    "h_relvol_15m": 4, "h_relvol_1h": 4, "h_relvol_24h": 4, "h_relvol_30d": 4,
    "h_dd_15m": 4, "h_dd_1h": 4, "h_dd_4h": 4, "h_dd_24h": 4,
    "h_dd_7d": 4, "h_dd_30d": 4,
    "h_range_pos_15m": 4, "h_range_pos_7d": 4, "h_range_pos_30d": 4,
    "h_cumret_7d": 4, "h_cumret_30d": 4,
    # returns / trend
    "logret_3": 5, "logret_12": 5, "logret_48": 5, "logret_288": 4,
    "momentum_accel_12_48": 4,
    "ema_gap_atr_12_48": 5, "ema_gap_atr_144_288": 5, "macd_hist_atr": 5,
    "trend_efficiency_48": 4, "micro_efficiency_12": 4, "macro_efficiency_720": 4,
    "adx_14": 4, "aroon_oscillator_28": 4,
    # volatility
    "atr_pct_14": 5, "atr_ratio_14_288": 4, "realized_vol_48": 4,
    "realized_vol_288": 4, "range_expansion_12_48": 4, "bollinger_width_20": 4,
    "volatility_regime_percentile_288": 4,
    "macro_volatility_ratio_288_1440": 4,
    # price action
    "close_location_value": 4, "body_to_true_range": 4,
    "upper_wick_to_true_range": 4, "lower_wick_to_true_range": 4,
    "breakout_strength_48": 4, "breakdown_strength_48": 4,
    "close_location_mean_12": 4, "body_direction_run_12": 5,
    # volume
    "volume_zscore_48": 5, "volume_ratio_12_48": 4,
    "signed_volume_imbalance_48": 4, "volume_price_divergence_48": 4,
    # statistics
    "return_autocorr_48": 4, "return_skew_288": 4, "sign_flip_rate_48": 4,
    "downside_vol_ratio_48_288": 4,
    "drawdown_from_peak_288": 4, "distance_from_rolling_low_288": 4,
    # oscillators / multiscale
    "rsi_21": 5, "micro_rsi_3": 4,
    "macro_channel_pos_1440": 5, "macro_range_pct_rank_1440": 4,
    # flow
    "obv_zscore_288": 5, "ema_dist_atr_288": 5,
    # BTC context
    "btc_logret_12": 5, "btc_logret_48": 5, "btc_logret_288": 4,
    "btc_realized_vol_288": 4, "btc_volatility_ratio_48_288": 4,
    "symbol_relative_return_288": 4, "symbol_btc_corr_288": 4,
}


class _ModeConfig:
    pass


MODE_CONFIG = {
    "binary":          {"num_classes": 2, "values": ["Inactive (0)", "Active (1)"], "dont_care": 2},
    "ternary":         {"num_classes": 3, "values": ["Negative (-1)", "Neutral (0)", "Positive (1)"], "dont_care": 3},
    "positive":        {"num_classes": 5, "values": ["Very Low", "Low", "Medium", "High", "Very High"], "dont_care": 5},
    "sparse_positive": {"num_classes": 5, "values": ["Very Low", "Low", "Medium", "High", "Very High"], "dont_care": 5},
    "signed":          {"num_classes": 10, "values": ["Extreme Bearish", "Strong Bearish", "Bearish", "Weak Bearish", "Neutral Negative", "Neutral Positive", "Weak Bullish", "Bullish", "Strong Bullish", "Extreme Bullish"], "dont_care": 10},
    "sparse_signed":   {"num_classes": 5, "values": ["Strong Negative", "Weak Negative", "Exactly Zero", "Weak Positive", "Strong Positive"], "dont_care": 5},
}


class FeatureConfig:
    MODE_CONFIG = {
        "binary": {
            "num_classes": 2,
            "values": ["Inactive (0)", "Active (1)"],
            "dont_care": 2
        },
        "ternary": {
            "num_classes": 3,
            "values": ["Negative (-1)", "Neutral (0)", "Positive (1)"],
            "dont_care": 3
        },
        "positive": {
            "num_classes": 5,
            "values": ["Very Low", "Low", "Medium", "High", "Very High"],
            "dont_care": 5
        },
        "sparse_positive": {
            "num_classes": 5,
            "values": ["Very Low", "Low", "Medium", "High", "Very High"],
            "dont_care": 5
        },
        "signed": {
            "num_classes": 10,
            "values": [
                "Extreme Bearish", "Strong Bearish", "Bearish", "Weak Bearish", "Neutral Negative",
                "Neutral Positive", "Weak Bullish", "Bullish", "Strong Bullish", "Extreme Bullish"
            ],
            "dont_care": 10
        },
        "sparse_signed": {
            "num_classes": 5,
            "values": ["Strong Negative", "Weak Negative", "Exactly Zero", "Weak Positive", "Strong Positive"],
            "dont_care": 5
        }
    }

    def __init__(self):
        self.feature_info: List[Dict] = []
        self.atomic_conditions: List[Dict] = []

    def detect_feature_mode(self, series: np.ndarray) -> str:
        if series.dtype.kind in ('U', 'S', 'O'):
            unique_vals = np.unique(series)
            n_unique = len(unique_vals)
            if n_unique <= 2:
                return "binary"
            elif n_unique <= 3:
                return "ternary"
            else:
                return "positive"

        unique_vals = np.unique(series[np.isfinite(series)])
        n_unique = len(unique_vals)

        if n_unique <= 2 and np.all(np.isin(unique_vals, [0, 1])):
            return "binary"

        if n_unique <= 3 and np.all(np.isin(unique_vals, [-1, 0, 1])):
            return "ternary"

        zero_ratio = np.mean(series == 0)

        if np.min(series) < 0:
            return "sparse_signed" if zero_ratio > 0.3 else "signed"

        return "sparse_positive" if zero_ratio > 0.3 else "positive"

    def build_feature_info(self, data: Dict[str, np.ndarray]) -> List[Dict]:
        self.feature_info = []

        for col, arr in data.items():
            if col == "symbol":
                unique_vals = np.unique(arr)
                num_classes = len(unique_vals)
                self.feature_info.append({
                    "col": col,
                    "mode": "categorical",
                    "num_classes": num_classes,
                    "dont_care": num_classes,
                    "values": [str(v) for v in unique_vals]
                })
            else:
                mode = self.detect_feature_mode(arr)
                if mode in ("binary", "ternary"):
                    config = self.MODE_CONFIG[mode]
                    num_classes = config["num_classes"]
                    values = config["values"]
                else:
                    # adaptive, ordered quantile banding (3..10 supported)
                    num_classes = int(
                        ADAPTIVE_BIN_COUNTS.get(col, DEFAULT_CONTINUOUS_CLASSES))
                    num_classes = max(2, min(10, num_classes))
                    values = BAND_LABELS[num_classes]
                self.feature_info.append({
                    "col": col,
                    "mode": mode,
                    "num_classes": num_classes,
                    "dont_care": num_classes,
                    "values": values
                })

        return self.feature_info

    def build_atomic_conditions(self) -> List[Dict]:
        self.atomic_conditions = []

        for feat_idx, info in enumerate(self.feature_info):
            for val_idx, value_name in enumerate(info["values"]):
                self.atomic_conditions.append({
                    "id": len(self.atomic_conditions),
                    "feature_idx": feat_idx,
                    "feature_name": info["col"],
                    "value_code": val_idx,
                    "value_name": value_name,
                    "condition_string": f"[{info['col']}] IS {value_name}"
                })

        return self.atomic_conditions

    def decode_gene_value(self, gene: int, feature_info: Dict) -> str:
        mode = feature_info["mode"]

        if mode == "categorical":
            values = feature_info["values"]
            if 0 <= gene < len(values):
                return values[gene]
            return "Unknown"

        values = MODE_CONFIG[mode]["values"]
        if 0 <= gene < len(values):
            return values[gene]
        return "Unknown"

    def encode_value_to_gene(self, value_name: str, feature_info: Dict) -> int:
        mode = feature_info["mode"]

        if mode == "categorical":
            values = feature_info["values"]
            if value_name in values:
                return values.index(value_name)
            return feature_info["dont_care"]

        values = MODE_CONFIG[mode]["values"]
        if value_name in values:
            return values.index(value_name)
        return feature_info["dont_care"]




class FeatureDiscretizer:
    """Discretizes continuous features into categorical values with condition lookup table."""

    def __init__(self):
        self.config = FeatureConfig()
        self.feature_info: List[Dict] = []
        self.condition_lookup: List[Dict] = []
        self.bins: Dict[str, np.ndarray] = {}

    def fit(self, data: Dict[str, np.ndarray]) -> Tuple[List[Dict], List[Dict]]:
        """Fit discretizer and build condition lookup table."""
        self.feature_info = self.config.build_feature_info(data)
        self.condition_lookup = self._build_condition_lookup()
        self._build_bins(data)
        return self.feature_info, self.condition_lookup

    def _build_condition_lookup(self) -> List[Dict]:
        """Build lookup table of all possible conditions (feature, value pairs)."""
        condition_lookup = []
        condition_id = 0

        for feat_idx, info in enumerate(self.feature_info):
            feature_name = info["col"]
            values = info["values"]

            for val_idx, value_name in enumerate(values):
                condition_lookup.append({
                    "id": condition_id,
                    "feature_idx": feat_idx,
                    "feature_name": feature_name,
                    "value_code": val_idx,
                    "value_name": value_name,
                    "condition_string": f"[{feature_name}] IS {value_name}"
                })
                condition_id += 1

        return condition_lookup

    @staticmethod
    def _quantile_edges(finite: np.ndarray, num_classes: int) -> Optional[np.ndarray]:
        """Quantile edges with -inf/+inf outer bounds, or None if the
        distribution cannot support (num_classes - 1) unique interior edges."""
        bins = np.quantile(finite, np.linspace(0, 1, num_classes + 1))
        bins[0], bins[-1] = -np.inf, np.inf
        if np.allclose(bins[1:-1], bins[1]):
            return None
        return bins

    def _build_bins(self, data: Dict[str, np.ndarray]):
        """Build bin edges for continuous features (train data only).

        Support floor: the requested class count is reduced until the quantile
        edges are non-degenerate; features too constant for 3 ordered bins
        collapse to a single (-inf, inf) bin. Nothing is silently duplicated."""
        self.bins = {}

        for info in self.feature_info:
            col = info["col"]
            mode = info["mode"]
            num_classes = info["num_classes"]

            if mode in ["binary", "ternary", "categorical"]:
                continue

            arr = data[col]
            finite = arr[np.isfinite(arr)]

            if finite.size == 0:
                self.bins[col] = np.array([-np.inf, np.inf])
                continue

            bins = None
            for k in range(num_classes, MIN_CONTINUOUS_CLASSES - 1, -1):
                bins = self._quantile_edges(finite, k)
                if bins is not None:
                    if k != num_classes:
                        # keep feature_info consistent with the real partition
                        info["num_classes"] = k
                        info["dont_care"] = k
                        info["values"] = BAND_LABELS[k]
                        num_classes = k
                    break
                bins = None

            if bins is None:
                self.bins[col] = np.array([-np.inf, np.inf])
                info["num_classes"] = 1
                info["dont_care"] = 1
                info["values"] = ["Single"]
            else:
                self.bins[col] = bins

    def transform(self, data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Discretize data using fitted bins."""
        discretized = {}

        for info in self.feature_info:
            col = info["col"]
            mode = info["mode"]
            num_classes = info["num_classes"]
            arr = data[col].copy()

            if mode in ["binary", "ternary", "categorical"]:
                discretized[col] = arr.astype(int)
            else:
                bins = self.bins.get(col)
                if bins is None:
                    discretized[col] = np.zeros_like(arr, dtype=int)
                else:
                    discretized[col] = np.digitize(arr, bins[:-1]) - 1
                    discretized[col] = np.clip(discretized[col], 0, num_classes - 1)

        return discretized

    def fit_transform(self, data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Fit and transform in one call."""
        self.fit(data)
        return self.transform(data)


# ---- merged from legacy parser.py ----
def safe_profit_factor(gross_wins: float, gross_losses: float) -> float:
    if gross_losses <= 0 and gross_wins > 0:
        return 99.0
    if gross_losses <= 0:
        return 0.0
    return gross_wins / gross_losses


def normalize_direction(direction: Optional[str]) -> str:
    direction = "long" if direction is None else str(direction).strip().lower()
    if direction not in {"long", "short"}:
        raise ValueError("TRADE_DIRECTION must be either 'long' or 'short'.")
    return direction


def parse_condition(condition: str) -> tuple:
    if " IS " not in condition:
        raise ValueError(f"Invalid condition format: {condition}")
    feature_part, value_part = condition.split(" IS ", 1)
    feature_name = feature_part.strip("[]").strip()
    value_name = value_part.strip()
    return feature_name, value_name