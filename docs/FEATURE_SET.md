# Feature Set Documentation

Complete reference for the causal feature set used by the research pipeline
(rule mining + NSGA-II optimization) and consumed unchanged by the live trader.

Source of truth: `src/features/registry.py` (definitions),
`src/features/builder.py` (pipeline), `src/features/labels.py` (labels),
`scripts/prune_features.py` (redundancy pruning).

---

## 1. Design principles

Every feature in the set obeys four rules:

1. **Causality** — a feature at decision time `t` is a pure function of
   completed candles at or before `t`. No shifting errors: rolling references
   to past extremes use `prev=True` (`p.rmax/p.rmin(..., prev=True)`) so the
   bar being classified is excluded from its own window. No feature touches
   the label window.
2. **Scale-free** — prices, distances, and volatilities are normalized
   (ATR-units, z-scores, rolling percentiles, range positions) so a value is
   comparable across symbols, price levels, and volatility regimes.
3. **Backward windows only** — every window is expressed in 5m bars and looks
   strictly backwards. Longest effective lookback ≈ 30 days.
4. **Safe missingness** — warm-up / mathematically undefined values are NaN;
   NaN is never treated as a signal. The builder maps NaN → 0.0 *after*
   train-only min/max fitting, i.e. "undefined" maps to the fitted minimum
   and the discretizer receives an explicit 0 code.

---

## 2. Pipeline overview

```
raw candles (CSVSource, 1m or 5m)
  → optional resample (cfg.resample_to; windows rescale via factor k)
  → labels (src/features/labels.py)
  → registry features (shared Primitives per symbol)
  → BTC market-context join (strict same-timestamp, completed bars only)
  → trim warm-up (max lookback) + rows with undefined labels
  → pool all symbols, sort by datetime
  → split at cfg.train_test_split
  → min/max normalization fitted on TRAIN only
  → drop redundant features (|Spearman| > 0.90, train-fitted)
  → dataset/features/{train,test}.{parquet,csv}
```

Outputs:

| File | Contents |
|---|---|
| `dataset/features/train.parquet` / `.csv` | normalized train rows (features + labels + meta) |
| `dataset/features/test.parquet` / `.csv` | normalized test rows |
| `dataset/features/pruning_report.csv` | which features were dropped and why |
| `dataset/features/normalization_report.csv` | train/test min-max per feature |
| `dataset/features/feature_scaler.pkl` | min/max artifact (train-only fit), reused for live parity |

Current state after pruning: **86 candidate features** out of ~122 raw
registry features (16 dropped in the earlier Pearson-0.95 pass, 20 in the
Spearman-0.90 pass). The kept list is in
`dataset/features/pruning_report.csv` (rows with `action=keep`).

---

## 3. Shared primitives

`registry.Primitives(o, h, l, c, v, k)` computes per-symbol building blocks
once, memoized, so each feature reuses them cheaply:

| Primitive | Definition |
|---|---|
| `p.w(n)` | window rescale: `round(k * n)` bars (k = timeframe scale factor) |
| `p.logret(n)` | `log(c_t / c_{t-n})` — n completed candles lookback |
| `p.ema(n)` | EMA of close with span n |
| `p.atr(n)` | simple mean of true-range over n bars (default 14) |
| `p.rv(n)` | realized volatility: rolling std of 1-bar log-returns × √n |
| `p.roll(n, kind, src, prev)` | rolling max/min/mean/std/sum of `o/h/l/c/v/tr/lr`; `prev=True` shifts by 1 bar |
| `p.rmax/rmin(src, n, prev)` | convenience wrappers for rolling extremes |
| `p.tr` | true range `max(h-l, |h-c_prev|, |l-c_prev|)` |

All rolling ops use `min_periods = window` (no partial-window noise), and
divisions guard against zero denominators with `EPS = 1e-12` (result NaN, not
inf).

---

## 4. Feature categories (post-pruning, 86 features)

### 4.1 returns — momentum of the close (11)

| Feature | Definition |
|---|---|
| `logret_{1,3,6,12,24,48,72,144,288,576}` | cumulative log-return over n bars (15m … 48h) |
| `momentum_accel_12_48` | `logret(12) − logret(48)·(12/48)` — short-vs-slow momentum slope |
| `momentum_sign_agreement_12_48_288` | mean of sign(logret) over 12/48/288 bars ∈ [−1, 1] |

### 4.2 trend (4)

| Feature | Definition |
|---|---|
| `ema_gap_atr_12_48` | `(ema(12) − ema(48)) / atr(14)` — fast/slow trend in ATR units |
| `ema_gap_atr_144_288` | same on 144/288 EMA pair — macro trend |
| `trend_efficiency_48` | Kaufman efficiency ratio over 48 bars: net move / path length |
| `trend_efficiency_288` | same over 288 bars |
| `macd_hist_atr` | MACD histogram (12/26/9) in ATR units |

### 4.3 volatility (7)

| Feature | Definition |
|---|---|
| `atr_pct_14`, `atr_pct_288` | ATR as fraction of price (short/long vol regime) |
| `atr_ratio_14_288` | short/long ATR ratio — vol expansion or contraction |
| `range_expansion_12_48` | mean true-range(12) / mean true-range(48) |
| `bollinger_percent_b_20` | %B: price position within 2σ Bollinger band ∈ (roughly) [0, 1] |
| `bollinger_width_20` | band width (4σ) / mean price — volatility regime |
| `volatility_regime_percentile_288` | percentile rank of rv(48) within its 288-bar history |

### 4.4 price_action — candle anatomy (8)

| Feature | Definition |
|---|---|
| `close_location_value` | `((c−l) − (h−c)) / (h−l)` — where the close sits in the bar ∈ [−1, 1] |
| `body_to_true_range` | |c − o| / true range — body dominance |
| `upper_wick_to_true_range` | rejection above the body |
| `lower_wick_to_true_range` | rejection below the body |
| `breakout_strength_48` | `(c − max_high(prev 48)) / atr(14)` — breakout up |
| `breakdown_strength_48` | `(min_low(prev 48) − c) / atr(14)` — breakdown down |
| `gap_atr` | bar-open gap vs previous close in ATR units |
| `close_location_mean_12` | 12-bar mean of `close_location_value` |

### 4.5 volume (7)

| Feature | Definition |
|---|---|
| `volume_zscore_48`, `volume_zscore_288` | (v − rolling mean) / rolling std — abnormal volume |
| `volume_ratio_12_48`, `volume_ratio_48_288` | short/long mean-volume ratio |
| `signed_volume_imbalance_48/288` | Σ(±v per candle direction) / Σv over window — directional flow; doji candles contribute 0 |
| `volume_price_divergence_48` | rolling corr of logret(1) with log(1+v) |

### 4.6 statistics — distributional shape (9)

| Feature | Definition |
|---|---|
| `return_autocorr_48/288` | lag-1 autocorrelation of 1-bar log-returns (mean reversion vs momentum) |
| `return_skew_288` | skewness of 1-bar log-returns |
| `return_kurtosis_288` | excess kurtosis — tail-risk regime |
| `sign_flip_rate_48` | fraction of direction flips — chop regime |
| `upside_vol_ratio_48_288` | upside semi-vol(48) / total vol(288) |
| `downside_vol_ratio_48_288` | downside semi-vol(48) / total vol(288) |
| `drawdown_from_peak_288` | c / rolling max(c, 288) − 1 |
| `distance_from_rolling_low_288` | c / rolling min(l, 288) − 1 |

### 4.7 oscillator (4)

Only robust, standard-scale survivors after pruning:

| Feature | Definition |
|---|---|
| `rsi_21` | Wilder-style RSI (as bounded up/down ratio ∈ [0, 1], not 0–100) |
| `aroon_oscillator_28` | (days since 28-bar high − days since 28-bar low) / 28 ∈ [−1, 1] |
| `adx_14` | ADX (Wilder-smoothed), trend *strength* regardless of direction |
| `obv_zscore_288` | OBV z-scored over its 288-bar rolling mean/std |

Removed at Spearman 0.90: `rsi_7/14`, `rsi_slope_3`, `rsi_position_vs_price`,
`stochastic_k/d_14`, `williams_r_14`, `cci_20`, `adx ±DI`, `di_net_14`,
`macd_rsi_cross_14`, `keltner_pos_20`, `ema_dist_atr_50` — all algebraic or
statistical near-duplicates of the above.

### 4.8 flow_structure (1)

| Feature | Definition |
|---|---|
| `ema_dist_atr_288` | `(c − ema(288)) / atr(14)` — distance from the macro mean in ATR units |

(`vwap_dist_atr_288` and `channel_pos_288` were dropped as redundant with this.)

### 4.9 multiscale — micro / macro mirrors (5)

Same quantities at extreme horizons so the miner separates intrabar noise
from multi-day regime:

| Feature | Definition |
|---|---|
| `micro_efficiency_12` | trend efficiency over 12 bars (≈ 1h) |
| `micro_rsi_3` | RSI over 3 bars |
| `macro_efficiency_720` | trend efficiency over 720 bars (≈ 2.5 days) |
| `macro_channel_pos_1440` | price position in the 1440-bar (≈ 5 day) high/low range |
| `macro_range_pct_rank_1440` | percentile rank of that position over the last 288 bars |
| `macro_volatility_ratio_288_1440` | rv(288) / rv(1440) — vol regime across scales |

### 4.10 horizon — configurable horizon block (~20)

For each configured horizon H (default: `3, 12, 48, 288, 2016, 8640` bars =
15m, 1h, 4h, 24h, 7d, 30d) up to 8 families are registered, **skipping any
family that already exists in the base registry at that exact scale**:

| Family | Definition |
|---|---|
| `h_cumret_{H}` | logret(H) |
| `h_momz_{H}` | z-score of logret(H) over an H-bar window |
| `h_vol_{H}` | realized vol over H bars |
| `h_atr_pct_{H}` | ATR(H) / price |
| `h_relvol_{H}` | z-score of log(1+v) over H bars |
| `h_range_pos_{H}` | price position in the H-bar high/low range |
| `h_dd_{H}` | drawdown from the H-bar rolling max |
| `h_trend_atr_{H}` | `(c − ema(H)) / atr(H)` — scale-free trend strength |

`register_horizon_features(horizons)` rebuilds this category on every build
and after every feature recompute, so the horizon list is config-driven
(`cfg.horizons`). The 2H warm-up (2 × widest horizon) is derived from this
list automatically.

### 4.11 BTC market context (7)

Computed from BTC candles only, joined to each symbol by **exact completed
timestamp** (never future BTC bars):

| Feature | Definition |
|---|---|
| `btc_logret_12/48/288` | BTC log-return at 1h / 4h / 24h horizons |
| `btc_realized_vol_288` | BTC realized vol over 24h |
| `btc_volatility_ratio_48_288` | BTC short/long vol ratio — market-wide stress |
| `symbol_relative_return_288` | `logret_288(symbol) − logret_288(BTC)` — alpha vs the market (exact 0 for BTC itself) |
| `symbol_btc_corr_288` | 288-bar rolling corr of the symbol's 1-bar returns with BTC's (exact 1 for BTC) |

(`btc_logret_1` is an internal input for the correlation and is dropped from
the candidate set.)

---

## 5. Labels

Computed in `src/features/labels.py` from the same candles, per symbol
(window = 288 bars = 24h on 5m):

| Label | Definition |
|---|---|
| `label_open_next` | next candle's open — the entry price used by backtest and live |
| `label_close_288` | close 288 bars later (time exit reference) |
| `label_max_288` | max of highs over bars t+1 … t+288 |
| `label_min_288` | min of lows over bars t+1 … t+288 |
| `label_max_before_min` | 1 if the window max occurred before the min (TP-first resolution for longs) |

The `max_before_min` kernel is numba-jitted and scans bars `i+1 … i+w` only:
the signal bar itself never arbitrates TP/SL ordering (this fixed an
off-by-one look-ahead bias in the legacy dataset — see labels.py:16-19).

Entry pricing is next-open: no feature and no label uses the decision bar's
own close as the fill price.

---

## 6. Discretization (how features become signals)

Features are consumed by the miner/live evaluator as **quantile codes**, not
raw floats:

- Train-only quantile bins (per feature) produce 6 ordered codes, e.g.
  `[rsi_21] IS Very Low / Low / Medium / High / Very High`, plus a dont-care
  token used by the genetic encoding.
- Bin edges are fitted on the TRAIN split only, then serialized via
  `src/report.py` into `fit_package.pkl` and restored by
  `src/live/fit_package.py` — research and live use **identical** edges.
- NaN → 0.0 happens after min/max fitting, so the undefined value maps to the
  fitted minimum bin consistently everywhere.

Consequence: the discretizer only cares about *rank/monotonic position*,
which is why Spearman (not Pearson) is the right pruning criterion.

---

## 7. Redundancy pruning

`scripts/prune_features.py`, fitted on TRAIN rows only:

1. **na_dominant** — > 50 % of train rows NaN → drop.
2. **near-constant** — < 5 distinct values or zero range → drop.
3. **redundant** — |Spearman correlation| > 0.90 between two features; from
   each redundant cluster the member with the lowest mean |Spearman| to the
   rest of the set is dropped iteratively (the most *central* feature is
   kept).

Spearman is used (rather than Pearson) because the downstream discretizer is
rank-based: two features with the same monotonic ordering carry the same
quantile information even if their Pearson correlation is modest.

Report: `dataset/features/pruning_report.csv` (feature, action, reason,
na_frac, nunique). Re-run this script after every feature rebuild; if you
change the threshold, edit `CORR_MAX` (and `CORR_METHOD`) at the top of the
script.

Current summary: 106 features entered the Spearman pass → **86 kept**.

---

## 8. Normalization

`builder._normalize`: min-max to [0, 1], **fitted on train only**
(`feature_scaler.pkl`). Test values may exceed [0, 1] — that is genuine
distribution shift and is deliberately left visible. NaN stays NaN through
normalization and is mapped to 0.0 only in the final injection step.

---

## 9. Live parity

The live evaluator (`src/live/evaluator.py`) recomputes features with the
same registry, the same `Primitives`/scale factor, the persisted bin edges
from `fit_package.pkl`, and the same BTC-context join. Rules referencing a
feature that is missing or not portable in the live build are skipped with a
logged warning (never silently mis-evaluated).

To change the feature set safely:

1. Edit `src/features/registry.py` (add/remove definitions).
2. Rebuild: `FeatureBuilder(load_config()).build_from_candles(use_cache=True)`
   with `force=True` as needed.
3. Re-run `scripts/prune_features.py`.
4. Re-run the research pipeline and regenerate `fit_package.pkl` before the
   live trader picks the new rules up.
5. Re-run shadow replay (`src/live/shadow_replay.py`) before trusting live.

---

## 10. Known gaps / future candidates

- **Order-book features** (depth imbalance, spread, depth pressure) — order
  book is fetched live but not used as a feature. Highest-potential addition
  for an illiquid venue.
- **Taker buy/sell flow** from klines (exchange-provided directional volume)
  as a cleaner alternative to candle-direction-signed imbalance.
- **Intraday seasonality** — time-of-day / time-of-week encodings.
- **Richer BTC-relative block** — BTC lead-lag terms, alt-vol-beta.
- `obv_zscore_288` uses cumsum OBV, which is anchor-dependent across a
  symbol's history; z-scoring mitigates but does not eliminate this —
  candidate for replacement by a rolling signed-volume measure.
