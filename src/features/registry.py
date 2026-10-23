"""Registry of causal candidate features (5m bars, per-symbol) — designed set.

Post-audit feature set: redundant horizons/oscillators/flow variants pruned,
two candle-structure features added (body_direction_run_12,
range_compression_flag). Binning uses adaptive per-feature class counts
(see src/core/discretizer.py ADAPTIVE_BIN_COUNTS).

Single source of truth. Every feature is a pure function of completed candles at
or before the decision timestamp t (causal). Shared primitives (log-returns,
true-range, EMAs, rolling helpers) are computed once per symbol to minimize work.

Windows are expressed in 5m candles. If the builder resamples to a larger
timeframe, a scale factor k multiplies every window so lookback duration stays
identical (~7 days max warm-up).

Feature functions take `Primitives` (p) and return a float Series aligned with
the symbol's candles. NaN = warm-up / undefined (documented downstream policy).
"""
import numpy as np
import pandas as pd

EPS = 1e-12
FEATURES: dict = {}          # name -> (category, fn)   # registry


def Primitives(o, h, l, c, v, k=1):
    """Shared per-symbol primitives. k = window scale factor."""
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    tr.iloc[0] = h.iloc[0] - l.iloc[0]
    logc = np.log(c)

    def w(n: int) -> int:
        return int(round(k * n))

    _lr: dict = {}
    _ema: dict = {}
    _atr: dict = {}

    def logret(n: int) -> pd.Series:
        """log(c_t / c_{t-n}) — n complete candles look-back."""
        n_ = w(n)
        return (_lr[n_] if n_ in _lr else _lr.setdefault(n_, logc - logc.shift(n_)))

    def ema(n: int) -> pd.Series:
        n_ = w(n)
        return _ema[n_] if n_ in _ema else _ema.setdefault(n_, c.ewm(span=n_, adjust=False).mean())

    def atr(n: int = 14) -> pd.Series:
        n_ = w(n)
        return _atr[n_] if n_ in _atr else _atr.setdefault(n_, tr.rolling(n_, min_periods=n_).mean())

    def roll(n: int, kind: str, src: str, prev=False):
        series = {"c": c, "h": h, "l": l, "o": o, "v": v, "tr": tr, "lr": logret(1)}[src]
        n_ = w(n)
        r = series.rolling(n_, min_periods=n_)
        out = getattr(r, kind)()
        return out.shift(1) if prev else out

    def rv(n: int) -> pd.Series:  # realized vol
        n_ = w(n)
        return logret(1).rolling(n_, min_periods=n_).std(ddof=1) * np.sqrt(n_)

    class P:
        pass

    p = P()
    p.o, p.h, p.l, p.c, p.v, p.tr, p.k = o, h, l, c, v, tr, k
    p.w, p.logret, p.ema, p.atr, p.roll, p.rv = w, logret, ema, atr, roll, rv
    p.rmax = lambda src, n, prev=False: roll(n, "max", src, prev)
    p.rmin = lambda src, n, prev=False: roll(n, "min", src, prev)
    return p


def _register(category: str, names_and_fns: dict):
    for name, fn in names_and_fns.items():
        FEATURES[name] = (category, fn)


# ------------------------------ returns ------------------------------
# Near-geometric ladder 3 -> 12 -> 48 -> 288 (15m -> 1h -> 4h -> 1d). The very
# short (1/6 bars) and intermediate (24/72/144/576) horizons were removed:
# highly overlapping with neighbours (Spearman > 0.9) and their incremental
# information is carried by the momentum z-scores and higher-order statistics.
# 7d/30d cumulative returns live in the horizon block (h_cumret_7d/30d).
_register("returns", {
    f"logret_{n}": (lambda p, _n=n: p.logret(_n))
    for n in (3, 12, 48, 288)
})


def _f_momentum_accel_12_48(p):
    return p.logret(12) - p.logret(48) * (12.0 / 48.0)


def _f_momentum_sign_agreement_12_48_288(p):
    return (np.sign(p.logret(12)) + np.sign(p.logret(48)) + np.sign(p.logret(288))) / 3.0


_register("returns", {
    "momentum_accel_12_48": _f_momentum_accel_12_48,
    "momentum_sign_agreement_12_48_288": _f_momentum_sign_agreement_12_48_288,
})

# ------------------------------ trend ------------------------------
_register("trend", {
    "ema_gap_atr_12_48": lambda p: (p.ema(12) - p.ema(48)) / p.atr(14),
    "ema_gap_atr_144_288": lambda p: (p.ema(144) - p.ema(288)) / p.atr(14),
    "macd_hist_atr": lambda p: (
        (dif := p.ema(12) - p.ema(26), (dif - dif.ewm(span=p.w(9), adjust=False).mean()) / p.atr(14))[1]
    ),
})


def _channel_pos(n):
    return lambda p: (lambda lo, hi: (p.c - lo) / (hi - lo).where((hi - lo) > 0))(p.rmin("l", n), p.rmax("h", n))


def _f_range_compression(p):
    """Binary: current normalized bandwidth in the bottom quintile of the
    trailing 288-bar bandwidth distribution (previous bars only — causal)."""
    bbw = (4 * p.roll(20, "std", "c")) / p.roll(20, "mean", "c").where(p.roll(20, "mean", "c").abs() > EPS)
    q20 = bbw.rolling(p.w(288), min_periods=p.w(288)).quantile(0.20).shift(1)
    return (bbw < q20).astype(np.int8).where(q20.notna())


def _trend_efficiency(n):
    def fn(p):
        move = (p.c - p.c.shift(p.w(n))).abs()
        path = p.c.diff().abs().rolling(p.w(n), min_periods=p.w(n)).sum()
        return move / path.where(path > 0)
    return fn


# Efficiency ladder: micro (12) lives in multiscale, macro (720) in multiscale.
# The 288-bar variant sits between 48 and 720 with little unique information.
_register("trend", {
    "trend_efficiency_48": _trend_efficiency(48),
})

# ------------------------------ volatility ------------------------------
# atr_pct_288 / realized_vol overlap heavily; realized vol is the representative.
# gap_atr removed: 5m crypto almost never gaps (near-constant feature).
# bollinger_percent_b removed: duplicates range-position (channel/RSI family).
# range_compression_flag ADDed: genuine volatility-structure information.
_register("volatility", {
    "atr_pct_14": lambda p: p.atr(14) / p.c,
    "atr_ratio_14_288": lambda p: p.atr(14) / p.atr(288).where(p.atr(288) > EPS),
    **{f"realized_vol_{n}": (lambda p, _n=n: p.rv(_n)) for n in (48, 288)},
    "range_expansion_12_48": lambda p: p.roll(12, "mean", "tr") / p.roll(48, "mean", "tr").where(p.roll(48, "mean", "tr") > EPS),
    "bollinger_width_20": lambda p: (4 * p.roll(20, "std", "c")) / p.roll(20, "mean", "c").where(p.roll(20, "mean", "c").abs() > EPS),
    "volatility_regime_percentile_288": lambda p: p.rv(48).rolling(p.w(288), min_periods=p.w(288)).rank(pct=True),
    "range_compression_flag": _f_range_compression,
})

# ------------------------------ price_action ------------------------------
_register("price_action", {
    "close_location_value": lambda p: ((p.c - p.l) - (p.h - p.c)) / (p.h - p.l).where((p.h - p.l) > 0),
    "body_to_true_range": lambda p: (p.c - p.o).abs() / p.tr.where(p.tr > EPS),
    "upper_wick_to_true_range": lambda p: (p.h - pd.concat([p.o, p.c], axis=1).max(axis=1)) / p.tr.where(p.tr > EPS),
    "lower_wick_to_true_range": lambda p: (pd.concat([p.o, p.c], axis=1).min(axis=1) - p.l) / p.tr.where(p.tr > EPS),
    "breakout_strength_48": lambda p: (p.c - p.rmax("h", 48, prev=True)) / p.atr(14),
    "breakdown_strength_48": lambda p: (p.rmin("l", 48, prev=True) - p.c) / p.atr(14),
    "inside_bar_flag": lambda p: ((p.h <= p.h.shift(1)) & (p.l >= p.l.shift(1))).astype(np.int8),
    "close_location_mean_12": lambda p: FEATURES["close_location_value"][1](p).rolling(p.w(12), min_periods=p.w(12)).mean(),
    # ADD: signed net directional-candle count over 12 bars. Captures
    # continuation / exhaustion sequences that single-candle ratios miss.
    "body_direction_run_12": lambda p: np.sign(p.c - p.o).fillna(0.0).rolling(p.w(12), min_periods=p.w(12)).sum(),
})

# ------------------------------ volume ------------------------------
_register("volume", {
    "volume_zscore_48": (
        lambda p: (p.v - p.roll(48, "mean", "v")) / p.roll(48, "std", "v").where(p.roll(48, "std", "v") > EPS)
    ),
    "volume_ratio_12_48": (
        lambda p: p.roll(12, "mean", "v") / p.roll(48, "mean", "v").where(p.roll(48, "mean", "v") > EPS)
    ),
    "signed_volume_imbalance_48": lambda p: _signed_vol_imbalance_fb(p, 48),
    "volume_price_divergence_48": lambda p: p.logret(1).rolling(p.w(48), min_periods=p.w(48)).corr(np.log1p(p.v)),
})


def _signed_vol_imbalance_fb(p, n):
    dirn = np.sign(p.c - p.o)
    # flat (doji) candles contribute 0 volume weight
    sgn_vol = dirn * p.v
    num = sgn_vol.rolling(p.w(n), min_periods=p.w(n)).sum()
    den = p.roll(n, "sum", "v")
    return num / den.where((den.abs()) > EPS)


# ------------------------------ statistics ------------------------------
_register("statistics", {
    "return_autocorr_48": lambda p: (
        p.logret(1).rolling(p.w(48), min_periods=p.w(48)).corr(p.logret(1).shift(1))
    ),
    "return_skew_288": lambda p: p.logret(1).rolling(p.w(288), min_periods=p.w(288)).skew(),
    "sign_flip_rate_48": lambda p: (
        (np.sign(p.logret(1)) != np.sign(p.logret(1)).shift(1)).astype(np.float64)
        .rolling(p.w(48), min_periods=p.w(48)).mean()
    ),
    # only the downside semi-volatility ratio is kept: up/down ratios are near
    # mirror images, so keeping both encodes the same information twice.
    "downside_vol_ratio_48_288": lambda p: _semivol_ratio(p, "down"),
    "drawdown_from_peak_288": lambda p: p.c / p.roll(288, "max", "c").where(p.roll(288, "max", "c") > EPS) - 1.0,
    "distance_from_rolling_low_288": lambda p: p.c / p.roll(288, "min", "l").where(p.roll(288, "min", "l") > EPS) - 1.0,
})


def _semivol_ratio(p, side):
    sel = p.logret(1).where(p.logret(1) > 0, 0.0) if side == "up" \
        else p.logret(1).where(p.logret(1) < 0, 0.0)
    short = sel.rolling(p.w(48), min_periods=p.w(48)).std(ddof=1)
    return short / p.rv(288).where(p.rv(288) > EPS)


# ------------------------------ oscillators (momentum) ------------------------------
# Professional oscillator suite on multiple scales: fast (7), standard (14/21),
# macro smooth (21). Wilder-smoothed where the source indicator defines it.

def _f_rsi(n):
    def fn(p):
        d = p.c.diff()
        up = d.clip(lower=0.0).ewm(alpha=1.0 / p.w(n), adjust=False).mean()
        dn = (-d).clip(lower=0.0).ewm(alpha=1.0 / p.w(n), adjust=False).mean()
        return up / (up + dn).where((up + dn) > EPS)
    return fn


def _f_aroon(n):
    def fn(p):
        n_ = p.w(n)
        days_up = p.c.rolling(n_, min_periods=n_).apply(lambda x: n_ - 1 - int(np.argmax(x)), raw=True)
        days_dn = p.c.rolling(n_, min_periods=n_).apply(lambda x: n_ - 1 - int(np.argmin(x)), raw=True)
        return (days_up - days_dn) / n_
    return fn


def _f_adx_name(kind):
    """kind in ('adx','pdi','mdi')."""
    def fn(p):
        n = 14
        n_ = p.w(n)
        up = p.h.diff()
        dn = -p.l.diff()
        plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
        minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
        atr_ = p.tr.ewm(alpha=1.0 / n_, adjust=False).mean() \
            .where(p.tr > EPS, np.nan)
        pdi = 100.0 * pd.Series(plus_dm, index=p.c.index).ewm(alpha=1.0 / n_, adjust=False).mean() / atr_
        mdi = 100.0 * pd.Series(minus_dm, index=p.c.index).ewm(alpha=1.0 / n_, adjust=False).mean() / atr_
        if kind == "pdi":
            return pdi
        if kind == "mdi":
            return mdi
        dx = (pdi - mdi).abs() / (pdi + mdi).where((pdi + mdi) > EPS)
        return dx.ewm(alpha=1.0 / n_, adjust=False).mean()
    return fn


# Redundancy audit: stochastics / Williams %R / CCI duplicate range-position
# (channel_pos / close_location family); DI pairwise splits duplicate di_net;
# short- and macro-window RSIs duplicate the kept 14/21 and momentum z-scores.
_register("oscillator", {
    "rsi_21": _f_rsi(21),
    "adx_14": _f_adx_name("adx"),
    "aroon_oscillator_28": _f_aroon(28),
})


# ------------------------------ flow / structure (pro set-ups) ------------------------------
def _f_obv_zscore_288(p):
    obv = (np.sign(p.c.diff()).fillna(0.0) * p.v).cumsum()
    m = obv.rolling(p.w(288), min_periods=p.w(288)).mean()
    s = obv.rolling(p.w(288), min_periods=p.w(288)).std(ddof=1)
    return (obv - m) / s.where(s > EPS)


# vwap_dist / keltner_pos / ema_dist_atr_50 removed (redundant with the kept
# macro range-position and ema_dist_atr_288; all correlate > 0.9 on train).
_register("flow_structure", {
    "obv_zscore_288": _f_obv_zscore_288,
    "ema_dist_atr_288": lambda p: (p.c - p.ema(288)) / p.atr(14),
})

# ------------------------------ multi-scale (micro / macro) ------------------------------
# Same signals on very different horizons so the miner separates intrabar vs
# multi-day quantiles: micro = 1-24 bars (minutes-hours), macro = 720-1440 bars (2-5 days).

_register("multiscale", {
    # micro scale
    "micro_efficiency_12": _trend_efficiency(12),
    "micro_rsi_3": _f_rsi(3),
    # macro scale (2-5 day horizons)
    "macro_efficiency_720": _trend_efficiency(720),
    "macro_channel_pos_1440": _channel_pos(1440),
    "macro_range_pct_rank_1440": lambda p: _channel_pos(1440)(p).rolling(p.w(288), min_periods=p.w(288)).rank(pct=True),
    "macro_volatility_ratio_288_1440": lambda p: p.rv(288) / p.rv(1440).where(p.rv(1440) > EPS),
})


def _zscore_over(series: pd.Series, n: int):
    m = series.rolling(n, min_periods=n).mean()
    s = series.rolling(n, min_periods=n).std(ddof=1)
    return (series - m) / s.where(s > EPS)


# ------------------------------ multi-horizon block ------------------------------
# Simplified per the 2026-10 redundancy audit: the previous 8-family x 6-horizon
# block re-implemented base features at other windows and dominated the prune
# drop-list. Now each horizon contributes at most three families that are NOT
# covered by the base registry at that scale (skip table below). Config-driven:
# horizons are supplied in config.py (5m bars; Primitives.w() rescales).
# All purely causal; NaN before warm-up (downstream maps NaN -> 0.0).

HORIZON_SUFFIX = {3: "15m", 12: "1h", 48: "4h", 288: "24h", 2016: "7d", 8640: "30d"}


def _horizon_suffix(n: int) -> str:
    return HORIZON_SUFFIX.get(n, f"{n}b")


def register_horizon_features(horizons):
    """(Re)build the 'horizon' category for the given 5m-bar horizons."""
    FEATURES = globals()["FEATURES"]
    for name in [n for n, (cat, _) in FEATURES.items() if cat == "horizon"]:
        del FEATURES[name]

    entries = {}

    # families already present in the (pruned) base registry at these scales
    SKIP = {
        "cumret":  {3, 12, 48, 288},          # logret_{n}
        "momz":    set(),                     # base momentum z-scores were pruned
        "relvol":  {48, 288, 2016},           # volume_zscore_{n} / h_relvol_30d
        "dd":      {288},                     # drawdown_from_peak_288 (rho=1.0)
        "rangpos": {12, 48, 288, 2016},       # channel-pos family (macro_* retained)
    }

    for n in horizons:
        s = _horizon_suffix(n)

        fn_cum = (lambda p, _n=n: p.logret(_n)) if n not in SKIP["cumret"] else None
        if fn_cum is not None:
            entries[f"h_cumret_{s}"] = ("horizon", fn_cum)

        fn_mom = (lambda p, _n=n: _zscore_over(p.logret(_n), _n))
        entries[f"h_momz_{s}"] = ("horizon", fn_mom)

        fn_rel = (lambda p, _n=n: _zscore_over(np.log1p(p.v), _n)) \
            if n not in SKIP["relvol"] else None
        if fn_rel is not None:
            entries[f"h_relvol_{s}"] = ("horizon", fn_rel)

        fn_rng = _channel_pos(n) if n not in SKIP["rangpos"] else None
        if fn_rng is not None:
            entries[f"h_range_pos_{s}"] = ("horizon", fn_rng)

        fn_dd = (lambda p, _n=n: p.c / p.roll(_n, "max", "c").where(p.roll(_n, "max", "c") > EPS) - 1.0) \
            if n not in SKIP["dd"] else None
        if fn_dd is not None:
            entries[f"h_dd_{s}"] = ("horizon", fn_dd)

    for name, (cat, fn) in entries.items():
        FEATURES[name] = (cat, fn)


_register("horizon", {})


# ------------------------------ statistics ------------------------------
def list_features():
    return sorted(FEATURES)
