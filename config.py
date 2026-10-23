from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class Config:
    # ---------------- phase (phase 1 = research/CSV, phase 2 = live/Wallex) ----------------
    phase: str = "research"  # "research" | "live"
    direction: str = "long"  # long | short

    # ---------------- data ----------------
    dataset_dir: str = "dataset"
    candles_dir: str = "dataset/candles"  # raw OHLCV: {PAIR}_5m.csv (open,high,low,close,volume)
    feature_cache_dir: str = "dataset/features"  # cached built feature arrays (parquet)
    labels_dir: str = "dataset/labels"
    train_test_split: str = "2026-08-21"  # first TEST datetime boundary; earlier = train
    timeframe: str = "5m"  # builder resamples only if resample_to is set
    resample_to: Optional[str] = None  # e.g. "15m" to reduce candle count
    btc_pair: str = "BTCUSDT"  # market-context symbol
    symbols: List[str] = field(default_factory=lambda: ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT"])

    # ---------------- feature system (the only piece that changes between phases) ----------------
    feature_engine: str = "h111b"  # feature builder id (base + horizon block)
    horizons: List[int] = field(default_factory=lambda: [3, 12, 48, 288, 2016, 8640])
    # feature horizons in 5m bars: 15m/1h/4h/24h/7d/30d
    feature_subset: Optional[List[str]] = None  # None = all built features (the drop dial for train/test)

    # ---------------- optimization ----------------
    max_rules: int = 9
    min_rules: int = 2
    crossover_rate: float = 0.95
    mutation_rate: float = 0.25
    mutation_eta: int = 10
    pop_size: int = 200
    seed: int = 43
    max_generations: int = 200
    max_eval: int = 200_000
    min_trade: int = min_rules * 100
    min_trade_per_rule: int = 0
    max_open_exposure: int = 400
    max_open_positions: int = 40
    n_folds: int = 6
    pool_clusters: int = 32
    pool_ratio: float = 1.0
    pool_csv: Optional[str] = None
    seed_dir: Optional[str] = None

    # ---------------- miner / smoke ----------------
    miner_use_cache: bool = True
    quick_subsample: int = None  # set in --quick: tail bars per symbol subsample


def load_config(quick: bool = False, direction: str = None) -> Config:
    cfg = Config()
    if direction:
        cfg.direction = direction  # set BEFORE pool_csv/seed_dir are derived
    if quick:  # full pipeline smoke: target < 60s end-to-end
        cfg.quick_subsample = 8000  # tail bars (per symbol) to keep NSGA eval cheap
        cfg.pop_size = 64
        cfg.max_generations = 5
        cfg.max_eval = 500_000_000
        cfg.min_rules = 2
        cfg.min_trade = 10
        cfg.pool_clusters = 6
    cfg.pool_csv = cfg.pool_csv or f"results/pool/{cfg.direction}/all_symbols.csv"
    cfg.seed_dir = cfg.seed_dir or f"results/fits/{cfg.direction}/"
    return cfg
