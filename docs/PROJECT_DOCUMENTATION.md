# Trading-bot — Phase 1 (research) documentation

Pooled multi-symbol rule mining + NSGA-II optimizer for crypto 5m trading.
Data contract is phase-2-ready: swap the CSV source for the Wallex DB source without touching the pipeline.

## Layout

```
config.py              everything configurable (symbols, split, dirs, quick-mode sizes)
main.py                entrypoint — pooled NSGA-II optimization (all symbols in ONE run)
src/
  data/                source.py (CSVSource / DatabaseSource stub), loader.py (assemble → arrays)
  features/            registry.py (56 base features), builder.py (BTC context, labels,
                       min-max normalization, train/test split), labels.py
  core/                discretizer.py, backtest.py (code-equality rule simulator),
                       fitness.py, decoder.py, callbacks.py, output.py
  initialization.py    RulePool + PoolBasedSampling + cold-start synthetic pool
  miner.py             XGBoost rule miner → results/pool/{direction}/all_symbols.csv
  report.py            test-set evaluation → results/fits/{direction}/test_eval.csv + strategy JSONs
  execution/wallex.py  phase-2 live-execution stub
scripts/download_data.py  Binance klines → dataset/candles/{PAIR}_5m.csv (resumable)
experiments/dev.ipynb  development notebook (loads main module directly, plots, sanity checks)
dataset/               candles/ , features/  (ONLY train.{csv,parquet} + test.{csv,parquet})
results/               pool/ , models/ , fits/ , logs/
```

## 1. Setup

```bash
cd trading-bot
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## 2. Download candles (phase 1 = CSV source)

```bash
.venv/bin/python scripts/download_data.py      # BTC, ETH, SOL, XRP, DOGE @ 5m, ~8 months, resumable
```

## 3. Build the dataset (features + labels, normalized, merged symbols)

**One-off bootstrap only — the research pipeline never touches `dataset/candles`.**
Everything downstream (miner, optimizer, report, notebook) reads ONLY
`dataset/features/train.{csv,parquet}` + `test.{csv,parquet}`, which carry the features,
the labels AND the raw `candle_*` OHLCV columns.

Explicit rebuild (e.g. after editing `src/features/registry.py`):

```bash
.venv/bin/python -c "
from config import load_config
from src.features.builder import FeatureBuilder
FeatureBuilder(load_config()).finalize(force=True)"
```

Without `force=True`, `finalize()` errors with instructions when the files are missing —
it never silently rebuilds from candles.

- All symbols merged side-by-side, sorted by `(datetime)`, split at `config.train_test_split`
  (2026-08-21 → 6mo train / 2mo test).
- All feature columns min-max normalized to **[0,1] fitted on the TRAIN split only**.
- If a feature set changes (edit `src/features/registry.py` or `builder.py`), delete
  `dataset/features/train.* test.*` and rebuild — the miner/optimizer always load these files.

## 4. Mine the rule pool (XGBoost)

```bash
.venv/bin/python -m src.miner --direction long     # full: 3000-tree forest (~10-40 min, cached model reused)
.venv/bin/python -m src.miner --direction short
MINER_FAST=1 .venv/bin/python -m src.miner --direction long   # 30s smoke (60 trees, loose filters)
MINER_DEBUG=1          # env flag: print per-rule rejection metrics
```

- Trains ONE pooled model over all symbols, extracts tree-path rules, backtests each against
  the TP/SL engine grid and keeps only rules that pass trade-count / PF / drawdown /
  environment-consistency filters.
- Output: `results/pool/{long,short}/all_symbols.csv` + model cache in `results/models/`.
- TP/SL grid constraint: 8 pairs = TP/SL **ratio {2, 3}** × **SL {0.5, 1.0, 1.5, 2.0} %**
  (see `TP_SL_COMBOS` in `src/miner.py`).
- If the pool already exists it is REPLACED. Delete `results/models/*f.pkl` to force retraining.

## 5. Optimize (NSGA-II)

```bash
.venv/bin/python main.py --direction long            # full run
.venv/bin/python main.py --quick                     # smoke: 5 gens, ~10 s
```

Cold start: if the pool has < 3 rules the optimizer pads itself with synthetic single-condition
rules, so it can run before ever mining. During the run it mixes pool rules with random
condition genes, evolving multi-rule rule-sets (tp/sl/cap are genes, too).

## 6. Evaluate on the test split

```bash
.venv/bin/python -m src.report --direction long
```

Simulates every pool rule on the 2-month unseen test portion and writes
`results/fits/{direction}/test_eval.csv` plus the top-N `strategy_*.json`
(this JSON format is what the phase-2 executor will consume).

## 7. Notebook (development)

```bash
.venv/bin/pip install notebook            # already installed in .venv
.venv/bin/python -m notebook experiments/dev.ipynb     # open from the project root
```

Cells: data sanity → label stats → discretizer introspection → rule pool view →
single-rule backtest → runs `main.main(quick=True)` in-process → test-eval tables + equity plot.

## 8. Config knobs (config.py)

| Key | Purpose |
|---|---|
| `symbols` | pooled symbols (BTC = market-context symbol, never excluded from trading) |
| `train_test_split` | first TEST datetime boundary |
| `quick` / `load_config(quick=True)` | small pop/gens/subsample for smoke tests |
| `resample_to` | optionally build features on a coarser tf (all windows scale by k) |
| `candles_dir` / `feature_cache_dir` / `results_dir` | data layout |

## 9. Phase-2 seams (not yet implemented)

1. `src/data/source.py:DatabaseSource` — implement `read_pair()` over the Wallex DB; the loader
   expects columns `datetime, open, high, low, close, volume` per pair.
2. `src/execution/wallex.py:WallexExecutor` — implement fetch/place/cancel; works off the
   `strategy_*.json` rule-sets (regularly re-evaluated by `src/live/*` when built).
3. Remember: research prices are Binance; Wallex prices differ (basis risk) — the feature
   discretization is scale-free after min-max but thresholds were fitted on Binance data.
