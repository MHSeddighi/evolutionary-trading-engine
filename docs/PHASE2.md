# Phase 2 — Wallex Live Deployment

Phase 1 (pooled NSGA-II rule search) stays untouched. Phase 2 adds a live
deployment layer that takes the fitted strategy and trades it on Wallex.

## Pipeline

```
results/fits/{direction}/ruleset.json     your deployed rules (you write/refresh it)
results/fits/{direction}/fit_package.pkl  phase-1 quantile bins (scripts/package_fit.py)

main_live.py
  ├─ WallexClient        REST: markets / udf-history / balances / orders (X-API-Key)
  ├─ LiveCandleStore     candle storage: dockerized TimescaleDB (default) or CSV
  ├─ LiveEvaluator       registry features -> persisted bins -> rule compile + eval
  ├─ WallexExecutor      MARKET orders at bar close (paper fills when --dry-run)
  ├─ LiveState           results/live_state.json (positions, atomic writes)
  └─ logger              results/logs/{signals,trades,cycles}.jsonl
```

## Shadow testing / paper trading

Three evaluation tiers, safest first:

| tier | command | what happens |
|---|---|---|
| shadow (offline) | `python main_live.py --shadow-replay 14 --symbols BTCUSDT ETHUSDT` | replays the last N days of cached bars through the fitted rules: entry next bar open, intrabar TP/SL (conservative SL-first), 0.35% fee + 0.05% slippage; prints + saves per-rule attribution |
| shadow (live-loop) | `python main_live.py --shadow` | same simulation as real cycles fire — logs per-rule shadow stats every cycle |
| paper (dry-run) | `python main_live.py --dry-run` | full order flow simulated inside WallexExecutor (pool accounting), no risk |

Artifacts:
- `results/logs/shadow_signals.jsonl` — every evaluated signal, fully explained
- `results/logs/shadow_trades.jsonl` — every simulated fill (entry/exit, hold bars, pnl)
- `results/logs/shadow_stats.json` — rolling per-rule stats: trades, win rate,
  profit factor, avg pnl, avg hold, equity, total return

Validation discipline: shadow-replay before deploy; then compare live-shadow
per-rule stats against the phase-1 backtest expectation in `strategy_*.json`
(`results/fits/{direction}/`). If live PF/win-rate degrades materially (> ~30%)
after 100+ shadow trades, do NOT promote paper→live — the live market has
drifted from research conditions.

## Candle storage (docker DB)

```bash
docker compose up -d candles-db          # postgres:16-alpine on localhost:5434
```

- Table `candles` (symbol, datetime, ohlcv), bulk COPY-based upsert, idempotent.
- DSN configurable via env `WALLEX_DSN` (default `postgresql://trader:trader@localhost:5434/tradingbot`).
- **DB is the store of record**: `main_live.py --store db` (default). If the
  container is down the trader logs a warning and continues on the CSV cache.
- Existing CSV caches migrate in: `LiveCandleStore(...).csv_to_db()`
  (auto-run schema + import; also imports phase-1 `dataset/candles`).
- Every DB write mirrors back to `dataset/candles_live/*.csv` so both stores
  stay in sync and features/phases can switch freely.
- If your Docker Hub access never allows the timescale image, plain postgres
  works (code auto-detects; add the extension later without schema changes).


## Commands

```bash
# one-time (after any phase-1 fit change): export the discretizer bins
.venv/bin/python scripts/package_fit.py --direction long

# paper cycle (safe, keyless): backfills 7-day warmup, evaluates one 5m bar
.venv/bin/python main_live.py --dry-run --once --symbols BTCUSDT ETHUSDT

# continuous paper trading (fires ~8s after each bar close)
.venv/bin/python main_live.py --dry-run

# explained signals (per-condition raw value, bin code, matched band)
.venv/bin/python main_live.py --report

# real money (requires env key; refuses to start without it)
WALLEX_API_KEY=... .venv/bin/python main_live.py --live
```

## Ruleset JSON format

```json
{
  "direction": "long",
  "rules": [
    {
      "rule_id": 12, "name": "pullback_breakout",
      "conditions": "[btc_logret_12] IS Weak Bullish AND [symbol_relative_return_288] IS Bullish",
      "tp": 1.5, "sl": 1.0, "capital_pct": 5.0
    }
  ]
}
```

- `tp`/`sl` are percents of entry, `capital_pct` is percent of free USDT per entry.
- Rules referencing `[symbol]` or unknown features are skipped (reason logged) —
  research-side categorical symbol codes are not portable to other symbols.

## Latency budget (5m bar)

- Symbols limited by `--max-symbols` (default 12), auto-ranked by 24h USDT volume.
- Each cycle: ~1 REST call per symbol in 4 threads, feature+eval < 250 ms/symbol.
- Measured `cycle_ms` logged in `results/logs/cycles.jsonl` every run.

## Safety

- Dry-run is the default; fills are simulated locally with 0.35% taker fee.
- `--live` requires both `WALLEX_API_KEY` and explicit flag — fails loudly otherwise.
- TP/SL are checked every cycle against latest bar high/low; state survives restart.
