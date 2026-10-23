# AI Trading Bot for Wallex — Starkoach Agentic AI Hackathon

An automated trading engine that designs its own strategy with AI, then trades
it 24/7 on the Wallex exchange with strict risk control and a full decision log
for every trade.

- **Strategy discovery is AI** — an XGBoost model mines candidate rules, and a
  multi-objective genetic algorithm (NSGA-II) composes them into a portfolio of
  rules.
- **Execution is deterministic** — the live engine evaluates the learned rules
  every 5 minutes and places real market orders. This makes the system fully
  auditable and re-runnable by judges.

---

## 1. The big picture

We designed the system in two phases. Phase 1 (offline) *learns* the strategy.
Phase 2 (live) *executes* it. The two phases are connected by only two small
files, so judges can rerun each phase separately.

```
PHASE 1 — RESEARCH (Binance history)          PHASE 2 — LIVE (Wallex)
────────────────────────────────              ────────────────────────
download candles (5 symbols, 5m)              WallexClient (REST)
        ↓                                     LiveCandleStore (TimescaleDB)
build 86 causal features + labels             LiveEvaluator (features → rules)
        ↓                                     WallexExecutor (market orders)
XGBoost rule miner → candidate pool           LiveState (positions, restart-safe)
        ↓                                     JSONL logs (signals / trades / cycles)
NSGA-II optimizer → final ruleset
        ↓
ruleset.json + fit_package.pkl  ────────►     live engine reads these two files
```

The two artifacts that phase 1 hands to phase 2:

| File | What it is | Why it matters |
|---|---|---|
| `results/fits/{direction}/ruleset.json` | the deployed rules (conditions, TP, SL, position size) | the strategy itself |
| `results/fits/{direction}/fit_package.pkl` | quantile bin edges, fitted on **train data only** | guarantees live evaluates conditions on exactly the same statistical bands as backtest |

---

## 2. Data and features

- 5 symbols: BTC, ETH, SOL, XRP, DOGE — 5-minute candles.
- Dataset: **172,440 training rows** (Apr 22 → Aug 19, 2026) and
  **88,205 test rows** (Aug 21 → Oct 21, 2026). The test window is fully
  unseen by the optimizer.
- **86 features** in 11 families: returns, trend, volatility, candle anatomy,
  volume flow, distributional statistics, oscillators, multiscale mirrors,
  horizon features, and 7 BTC market-context features (e.g. how much the symbol
  outperformed BTC, its correlation with BTC).

Four rules that keep the features honest:

1. **Causality** — a feature at time `t` uses only completed candles up to `t`.
   No shifting errors, no future data.
2. **Entry price is the next bar's open** — never the decision bar's own close.
3. **All normalization and bin edges are fitted on train only.**
4. **Undefined values are NaN and are never treated as signals.**

**Labels** describe what really happens after a signal: take-profit hit, or
stop-loss hit, or 24-hour time exit — resolved intrabar with first-touch
ordering. This label kernel fixed a real look-ahead bias in our first version.

---

## 3. How the AI designs the strategy

### Step 1 — XGBoost rule mining

One XGBoost forest (2,000 trees) is trained over all symbols pooled together.
Each tree path is extracted as a human-readable candidate rule, e.g.:

> `[return_autocorr_48] IS High AND [rsi_21] IS Low`

Every candidate is then **backtested** on a grid of 8 TP/SL pairs and must pass
strict filters before entering the pool:

| Filter | Threshold |
|---|---|
| executed trades | ≥ 100 |
| profit factor | ≥ 1.0 |
| max drawdown | ≤ 15 % |
| PnL concentration (Gini) | ≤ 0.6 — rejects one-lucky-trade rules |

Result: ~1,139 long rules and ~5,716 short rules survive into the pool.

### Step 2 — NSGA-II multi-objective optimization

A genetic algorithm composes pool rules into a portfolio. Its genome encodes,
per rule: *enabled or not, which rule, TP level, SL level, position size* —
so even stops and sizing are learned, not hand-picked.

The fitness is a real portfolio backtest with fees, and it optimizes **three
objectives at once**:

| Objective | Meaning |
|---|---|
| **REXP** | average net return *per trade* (scale-invariant — sizing cannot game it) |
| **RWP** | portfolio return **minus 0.5 × Ulcer index** — rewards a smooth equity curve |
| **ROB** | return of the **worst** of 24 chronological folds — consistency, not average |

Five risk constraints start loose and tighten linearly while the run proceeds:
minimum rules, minimum executed trades per rule, max open exposure, max
concurrent positions, rule count cap.

**Why this counts as real AI:** discovery is machine-learned, composition is
an evolutionary optimizer with backtest-in-the-loop, and only the final
deterministic evaluation runs live — exactly what the reproducibility
requirement of the challenge wants.

---

## 4. When does the bot buy and sell?

A rule fires on the close of a 5-minute bar when **all** of its quantile-band
conditions match on the last completed candle. Example from our deployed
long ruleset:

```
RULE r3:
  [h_dd_7d]                IS Low    ← the symbol has had a shallow 7-day drawdown
  [h_cumret_30d]           IS High   ← strong 30-day momentum
  [symbol_btc_corr_288]    IS High   ← it moves with the market (BTC-led regime)

  action: BUY market order, size = 3% of free USDT
  take-profit: +5.0%      stop-loss: −2.3%      time exit: 24 hours
```

Exits are always enforced: intrabar TP/SL (first-touch), a 24h time exit, and
every missed bar during a connection outage is replayed intrabar so a stop can
never be skipped.

---

## 5. Risk and money management

Risk is enforced at **three** levels, not one:

| Level | Control |
|---|---|
| Optimization time | exposure / position / drawdown constraints inside the fitness |
| Selection time | hard gates before deployment: test PnL > 0, PF ≥ 1.10, DD ≤ 12%, ≥ 150 test trades, profitable in *both* train and test, features must be computable live |
| Run time (live) | per-rule position ≤ 3% of equity · total exposure ≤ 90% · mandatory TP/SL · 24h time exit · stale-feed trading ban · ruin detection |

---

## 6. Results

All numbers are **net of 0.2% fee + 0.05% slippage per side** (real Wallex
taker fee). No per-symbol tuning; symbols were pooled.

**Deployed long strategy** (selected among 10+ candidates, score 1.77):

| Split | Net PnL | Max DD | Trades | Win rate | Profit factor |
|---|---|---|---|---|---|
| Train (~4 months) | **+25.2 %** | 1.6 % | 874 | 67.7 % | 4.69 |
| Test (~2 months, unseen) | **+5.8 %** | 4.0 % | 908 | 53.9 % | **1.47** |

**Deployed short strategy** (score 2.43):

| Split | Net PnL | Max DD | Trades | Win rate | Profit factor |
|---|---|---|---|---|---|
| Train (~4 months) | **+21.0 %** | 0.6 % | 540 | 80.4 % | 7.58 |
| Test (~2 months, unseen) | **+2.5 %** | 1.3 % | 268 | 53.4 % | **1.63** |

How to read this honestly:

- Train→test PF degradation (4.7 → 1.5) is expected on unseen data with real
  fees — and both directions still stay clearly profitable out-of-sample with
  hundreds of executed trades. This is not one lucky trade.
- The optimizer's ROB objective forced deployed strategies to profit even in
  the *worst* chronological training fold.
- In live shadow mode over a small observed window (26 simulated trades), the
  long ruleset scored −0.2% with per-rule attribution logged. We treat such
  short windows as validation instruments, not verdicts — the deployment gate
  demands 100+ shadow trades before promotion.

---

## 7. Backtesting and shadow testing

The challenge explicitly rewards both — they are first-class CLI tiers here.

### Backtesting (historical simulation)

`src/core/backtest.py` replays rules against historical candles with:

- entry at the **next bar open** (what a real order would get),
- intrabar TP/SL with first-touch ordering, gap fills at open if a bar gaps,
- **real costs**: 0.2% fee + 0.05% slippage per side (0.5% round trip),
- portfolio-level accounting: exposure caps, sizing, min notional, ruin handling,
- 24 chronological folds for robustness — equity resets per fold, no position carry.

### Shadow testing (live-tier simulation, no capital)

Three tiers, safest first:

| Tier | What happens | Capital at risk |
|---|---|---|
| `--shadow-replay N` | replays the last N days of **cached real bars** through the fitted rules, fully simulated | none |
| `--shadow` | the real live loop runs every 5m bar, but orders are simulated | none |
| `--dry-run` | full order flow, paper fills inside the executor | none |

Shadow parity with reality: entry at next open, conservative SL-first tie-break,
next-open gap fills, fees + slippage. Every cycle writes per-rule stats —
trades, win rate, profit factor, equity — to
`results/logs/shadow_{direction}/shadow_stats.json`.

**Promotion discipline:** before real money, shadow per-rule stats must match
the phase-1 backtest expectations; if live PF/win-rate degrades more than ~30%
across 100+ shadow trades, paper→live promotion is blocked (market drift).

---

## 8. The live trading system (Phase 2)

`main_live.py` runs a **bar-aligned loop** that fires ~8 seconds after each
5-minute bar closes. One cycle does exactly this:

1. **Refresh** candle caches for all symbols (threaded REST requests,
   features+rule evaluation < 250 ms per symbol; cycle latency is logged).
2. **Evaluate** every rule on each symbol's last **completed** bar only.
3. **Manage open positions** — TP/SL checked against the latest bar's high/low;
   missed bars are replayed intrabar, 24h time exit enforced.
4. **Enter** — a MARKET order sized at the rule's `capital_pct` of free USDT,
   only if total exposure stays under 90% and the symbol feed is fresh.
5. **Log everything** — signals, trades, cycles — as append-only JSONL files.

Every component can fail and recover:

| Failure | System behavior |
|---|---|
| Wallex unreachable at startup | *degraded mode*: starts anyway, retries with backoff, no entries until connectivity returns |
| symbol feed stale > 3 bars | that symbol is **never traded** until fresh (reason logged) |
| exit order fails | position stays open and retry — never silently marked closed |
| process crash | state file is written atomically; restart resumes positions |
| CPU/DB down | falls back to the CSV candle cache and continues |

---

## 9. Decision log (why every trade happened)

Every signal writes a fully self-contained record to
`results/logs/signals.jsonl`: for **each condition** — the raw feature value,
its bin code, and the matched band — plus rule TP/SL/size and the chosen side.
`trades.jsonl` holds fills, exits and PnL; `cycles.jsonl` holds health metrics.
A judge can answer *"why did this trade happen?"* without asking us, and any
manual trade would be visible by its absence.

---

## 10. Setup

```bash
git clone <repo> && cd trading-bot
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env        # fill WALLEX_API_KEY (only needed for --live)
```

Optional candle DB (recommended for live): `docker compose up -d candles-db`
(Postgres on port 5434; the system falls back to CSV cache if it is down).

---

## 11. Running the live trading system

This is the exact sequence we run, from safe to real money. **Dry-run and
shadow are the defaults — real orders require the explicit `--live` flag AND a
loaded `WALLEX_API_KEY`.**

### Step 0 — export the fitted artifacts (after any phase-1 change)

```bash
.venv/bin/python scripts/package_fit.py --direction both
```

Produces `results/fits/{long,short}/ruleset.json` + `fit_package.pkl`.
If they already exist and phase-1 output has not changed, skip.

### Step 1 — offline shadow test (replays real recent bars, zero risk)

```bash
.venv/bin/python main_live.py --shadow-replay 14 --direction both
```

Prints and saves per-rule attribution for the last 14 simulated days
(`results/logs/shadow_both_stats.json`). Check the per-rule table before
continuing.

### Step 2 — live-loop shadow (the loop itself runs, orders simulated)

```bash
.venv/bin/python main_live.py --shadow --direction both
```

Runs forever, one cycle per 5m bar, both directions with isolated state/log
dirs (`results/logs/shadow_{direction}/`). Watch it for as long as practical;
then `Ctrl+C` to stop. Inspect the rolling stats:

```bash
.venv/bin/python main_live.py --report          # last 50 explained signals, JSON
cat results/logs/shadow_long/shadow_stats.json  # per-rule live stats
```

### Step 3 — paper cycle (dry-run, single bar)

```bash
.venv/bin/python main_live.py --dry-run --once --direction both
```

Prints one full cycle as JSON — symbol-by-symbol feature values, matched
conditions, sizing decisions, simulated fills. This is the fastest way to
*show* the system working.

### Step 4 — paper trading (continuous)

```bash
.venv/bin/python main_live.py --dry-run --direction both
```

Same engine as live, simulated fills. Stop with `Ctrl+C` (both directions stop
gracefully; positions are persisted in state files).

### Step 5 — real trading on the challenge account

```bash
WALLEX_API_KEY=... .venv/bin/python main_live.py --live --direction both
```

- Refuses to start without the API key; paper/shadow remain the defaults.
- Entries: MARKET BUY (long) or SELL (short) ~8 s after bar close.
- Exits: MARKET orders at TP/SL touch or 24h, checked every cycle; failed
  exits retry.
- Settlement parity with the challenge rules: all positions are force-closed
  on the 24h horizon, so the final account value is realized in Toman, not an
  unrealized coin balance.

Useful operational flags:

| Flag | Purpose |
|---|---|
| `--symbols BTCUSDT ETHUSDT` | restrict traded symbols (default: auto top-12 by daily volume) |
| `--max-cycles N` | stop after N bars (soak test) |
| `--poll 15` | loop poll seconds between bar checks |
| `--store csv` | CSV candle cache instead of the docker DB |
| `--state / --log-dir` | override state/log paths (used for isolated runs) |

### What to look at while it runs

- `results/logs/cycles.jsonl` — per-cycle latency and health.
- `results/logs/signals.jsonl` — every trigger, fully explained.
- `results/logs/trades.jsonl` — orders, fills, exits, PnL.
- `results/live_state_{direction}.json` — open positions (restart-safe).

---

## 12. Re-running the research phase (for judges)

```bash
.venv/bin/python scripts/download_data.py          # 1. candles
.venv/bin/python scripts/prune_features.py         # 2. feature pruning
.venv/bin/python -c "from config import load_config; \
from src.features.builder import FeatureBuilder; \
FeatureBuilder(load_config()).finalize(force=True)"   # 3. features+labels
.venv/bin/python -m src.miner --direction long     # 4. rule mining
.venv/bin/python main.py --direction long          # 5. NSGA-II search
.venv/bin/python -m src.report --direction both    # 6. test evaluation
.venv/bin/python scripts/select_strategy.py --direction both   # 7. gated selection
```

Smoke test without full data: `python main.py --quick` (5 generations, <10 s).

Tests: `.venv/bin/pytest tests/ -q` — covers fitness/trade accounting,
shadow TP/SL semantics, and outage/robustness behaviors.

---

## 13. Key talking points (for the demo)

1. **Two small files connect research to live** — the deployed strategy and the
   train-fitted bins are the entire contract, so live can never drift from
   backtest semantics.
2. **AI is the decision-maker, determinism is the executor.**
3. **Drawdown is penalized inside the optimizer** (Ulcer index), and deployment
   is gated on unseen-test profitability *and* drawdown — we trade consistency.
4. **Every trade has a reason** it in the JSONL logs — algorithmic-only trading
   is provable.
5. **Availability:** degraded mode, stale-feed ban, retrying exits, atomic state —
   the engine fails safe, never silent.

---

*Detailed references: `docs/FEATURE_SET.md` (features), `docs/fitness.md`
(optimizer internals), `docs/PHASE2.md` (live engine), `docs/DEPLOY_SERVER.md`
(server deployment).*
