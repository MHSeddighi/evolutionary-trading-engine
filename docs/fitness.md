# Fitness & optimiser design (post-refactor)

Scope: `src/core/fitness.py`, `src/core/backtest.py` (attribution), `src/core/decoder.py`, `src/core/callbacks.py`, `main.py`.

## Objectives (pymoo receives `F = -score` for all three; pymoo minimises)

| Objective | Units | Direction (score) | Formula | Invalid / inactive behaviour |
|---|---|---|---|---|
| REXP — Return efficiency | % net return per trade, averaged on each trade's own notional | maximise | `mean(trade_pnls / trade_notional) * 100` | no trades, zero/negative notional, non-finite → `F = +1e6` |
| RWP — Risk-adjusted portfolio performance | % equity | maximise | `Rp − λ·Ulcer`; `Rp = (equity[-1]/initial_capital − 1)·100`; Ulcer = RMS of `%` drawdown vs running peak over the equity curve (starting equity prefixed) | no executed trades, equity ≤ min_position_notional (functional ruin), empty curve, non-finite → `F = +1e6` |
| ROB — Chronological robustness | % equity (worst fold) | maximise | `min_k(return_k)` over `n_folds` contiguous chronological folds, each simulated **independently** with identical execution rules (equity/exposure reset, no position carry). A fold with fewer executed trades than `min_fold_trades = max(2, min_trades // n_folds)` scores `min(return_k, 0)` (never positive). Zero executed trades overall → `F = +1e6` | under-traded folds score ≤ 0, so inactivity buys nothing |

`λ` = `rwdd_penalty` (constructor argument, default `0.5`): one Ulcer index point costs λ percentage points of return. Calibrate against representative strategies before changing; do not tune it against the final-test window.

Why REXP is per-trade on own notional instead of `sum(pnl)/sum(notional)`: compounded equities make notional paths quadratic in the sizing gene, so a portfolio-sum ratio is NOT exactly scale-invariant to `capital_pct`; the per-trade ratio is.

Sizing semantics: `capital_pct` remains an optimizable gene. It influences realized return THROUGH THE PORTFOLIO (exposure caps, min notional, ruin risk) but can no longer mechanically inflate objective values itself.

## Portfolio accounting convention (documented, deliberate)

`equity_curve` records equity at position **close events** only — marked at each closing event, including fees/slippage (all recorded pnl is net). Trades resolve TP/SL against a fixed horizon outcome, so intra-hold mark-to-market equity does not exist in this simulator; unrealized pnl of still-open positions is therefore not reflected until the release row (open positions are force-closed at end of data). Limitations are handled, not hidden: RWP/ROB treat ruin (equity ≤ min notional) as invalid, and identity `Σ trade_pnls = equity[-1] − initial_capital` holds exactly.

## Trade attribution semantics

Each executed entry is governed by exactly ONE rule (the first rule in decode order that covers its row uses its TP/SL/capital; simultaneous signals from other rules at the same row are rejected, they never execute as independent trades). `metrics["trade_rule_idx"]` maps every executed trade to its owning decoded-rule slot; `_con_min_trades_per_rule` counts EXECUTED trades per rule — raw signal-mask rows never count.

## Values seen during comparison on a representative stored strategy
(all results in this doc were produced on TRAIN-split data only; see `tests/compare_fitness.py`)

| Metric | OLD | NEW |
|---|---|---|
| REXP | `mean(pnl) − 0.1·std(pnl)` = **0.551 USD** (scale-dependent) | **2.479 %/trade** (size-invariant) |
| RWP | `0.5·mean + 0.3·p10 − 0.2·std` on stride-1 30-trade windows = **8.231 USD** (overlapping-window "robustness") | `Rp − 0.5·Ulcer` = **19.77 %** (Rp 19.94 %, ulcer 0.33) |
| ROB | *disabled* | worst of 4 chronological folds = **−0.34 %** — the overlapping-window score missed this entirely |
| DIV | count-based, rewarded ruleset bloat (3.00 for 3 rules) | diagnostics only, not an objective |

## Invalid genomes

Empty/undecodable/failing-sim genomes: `F = +1e6` on all objectives **and** `G = +1e6` on all constraints. All objective sentinels use the BAD convention (`BAD = -1e6` on the maximize-score axis, so `F` lands at worst `+1e6`); NaN/±inf are sanitized onto the violating side.

## Constraints & annealing schedule

`q = clipped(update_progress(progress))`; `progress = generation / n_max_gen` via `GenUpdater` (wired before every generation in `main.py`).

| Constraint | Start (q=0) | End (q=1) |
|---|---|---|
| min_rules | inactive | `len(decoded) >= min_rules` |
| min_trades | inactive | weakest rule has >= `min_trades_per_rule` EXECUTED trades |
| max_exposure | inactive | `max_total_open_exposure <= max_open_exposure` |
| rule_cap | inactive | `len(decoded) <= min_rules` (cap shrinks linearly from max_rules) |
| max_positions | inactive | `max_simultaneous_positions <= max_open_positions` |

Missing-callback fail-safe: before the first `update_progress()` the annealer holds the DOCUMENTED initial state (q=0, free exploration). If progress is never seen after `_MISSING_PROGRESS_WARN_EVALS = 2000` evaluations, constraints are forced to q=1 (strict) with a one-time loud warning — a missing callback cannot silently disable risk limits for the whole run.

## Folding policy & leakage boundaries

Folds are contiguous chronological row ranges. Entry signals use only discretized past-row features (no look-ahead). Documented caveat: TP/SL outcome labels near a fold edge were computed on the full per-symbol timeline, so an entry at a fold edge may resolve against price paths extending up to the horizon past the boundary. No portfolio state is carried across folds (each fold resets equity to initial capital). Random shuffling of time-series rows is never used.

## Decoder rejection policy (documented, silent)

Slot disabled / duplicate rule_id (first kept) / `tp <= sl` / final count < min_rules → the rule or the whole ruleset is dropped. The count of rules returned by `decode()` is the single source of truth; the rule-cap and min-trades constraints count exactly those rules, never the raw enabled genome slots.

## Diagnostics (report-only; never objectives)

`diversity_diagnostics()` reports unique features, max repeated feature, distinct TPs/SLs per decoded ruleset. Rule-level executed-trade attribution (`per-rule bincount`), exposure, max drawdown, ulcer and fold distributions are available from the simulator metrics. DIV is NOT an objective anymore; rule redundancy is controlled via `rule_cap`/`min_rules` annealing.

## Runtime

Full-sample single evaluation ≈ 1.3 s (mask construction dominates) + ≈ 15 ms/fold for the chronological folds (warm numba). The fold machinery removed the old always-on (but unused) `_fold_returns` computation. If evaluation cost becomes a problem, reduce `n_folds` in config or subsample — semantics unchanged.
