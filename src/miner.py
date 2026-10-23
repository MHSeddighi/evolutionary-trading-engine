#!/usr/bin/env python3
"""XGBoost rule miner (adapted from legacy scripts/xgb_miner.py).

Modified for the new system:
  - ALWAYS runs all symbols in ONE pass (no --symbol; data is pooled)
  - dataset paths come from config.py (CSV in phase 1, DB source in phase 2 — contract unchanged)
  - output rules go to results/pool/{direction}/all_symbols.csv (feeds the optimizer's RulePool)
"""
import argparse
import csv
import os
import pickle
import re
import sys
import time

import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import load_config
from src.core.discretizer import FeatureDiscretizer
from src.core.backtest import CapitalManagedTradeSimulator, build_rule_signal_mask
from src.core.discretizer import safe_profit_factor
from src.data.loader import assemble

# ---------------- fixed tuning (legacy values) ----------------
# TP/SL engine grid — user spec: 8 pairs built from TP/SL RATIO {2, 3} over SL {0.5, 1.0, 1.5, 2.0}
_SL_STEPS = (0.5, 1.0, 1.5, 2.0)
_RATIOS = (2.0, 3.0)
TP_SL_COMBOS = {
    d: [(round(r * s, 3), s, 5.0) for s in _SL_STEPS for r in _RATIOS]
    for d in ("long", "short")
}
MIN_TRADES = 100
MIN_PF = 1.0
MAX_DD = 15.0
TOP_K = 10000
MAX_GINI = 0.6
MIN_PNL = 0.0
N_ENVIRONMENTS = 2

_XGB_TREES = int(os.environ.get("XGB_TREES", "2000"))
_XGB_JOBS = int(os.environ.get("XGB_JOBS", min(12, os.cpu_count() or 4)))
XGB_PARAMS = {
    'n_estimators': _XGB_TREES, 'max_depth': 8, 'learning_rate': 0.05,
    'subsample': 0.8, 'colsample_bytree': 0.8, 'colsample_bylevel': 0.8, 'colsample_bynode': 0.8,
    'min_child_weight': 5, 'reg_alpha': 0.1, 'reg_lambda': 0.1, 'gamma': 0.1,
    'random_state': 42, 'n_jobs': _XGB_JOBS, 'verbosity': 1, 'tree_method': 'hist', 'max_bin': 256,
    'grow_policy': 'lossguide', 'max_leaves': 255,
}


if os.environ.get("MINER_FAST") == "1":   # smoke test: tiny forest + relaxed filters
    XGB_PARAMS.update(n_estimators=60, max_depth=6, verbosity=0)
    MIN_TRADES, MIN_PF, MIN_PNL, MAX_DD, MAX_GINI = 20, 0.6, -3.0, 30.0, 1.0


def gini_coefficient(pnls):
    if len(pnls) == 0:
        return 1.0
    positive_pnls = pnls[pnls > 0]
    if len(positive_pnls) == 0:
        return 1.0
    sorted_pnls = np.sort(positive_pnls)
    n = len(sorted_pnls)
    index = np.arange(1, n + 1)
    return (2 * np.sum(index * sorted_pnls)) / (n * np.sum(sorted_pnls)) - (n + 1) / n


def check_rule_passes(metrics):
    trades = metrics.get("executed_trades", 0)
    pf = metrics.get("profit_factor", 0)
    dd = metrics.get("max_drawdown_pct", 100)
    pnl = metrics.get("total_return_pct", 0)
    gini = gini_coefficient(metrics.get("trade_pnls", np.array([])))
    return (trades >= MIN_TRADES and pf >= MIN_PF and dd <= MAX_DD and
            gini <= MAX_GINI and pnl >= MIN_PNL)


def extract_rules_from_xgboost(booster, feature_cols, condition_lookup, feature_info):
    """Extract rules (paths to positive leaves) from XGBoost trees — single pass.

    Legacy version re-scanned all lines per node (O(n^2)); here children are
    indexed once per tree and DFS walks every node exactly once.
    """
    cond_memo = {}   # (feat_name, code) -> cond_id or None
    for cond in condition_lookup:
        cond_memo[(cond["feature_name"], cond["value_code"])] = cond["id"]

    rules = []
    # feature-info lookup by col name
    info_by_col = {info["col"]: info for info in feature_info}

    dump = booster.get_dump(dump_format='text')
    print(f"   Extracting rules from {len(dump)} trees...")

    node_re = re.compile(r'^(\d+):\[f(\d+)<([\d\.eE+-]+)\]')
    leaf_re = re.compile(r'^(\d+):leaf=([\d\.eE+-]+)')

    for tree_idx, tree_str in enumerate(dump, 1):
        if len(dump) > 500 and tree_idx % 500 == 0:
            print(f"   Processed {tree_idx}/{len(dump)} trees")
        lines = tree_str.split('\n')
        # node id -> line position (nodes AND leaves — leaves carry ids too)
        pos = {}
        for li, l in enumerate(lines):
            s = l.strip()
            m = node_re.match(s)
            if not m:
                m = leaf_re.match(s)
            if m:
                pos[int(m.group(1))] = li

        # DFS stack of (line_idx, conditions). 'yes' branch = condition TRUE (f<..)
        stack = [(0, [])] if pos else []
        while stack:
            li, conds = stack.pop()
            line = lines[li].strip()
            leaf_m = leaf_re.match(line)
            if leaf_m:
                if conds and float(leaf_m.group(1)) > 0:
                    rules.append(tuple(sorted(set(conds))))
                continue
            m = node_re.match(line)
            if not m:
                continue
            node_id, feat_idx, threshold = int(m.group(1)), int(m.group(2)), float(m.group(3))
            feat_name = feature_cols[feat_idx]
            info = info_by_col.get(feat_name)
            cond_id = None
            if info:
                code = int(round(threshold))
                code = max(0, min(info["num_classes"] - 1, code))
                cond_id = cond_memo.get((feat_name, code))
            if cond_id is None:
                # split value maps to no condition — prune deeper paths through this node
                continue
            node_m = re.search(r'yes=(\d+)', line)
            no_m = re.search(r'no=(\d+)', line)
            if not (node_m and no_m):
                continue
            yes_child, no_child = int(node_m.group(1)), int(no_m.group(1))
            if yes_child in pos:
                stack.append((pos[yes_child], conds + [cond_id]))
            if no_child in pos:
                stack.append((pos[no_child], conds))
    return rules


def create_time_environments(meta, n_envs):
    """Split data into n_envs random-day environments (indices into pooled arrays)."""
    dates = np.asarray(meta["datetime"], dtype="datetime64[ns]")
    unique_dates = np.unique(dates.astype("datetime64[D]"))
    np.random.seed(42)
    np.random.shuffle(unique_dates)

    n_days = len(unique_dates)
    env_dates = []
    day_size = n_days // n_envs
    for i in range(n_envs):
        start = i * day_size
        end = start + day_size if i < n_envs - 1 else n_days
        env_dates.append(unique_dates[start:end])

    return [np.where(np.isin(dates.astype("datetime64[D]"), env_dates[i]))[0] for i in range(n_envs)]


def split_dict(d, idx):
    return {k: v[idx] if isinstance(v, np.ndarray) else v for k, v in d.items()}


class XGBoostMiner:
    """Pooled all-symbols miner."""

    def __init__(self, direction="long", n_environments=N_ENVIRONMENTS, use_cache=True):
        self.cfg = load_config()
        self.direction = direction
        self.n_environments = n_environments
        self.cache_dir = os.path.join("results", "models")
        os.makedirs(self.cache_dir, exist_ok=True)
        self.use_cache = use_cache

    def get_cache_path(self, feature_cols):
        tag = "fast" if os.environ.get("MINER_FAST") == "1" else "full"
        name = f"all_symbols_{self.direction}_{len(feature_cols)}f_{tag}_{_XGB_TREES}t.pkl"
        return os.path.join(self.cache_dir, name)

    def get_target(self, labels):
        """Return % relative to entry open (pooled: comparable across symbols)."""
        entry = labels["label_open_next"].astype(np.float32)
        if self.direction == "long":
            return (labels["label_max_288"].astype(np.float32) - entry) / entry * 100.0
        return -(labels["label_min_288"].astype(np.float32) - entry) / entry * 100.0

    def load_data(self):
        from src.features.builder import run
        # RAW features (not min-max normalized): the FeatureDiscretizer bins
        # must be on the same scale the live trader computes at inference.
        pooled = run(self.cfg, split="train", normalize=False)
        return assemble(pooled)

    def train_xgboost(self, features, feature_cols, labels):
        cache_path = self.get_cache_path(feature_cols)
        if self.use_cache and os.path.exists(cache_path):
            print(f"📂 Loading cached model from {cache_path}")
            with open(cache_path, 'rb') as f:
                cached = pickle.load(f)
                return cached['model'], cached['cv_score']

        print(f"🌲 Training XGBoost on pooled ALL-SYMBOL data for {self.direction.upper()}...")
        X = np.column_stack([features[c].astype(np.float32) for c in feature_cols])
        y = self.get_target(labels)
        print(f"   Total samples: {len(X):,}, Features: {len(feature_cols)}")

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)
        model = xgb.XGBRegressor(**XGB_PARAMS)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=100)

        val_score = float(np.corrcoef(y_val, model.predict(X_val))[0, 1])
        print(f"   ✅ Validation correlation: {val_score:.4f}")
        with open(cache_path, 'wb') as f:
            pickle.dump({'model': model, 'cv_score': val_score}, f)
        return model, val_score

    def run(self):
        print(f"\n{'=' * 60}\n🎯 XGBoost Rule Miner — ALL SYMBOLS/pooled | {self.direction.upper()}\n{'=' * 60}\n")
        start = time.time()

        features, feature_cols, labels, meta = self.load_data()
        total_samples = len(features[feature_cols[0]])
        print(f"📊 Samples: {total_samples:,} | Features: {len(feature_cols)}")

        discretizer = FeatureDiscretizer()
        feature_info, condition_lookup = discretizer.fit(features)
        dont_cares = np.array([info["dont_care"] for info in feature_info])
        # integer code features: XGBoost thresholds then map 1:1 to condition value codes
        codes = discretizer.transform(features)

        # pooled train model
        model, cv_score = self.train_xgboost(codes, feature_cols, labels)

        # time environments for robustness checks
        print(f"🔄 Creating {self.n_environments} time-based environments...")
        env_indices = create_time_environments(meta, self.n_environments)
        env_engines = []
        for i, idx in enumerate(env_indices):
            sim = CapitalManagedTradeSimulator(
                split_dict(codes, idx), feature_cols,
                split_dict(labels, idx), {"_symbol_bar_index": meta["_symbol_bar_index"][idx]},
                self.direction, dont_cares)
            env_engines.append(sim)
            print(f"   Env {i + 1}: {len(idx):,} samples")

        # extraction
        print("\n📜 Extracting rules from XGBoost trees...")
        all_rule_ids = extract_rules_from_xgboost(model.get_booster(), feature_cols, condition_lookup, feature_info)
        unique_rules = list(set(all_rule_ids))
        print(f"   Raw: {len(all_rule_ids)}, Unique: {len(unique_rules)}")
        unique_rules = [r for r in unique_rules if 1 <= len(r) <= 6]
        print(f"   After length filter: {len(unique_rules)}")
        if not unique_rules:
            print("⚠️ No rules extracted!")
            return

        engine_train = CapitalManagedTradeSimulator(
            codes, feature_cols, labels, meta, self.direction, dont_cares)

        print(f"\n🔍 Backtesting {len(unique_rules)} rules across {self.n_environments + 1} environments...")
        id_to_cond = {c["id"]: c for c in condition_lookup}
        combos = TP_SL_COMBOS[self.direction]

        # pre-resolve condition strings in the parent (cheap)
        rule_conds = []
        for rule_ids in unique_rules:
            cs = [id_to_cond[cid]["condition_string"] for cid in rule_ids if cid in id_to_cond]
            if cs:
                rule_conds.append(cs)

        # rules are evaluated in forked workers; the engines were built above, so
        # every child inherits them copy-on-write (no pickling of the dataset)
        global _MP_ENGINES, _MP_ENV_IDX, _MP_COMBOS
        _MP_ENGINES = {"train": engine_train, "env": env_engines}
        _MP_ENV_IDX = env_indices
        _MP_COMBOS = combos

        n_workers = int(os.environ.get("MINER_WORKERS", max(2, min(6, (os.cpu_count() or 4) - 2))))
        chunk_size = 400
        chunks = [rule_conds[i:i + chunk_size] for i in range(0, len(rule_conds), chunk_size)]

        import multiprocessing as mp
        ctx = mp.get_context("fork")
        results, n_kept, n_signal_ok = [], 0, 0
        with ctx.Pool(processes=n_workers) as pool:
            for j, (chunk_results, n_ok) in enumerate(pool.imap_unordered(_eval_rule_chunk, chunks)):
                results.extend(chunk_results)
                n_signal_ok += n_ok
                n_kept += len(chunk_results)
                done = min(len(rule_conds), (j + 1) * chunk_size)
                print(f"   Progress: {done}/{len(rule_conds)} rules, {n_kept} passed", flush=True)

        elapsed = time.time() - start
        print(f"\n✅ Completed in {elapsed:.1f}s — rules passing: {len(results)} "
              f"(of {n_signal_ok} with enough signals)")

        results.sort(key=lambda x: -x['score'])
        elapsed = time.time() - start
        print(f"\n✅ Completed in {elapsed:.1f}s — rules passing: {len(results)}")

        self.save(results, cv_score)
        self.print_top(results)

    def save(self, results, cv_score):
        out_dir = os.path.join("results", "pool", self.direction)
        os.makedirs(out_dir, exist_ok=True)
        out_file = os.path.join(out_dir, "all_symbols.csv")
        with open(out_file, "w", newline="", encoding="utf8") as f:
            writer = csv.writer(f)
            header = ["conditions", "tp", "sl", "capital_pct",
                      "train_pnl", "train_dd", "train_pf", "train_trades",
                      "env_passed"]
            writer.writerow(header)
            for r in results[:TOP_K]:
                row = [" AND ".join(r['conditions']), r['tp'], r['sl'], r['cap'],
                       f"{r['train_pnl']:.2f}", f"{r['train_dd']:.2f}",
                       f"{r['train_pf']:.2f}", r['train_trades'],
                       f"{r['env_passed']}/{self.n_environments}"]
                writer.writerow(row)
        print(f"💾 Saved to {out_file}")

    def print_top(self, results):
        print("\n📊 Top 5 rules:")
        for i, r in enumerate(results[:5]):
            print(f"   {i + 1}. Score={r['score']:.2f}, Train PnL={r['train_pnl']:.2f}%, "
                  f"Avg Env PnL={np.mean(r['env_pnls']):.2f}%, TP={r['tp']}, SL={r['sl']}, "
                  f"Passed {r['env_passed']}/{self.n_environments}")
            print(f"      {' AND '.join(r['conditions'])}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--direction", default="long", choices=["long", "short"])
    ap.add_argument("--n-environments", type=int, default=N_ENVIRONMENTS)
    args = ap.parse_args()
    XGBoostMiner(args.direction, args.n_environments).run()
# ------------------------------------------------------------------ multiprocessing
_MP_ENGINES = None      # {"train": sim, "env": [sims]} — set in parent BEFORE fork
_MP_ENV_IDX = None
_MP_COMBOS = None


def _eval_rule_chunk(chunk):
    """Evaluate a chunk of rules inside a forked worker.

    Engines and dataset arrays are inherited copy-on-write, so workers only
    build their rule masks and simulate — no pickling of the dataset back.
    Returns (passing_rule_dicts, n_rules_with_enough_signals).
    """
    from src.core.backtest import build_rule_set_entries, _build_trade_outcomes_numba, _simulate_core_with_details
    train = _MP_ENGINES["train"]
    env_engines = _MP_ENGINES["env"]

    def sim_masked(engine, mask, tp, sl, cap):
        rule = {"signal_mask": mask, "tp": float(tp), "sl": float(sl), "capital_pct": float(cap)}
        idx_arr, tp_arr, sl_arr, cap_pct_arr, rule_idx_arr = build_rule_set_entries(
            engine.data_matrix, [rule], engine.dont_cares, engine.feature_dict)
        if len(idx_arr) == 0:
            return engine._empty_metrics()
        release_idx_arr = engine.release_index[idx_arr]
        ret_arr, _ = _build_trade_outcomes_numba(
            idx_arr, tp_arr, sl_arr, engine.max_ret, engine.min_ret,
            engine.close_ret, engine.max_before_min, engine.is_long)
        symbol_idx_arr = engine.symbol_codes[idx_arr]
         (total_ret, max_dd, avg_dd, executed, wins, losses,
          gross_profit, gross_loss, notional_sum, skipped,
          max_simul, max_total_exp, trade_pnls, trade_entry_idx,
          trade_exit_idx, trade_notional, trade_rule_idx, equity_curve) = _simulate_core_with_details(
            idx_arr, ret_arr, cap_pct_arr, release_idx_arr, symbol_idx_arr,
            rule_idx_arr,
            engine.initial_capital, engine.leverage, engine.max_total_exposure_rate,
            engine.min_position_notional, engine.round_trip_fee_rate)
        return {
            "total_return_pct": float(total_ret), "max_drawdown_pct": float(max_dd),
            "executed_trades": int(executed),
            "win_rate": (wins / executed * 100.0) if executed > 0 else 0.0,
            "profit_factor": safe_profit_factor(gross_profit, gross_loss),
            "trade_pnls": trade_pnls,
        }

    passing, n_signal_ok = [], 0
    for conditions in chunk:
        try:
            mask = build_rule_signal_mask(train.feature_dict, conditions)
        except ValueError:
            continue
        if int(mask.sum()) < MIN_TRADES:
            continue
        n_signal_ok += 1

        best_combo, best_score = None, -float("inf")
        for tp, sl, cap in _MP_COMBOS:
            try:
                mt = sim_masked(train, mask, tp, sl, cap)
                if not check_rule_passes(mt):
                    continue
                env_metrics, env_passed = [], 0
                for j, e in enumerate(env_engines):
                    m = sim_masked(e, mask[_MP_ENV_IDX[j]], tp, sl, cap)
                    env_metrics.append(m)
                    if m["executed_trades"] >= MIN_TRADES and m["profit_factor"] >= MIN_PF and \
                       m["total_return_pct"] >= MIN_PNL:
                        env_passed += 1
                if env_passed < max(1, len(env_engines) - 1):
                    continue
                pnl_train = mt["total_return_pct"]
                env_pnls = [m["total_return_pct"] for m in env_metrics]
                all_pnls = [pnl_train] + env_pnls
                avg_pnl = float(np.mean(all_pnls))
                min_pnl = float(np.min(all_pnls))
                stability = min_pnl / max(all_pnls) if max(all_pnls) > 0 else 0.0
                consistency = env_passed / len(env_engines)
                score = avg_pnl * stability * (0.5 + 0.5 * consistency)
                if score > best_score:
                    best_score = score
                    best_combo = {
                        "score": score, "avg_pnl": avg_pnl, "pnl_stability": stability,
                        "env_consistency": consistency, "env_passed": env_passed,
                        "conditions": list(conditions), "tp": tp, "sl": sl, "cap": cap,
                        "train_pnl": pnl_train, "train_dd": mt["max_drawdown_pct"],
                        "train_pf": mt["profit_factor"], "train_trades": mt["executed_trades"],
                        "train_gini": gini_coefficient(mt.get("trade_pnls", np.array([]))),
                        "env_pnls": env_pnls,
                        "env_dds": [m["max_drawdown_pct"] for m in env_metrics],
                        "env_pfs": [m["profit_factor"] for m in env_metrics],
                        "env_trades": [m["executed_trades"] for m in env_metrics],
                        "env_ginis": [gini_coefficient(m.get("trade_pnls", np.array([]))) for m in env_metrics],
                    }
            except Exception:
                continue
        if best_combo:
            passing.append(best_combo)
    return passing, n_signal_ok


if __name__ == "__main__":
    main()
