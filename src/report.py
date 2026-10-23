#!/usr/bin/env python3
"""Report: evaluate pooled mined rules on BOTH train and test splits.

    python -m src.report --direction long --top 20

Methodology (parity with live):
  - features built for TRAIN and TEST separately
  - the FeatureDiscretizer is fitted on TRAIN only; test rows are transformed
    with the SAME train bin edges (no test-set refit, no bin leakage)
  - every pool rule is simulated on the train engine and the test engine
  - ranking uses both splits (default: min(train_pnl, test_pnl)) so selection
    demands profitability on BOTH sides of the boundary

Writes:
    results/fits/{direction}/test_eval.csv        train+test metrics per rule
    results/fits/{direction}/strategy_{i}.json    top strategies with both metrics
    results/fits/{direction}/fit_package.pkl      train-fitted bins for phase 2
"""
import argparse
import csv
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import load_config
from src.core.discretizer import FeatureDiscretizer
from src.core.backtest import CapitalManagedTradeSimulator
from src.data.loader import assemble
from src.features.builder import run as build_features
from src.live.fit_package import save_discretizer


def _rank_score(r: dict, rank_by: str) -> float:
    if rank_by == "train":
        return r["train_pnl"]
    if rank_by == "test":
        return r["test_pnl"]
    if rank_by == "avg":
        return 0.5 * (r["train_pnl"] + r["test_pnl"])
    return min(r["train_pnl"], r["test_pnl"])  # "min"


def evaluate():
    cfg = load_config()
    if globals().get("_direction_override"):
        cfg.direction = globals()["_direction_override"]
    topk = globals().get("GLOBAL_TOPK", 10)
    rank_by = globals().get("_rank_by", "min")
    force = globals().get("_force", False)
    out_dir = f"results/fits/{cfg.direction}"
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n📊 Train+test evaluation ({cfg.direction})")

    print("   building TRAIN features (raw scale, live parity)...")
    train_df = build_features(cfg, split="train", force=force, normalize=False)
    tr_features, fc, tr_labels, tr_meta = assemble(train_df)
    n_train = len(next(iter(tr_features.values())))
    print(f"   train rows: {n_train:,}")

    # fit bins on TRAIN ONLY — the test transform must reuse these edges
    disc = FeatureDiscretizer()
    fi, _ = disc.fit(tr_features)
    dc = np.array([i["dont_care"] for i in fi])
    tr_engine = CapitalManagedTradeSimulator(
        disc.transform(tr_features), fc, tr_labels, tr_meta, cfg.direction, dc)

    print("   building TEST features (raw scale)...")
    test_df = build_features(cfg, split="test", force=force, normalize=False)
    te_features, fc_te, te_labels, te_meta = assemble(test_df)
    if fc_te != fc:
        raise RuntimeError("feature columns differ between train and test builds")
    te_engine = CapitalManagedTradeSimulator(
        disc.transform(te_features), fc, te_labels, te_meta, cfg.direction, dc)
    n_test = len(next(iter(te_features.values())))
    print(f"   test rows: {n_test:,}")

    # persist the train-fitted bins so phase 2 evaluates with EXACT parity
    pkg_path = os.path.join(out_dir, "fit_package.pkl")
    save_discretizer(disc, pkg_path)
    print(f"💾 train-fitted discretizer package -> {pkg_path}")

    pool_csv = f"results/pool/{cfg.direction}/all_symbols.csv"
    rows = []
    with open(pool_csv, "r", encoding="utf8") as f:
        for r in csv.DictReader(f):
            conds = [c.strip() for c in re.split(r" AND ", r["conditions"]) if c.strip()]
            rule = [{"conditions": conds, "tp": float(r["tp"]), "sl": float(r["sl"]),
                     "capital_pct": float(r["capital_pct"])}]
            try:
                tm = tr_engine.simulate_rule_set(rule)
                om = te_engine.simulate_rule_set(rule)
            except ValueError:
                continue
            if tm["executed_trades"] == 0 and om["executed_trades"] == 0:
                continue
            rows.append({
                "conditions": r["conditions"], "tp": r["tp"], "sl": r["sl"],
                "cap": r["capital_pct"],
                "train_pnl": round(tm["total_return_pct"], 2),
                "train_dd": round(tm["max_drawdown_pct"], 2),
                "train_pf": round(tm["profit_factor"], 2),
                "train_trades": tm["executed_trades"],
                "train_wr": round(tm["win_rate"], 2),
                "test_pnl": round(om["total_return_pct"], 2),
                "test_dd": round(om["max_drawdown_pct"], 2),
                "test_pf": round(om["profit_factor"], 2),
                "test_trades": om["executed_trades"],
                "test_wr": round(om["win_rate"], 2),
            })
    if not rows:
        print("⚠️  No rules to evaluate — run the miner first (python -m src.miner)")
        return
    rows.sort(key=lambda x: -_rank_score(x, rank_by))
    out_csv = os.path.join(out_dir, "test_eval.csv")
    with open(out_csv, "w", newline="", encoding="utf8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"💾 {len(rows)} rules evaluated (ranked by {rank_by}) -> {out_csv}")

    # top strategies JSON (phase-2 runtime load format) — require BOTH sides > 0
    kept = 0
    for r in rows:
        if kept >= topk:
            break
        if r["train_pnl"] <= 0 or r["test_pnl"] <= 0:
            continue
        if r["train_trades"] < 1 or r["test_trades"] < 1:
            continue
        with open(os.path.join(out_dir, f"strategy_{kept + 1}.json"), "w") as f:
            json.dump({
                "direction": cfg.direction,
                "strategy": {"rules_set": [
                    {"conditions": [c.strip() for c in r["conditions"].split(" AND ")],
                     "tp": float(r["tp"]), "sl": float(r["sl"]),
                     "capital_pct": float(r["cap"])}
                ]},
                "train": {"pnl": r["train_pnl"], "dd": r["train_dd"],
                          "trades": r["train_trades"], "pf": r["train_pf"]},
                "test": {"pnl": r["test_pnl"], "dd": r["test_dd"],
                         "trades": r["test_trades"], "pf": r["test_pf"]},
                "rank_key": rank_by,
            }, f, indent=2, default=str)
        kept += 1
    print(f"💾 {kept} top strategies (train & test both positive) saved")


GLOBAL_TOPK = 10


if __name__ == "__main__":
    _ap = argparse.ArgumentParser()
    _ap.add_argument("--direction", default="long", choices=["long", "short"])
    _ap.add_argument("--top", type=int, default=GLOBAL_TOPK)
    _ap.add_argument("--rank-by", default="min", choices=["min", "avg", "train", "test"])
    _ap.add_argument("--force", action="store_true",
                     help="rebuild train/test feature cache (e.g. after label changes)")
    _args = _ap.parse_args()
    globals()["_direction_override"] = _args.direction
    globals()["_rank_by"] = _args.rank_by
    globals()["_force"] = _args.force
    globals()["GLOBAL_TOPK"] = _args.top
    evaluate()
