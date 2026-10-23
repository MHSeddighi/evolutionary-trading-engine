"""Phase 1 entrypoint: pooled NSGA-II ruleset search across ALL symbols in one run.

    python main.py --quick          # 5-gen smoke/benchmark (< 10s target)
    python main.py --direction long # full run
Phase 2 later reuses this same pipeline with phase="live" (DB source + Wallex execution).
"""
import argparse
from typing import Any, Dict

import numpy as np

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.ux import UX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.optimize import minimize
from pymoo.termination.default import DefaultMultiObjectiveTermination

from config import load_config
from src.core.callbacks import HistorySaver, GenUpdater, CallbackCombiner
from src.core.discretizer import FeatureDiscretizer
from src.core.decoder import Decoder
from src.core.output import DynamicOutput
from src.core.fitness import MultiObjectiveProblem
from src.initialization import RulePool, PoolBasedSampling, build_pool
from src.data.loader import assemble
from src.features import builder


def _build_dataset(cfg, symbol_col=True):
    """Train split ONLY — the NSGA fitness must never see the test window.
    Raw feature scale: bins/conditions must match what live computes."""
    train_df = builder.run(cfg, split="train", normalize=False)
    return assemble(train_df)


def main(quick: bool = False, direction: str = None) -> Dict[str, Any]:
    cfg = load_config(quick=quick, direction=direction)

    print(f"\n{'=' * 60}\n🎯 Pooled NSGA-II | phase={cfg.phase} | {cfg.direction.upper()} | symbols={len(cfg.symbols)}\n{'=' * 60}")

    print("📦 Loading pooled data (all symbols)...")
    features, feature_cols, labels, meta = _build_dataset(cfg)
    if cfg.quick_subsample:
        bi = meta["_symbol_bar_index"]
        maxbi = np.zeros(bi.max() + 1, dtype=np.int64)
        np.maximum.at(maxbi, meta["symbol"], bi)
        keep = bi >= (maxbi[meta["symbol"]] - cfg.quick_subsample + 1)
        features = {k: v[keep] for k, v in features.items()}
        labels = {k: v[keep] for k, v in labels.items()}
        meta = {k: v[keep] if isinstance(v, np.ndarray) and len(v) == len(keep) else v for k, v in meta.items()}

    print(f"   Features: {len(feature_cols)} | Samples: {len(next(iter(features.values()))):,}")

    discretizer = FeatureDiscretizer()
    feature_info, condition_lookup = discretizer.fit(features)
    dont_cares = np.array([info["dont_care"] for info in feature_info])
    print(f"   Conditions: {len(condition_lookup)}")
    # backtester + optimizer evaluate conditions on DISCRETIZED codes so the
    # quantile-bin thresholds in conditions match the raw magic thresholds exactly
    features = discretizer.transform(features)

    rule_pool = build_pool(condition_lookup, features, csv_path=cfg.pool_csv,
                           n_clusters=cfg.pool_clusters)
    print(f"   Pool rules: {len(rule_pool)}")

    decoder = Decoder(
        max_rules=cfg.max_rules, min_rules=cfg.min_rules, direction=cfg.direction,
        rule_pool=rule_pool, cap_min=1.0, cap_max=3.0, cap_scale=0.2, min_tp=0.5, min_sl=0.5,
    )
    print(f"   Chromosome: {decoder.n_var} genes ({cfg.max_rules} slots × 5)")

    sampling = PoolBasedSampling(
        decoder=decoder, rule_pool=rule_pool, condition_lookup=condition_lookup,
        pool_ratio=cfg.pool_ratio, seed_ratio=0.0, seed_dir=cfg.seed_dir)

    mutation = PM(prob=cfg.mutation_rate, eta=cfg.mutation_eta, vtype=int, repair=RoundingRepair())
    problem = MultiObjectiveProblem(
        min_trades_per_rule=cfg.min_trade_per_rule,
        max_open_exposure=cfg.max_open_exposure,
        max_open_positions=cfg.max_open_positions,
        features=features, feature_cols=feature_cols, labels=labels, meta=meta,
        decoder=decoder, dont_cares=dont_cares, min_trades=cfg.min_trade, direction=cfg.direction,
        n_folds=cfg.n_folds,
    )
    algorithm = NSGA2(
        pop_size=cfg.pop_size, sampling=sampling, crossover=UX(prob=cfg.crossover_rate),
        mutation=mutation, output=DynamicOutput(), eliminate_duplicates=True,
    )
    history_saver = HistorySaver()
    gen_updater = GenUpdater(problem, fallback_max_gen=cfg.max_generations)
    combined = CallbackCombiner(problem, history_saver, gen_updater)
    termination = DefaultMultiObjectiveTermination(
        n_max_gen=cfg.max_generations, n_max_evals=cfg.max_eval,
        ftol=0.005, xtol=0.01, n_skip=10)

    print(f"\n🚀 Starting pooled optimization ({cfg.max_generations} gens, pop {cfg.pop_size})\n")
    result = minimize(problem, algorithm, termination, callback=combined, seed=cfg.seed, verbose=True)
    return {"result": result, "cfg": cfg, "history": history_saver.history,
            "features": features, "feature_cols": feature_cols, "labels": labels, "meta": meta,
            "decoder": decoder, "rule_pool": rule_pool}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--direction", default="long", choices=["long", "short"])
    args = ap.parse_args()
    main(quick=args.quick, direction=args.direction)
