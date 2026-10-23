"""Export the phase-1 fit needed by the live trader (--package step).

    python scripts/package_fit.py --direction long
    python scripts/package_fit.py --direction short --force

Builds the pooled RAW-scale research features (the same scale the live
evaluator computes), fits the FeatureDiscretizer on TRAIN rows only, and
pickles feature_info + bin edges into results/fits/{direction}/fit_package.pkl.

Why raw: min-max normalization is monotonic so quantile bands are unchanged,
but fitting bins on an already-normalized frame and then applying them to raw
live values (or vice versa) would make every "High"/"Low" condition wrong.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import load_config
from src.core.discretizer import FeatureDiscretizer
from src.data.loader import assemble
from src.features.builder import FeatureBuilder
from src.live.fit_package import (ARTIFACT_VERSION, load_ruleset,
                                  save_discretizer)


def _missing_features(ruleset, feature_info):
    known = {fi["col"] for fi in feature_info}
    missing = set()
    for rule in ruleset["rules"]:
        for cond in rule["conditions"]:
            missing.add(cond.split("]")[0].strip("[]"))
    return sorted(missing - known)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--direction", default="long", choices=["long", "short"])
    ap.add_argument("--split", default="train", choices=["train", "test"])
    ap.add_argument("--force", action="store_true",
                    help="rebuild the raw pooled feature cache from candles")
    args = ap.parse_args()

    cfg = load_config(direction=args.direction)
    builder = FeatureBuilder(cfg)
    print(f"📦 Building RAW pooled features (artifact v{ARTIFACT_VERSION}), "
          f"direction={args.direction}")
    pooled = builder.build_raw_pooled(use_cache=not args.force)
    train = FeatureBuilder.split_raw_pooled(pooled, cfg)
    features, cols, labels, meta = assemble(train)
    print(f"   train rows={len(next(iter(features.values()))):,} "
          f"features={len(cols)} (warm-up={builder.max_window} bars)")

    disc = FeatureDiscretizer()
    disc.fit(features)
    out = os.path.join("results", "fits", args.direction)
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, "fit_package.pkl")
    save_discretizer(disc, path)
    print(f"💾 Saved discretizer package: {path}")

    manifest = {
        "artifact_version": ARTIFACT_VERSION,
        "direction": args.direction,
        "fit_split": args.split,
        "train_rows": int(len(next(iter(features.values())))),
        "warmup_bars": int(builder.max_window),
        "feature_count": len(disc.feature_info),
        "features": [fi["col"] for fi in disc.feature_info],
        "fit_on": "raw train-only",
    }
    with open(os.path.join(out, "fit_package_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1, default=str)

    rs = os.path.join(out, "ruleset.json")
    if os.path.exists(rs):
        rules = load_ruleset(rs)
        missing = _missing_features(rules, disc.feature_info)
        if missing:
            raise SystemExit(f"❌ ruleset uses features missing from package: {missing}")
        print(f"✅ ruleset OK: {len(rules['rules'])} rules | "
              f"direction={rules['direction']} | all features covered")
    else:
        print(f"ℹ️ no ruleset yet at {rs} — run scripts/select_strategy.py")


if __name__ == "__main__":
    main()
