"""One-off feature-quality prune: remove near-constant and redundant features.

Criteria (fitted on TRAIN split only):
  1) na_dominant  : >50% of the feature's train rows are undefined (NaN).
  2) near-constant: ~zero range/variance or fewer than 5 distinct values.
  3) redundancy   : |Spearman correlation| > 0.90 — from each redundant cluster
                    the member with the lowest mean |Spearman| with other
                    features is dropped iteratively (keeps the most central).

Drops the columns in-place from dataset/features/{train,test}.{parquet,csv} and
writes dataset/features/pruning_report.csv.
"""
import os
import numpy as np
import pandas as pd

FEAT_DIR = "dataset/features"
NA_FRAC_MAX = 0.50
NUNIQ_MIN = 5
CORR_MAX = 0.90
CORR_METHOD = "spearman"

NON_FEATURE = ["datetime", "symbol", "_symbol_bar_index"]
NON_FEATURE += ["candle_open", "candle_high", "candle_low", "candle_close", "candle_volume"]


def main():
    train = pd.read_parquet(os.path.join(FEAT_DIR, "train.parquet"))
    feature_cols = [c for c in train.columns
                    if c not in NON_FEATURE and not c.startswith("label_")]
    X = train[feature_cols].astype(np.float64)

    na_frac = X.isna().mean()
    nunique = X.nunique()

    reasons = {}
    for c in feature_cols:
        if na_frac[c] > NA_FRAC_MAX:
            reasons[c] = f"na_frac={na_frac[c]:.2f}"
        elif nunique[c] < NUNIQ_MIN:
            reasons[c] = f"nunique={nunique[c]}"
        elif (X[c].max() - X[c].min()) < 1e-9:
            reasons[c] = f"zero_range={X[c].max() - X[c].min():.2e}"

    since = [c for c in feature_cols if c not in reasons]
    corr = X[since].corr(method=CORR_METHOD)
    corr_df_index = corr.index
    corr = corr.to_numpy(copy=True)
    diag = np.diag_indices_from(corr)
    corr[diag] = 0.0
    corr = pd.DataFrame(corr, index=corr_df_index, columns=corr_df_index)
    spear = X[since].corr(method="spearman").abs().mean()

    redundant = []
    cand = list(since)
    while True:
        sub = corr.loc[cand, cand]
        ub = np.triu(np.ones(sub.shape), 1).astype(bool)
        hit = sub.abs().where(ub).stack()
        hit = hit[hit > CORR_MAX]
        if hit.empty:
            break
        # cluster anchored at the worst pair; drop the member with the lowest
        # mean |spearman correlation| to the rest of the feature set
        pair = list(hit.index)[0]
        denom = (sub.abs() > CORR_MAX)
        cluster = [c for c in cand if denom.loc[pair[0], c] or denom.loc[pair[1], c]]
        cluster = sorted(set(cluster) | {pair[0], pair[1]})
        scores = spear.loc[cluster]
        victim = scores.idxmin()
        redundant.append(victim)
        reasons[victim] = f"corr>{CORR_MAX} with " + (
            pair[0] if pair[1] == victim else pair[1])
        cand.remove(victim)

    all_drop = sorted(reasons)
    keep = [c for c in feature_cols if c not in all_drop]

    report = pd.DataFrame({
        "feature": feature_cols,
        "action": ["drop" if c in reasons else "keep" for c in feature_cols],
        "reason": [reasons.get(c, "") for c in feature_cols],
        "na_frac": [float(na_frac[c]) for c in feature_cols],
        "nunique": [int(nunique[c]) for c in feature_cols],
    })
    report.to_csv(os.path.join(FEAT_DIR, "pruning_report.csv"), index=False)

    for split in ("train", "test"):
        path = os.path.join(FEAT_DIR, f"{split}.parquet")
        df = pd.read_parquet(path)
        to_drop = [c for c in all_drop if c in df.columns]
        if to_drop:
            df = df.drop(columns=to_drop)
            df.to_parquet(path)
            df.to_csv(path.replace(".parquet", ".csv"), index=False)
        print(f"   💾 pruned {split}: {len(df):,} rows, {len(df.columns)} cols")

    print(f"\n📊 Feature prune summary (fitted on train only)")
    print(f"   dropped near-constant/na-dominant: {len([c for c in all_drop if 'corr' not in reasons[c]])}")
    print(f"   dropped redundant (|{CORR_METHOD}|>{CORR_MAX}): {len(redundant)}")
    print(f"   kept: {len(keep)} / {len(feature_cols)}")
    groups = {}
    for c, r in reasons.items():
        key = r.split("=")[0].split(" ")[0]
        groups.setdefault(key, []).append(f"{c} ({r})")
    for k, v in groups.items():
        print(f"\n   [{k}] {len(v)}")
        for row in sorted(v):
            print(f"     - {row}")


if __name__ == "__main__":
    main()
