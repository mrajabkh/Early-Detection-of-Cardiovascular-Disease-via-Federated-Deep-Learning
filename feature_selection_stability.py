# feature_selection_stability.py
# RF-only stability feature selection:
# - Random Forest MDI importance across bootstraps
# Produces stability_combined CSV compatible with your top-K pipeline.
# Location: Project/Code/feature_selection_stability.py

from __future__ import annotations

import time
from typing import List

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

import config
import preprocess


#############################
# Utilities
#############################
def set_seeds(seed: int) -> None:
    np.random.seed(seed)


def _topk_freq_from_matrix(importance_mat: np.ndarray, feature_names: List[str], topk_ref: int) -> pd.Series:
    n_runs, n_feat = importance_mat.shape
    K = min(int(topk_ref), n_feat)

    counts = np.zeros(n_feat, dtype=np.int32)
    for i in range(n_runs):
        idx = np.argsort(-importance_mat[i])[:K]
        counts[idx] += 1

    return pd.Series(counts / float(n_runs), index=feature_names, name="stability_freq")


def _minmax_norm(s: pd.Series) -> pd.Series:
    mn = float(s.min())
    mx = float(s.max())
    if mx <= mn:
        return pd.Series(np.zeros(len(s), dtype=np.float64), index=s.index)
    return (s - mn) / (mx - mn)


#############################
# Main
#############################
def main() -> None:
    set_seeds(config.SEED)

    features_path = config.features_path(config.DISEASE)
    samples_path = config.samples_path(config.DISEASE)

    out_rf = config.stability_rf_path(config.DISEASE)
    out_combined = config.stability_combined_path(config.DISEASE)

    print("#############################")
    print("RF-only stability feature selection")
    print("#############################")
    print(f"Features: {features_path}")
    print(f"Samples:  {samples_path}")
    print(f"Out RF:   {out_rf}")
    print(f"Out Comb: {out_combined}")

    X_df, y, split_data = preprocess.load_build_split(
        features_parquet_path=features_path,
        samples_csv_path=samples_path,
        test_size=config.TEST_SIZE,
        random_state=config.SPLIT_RANDOM_STATE,
        stratify=config.STRATIFY_SPLIT,
        impute_strategy=config.IMPUTE_STRATEGY,
        scale_numeric=False,  # RF does not need scaling
    )

    feat_names = split_data.artifacts.valid_feature_names
    n_feat = len(feat_names)

    X_train_imp = split_data.X_train_imputed
    y_train = split_data.y_train

    idx_all = np.arange(len(y_train), dtype=np.int32)

    # Hold out a portion for bootstrap pool only (not strictly needed, but keeps behavior stable)
    idx_boot_pool, _ = train_test_split(
        idx_all,
        test_size=0.25,
        random_state=config.SEED,
        stratify=y_train if config.STRATIFY_SPLIT else None,
    )

    print("#############################")
    print(f"Train rows: {len(y_train)}")
    print(f"Boot pool rows: {len(idx_boot_pool)}")
    print(f"Features: {n_feat}")
    print("#############################")

    n_runs = int(getattr(config, "STAB_N_BOOTSTRAPS", 10))
    boot_frac = float(getattr(config, "STAB_BOOTSTRAP_FRAC", 0.8))
    topk_ref = int(getattr(config, "STAB_TOPK_REF", 50))
    freq_thr = float(getattr(config, "STAB_FREQ_THRESHOLD", 0.6))

    rf_params = dict(config.RF_TUNED_PARAMS)

    rf_importance_mat = np.zeros((n_runs, n_feat), dtype=np.float64)

    for r in range(n_runs):
        print("#############################")
        print(f"RF Bootstrap {r + 1}/{n_runs}")
        print("#############################")

        rs = np.random.RandomState(config.SEED + 1000 + r)

        boot_n = max(10, int(round(boot_frac * len(idx_boot_pool))))
        boot_idx = rs.choice(idx_boot_pool, size=boot_n, replace=True)

        Xb = X_train_imp[boot_idx]
        yb = y_train[boot_idx]

        t0 = time.time()
        rf = RandomForestClassifier(**rf_params)
        rf.fit(Xb, yb)
        rf_importance_mat[r, :] = rf.feature_importances_
        print(f"RF+MDI done in {time.time() - t0:.2f}s")

    rf_mean = pd.Series(rf_importance_mat.mean(axis=0), index=feat_names, name="mean_importance")
    rf_std = pd.Series(rf_importance_mat.std(axis=0), index=feat_names, name="std_importance")
    rf_freq = _topk_freq_from_matrix(rf_importance_mat, feat_names, topk_ref)

    rf_df = pd.concat([rf_mean, rf_std, rf_freq], axis=1).reset_index().rename(columns={"index": "feature"})
    rf_df = rf_df.sort_values(by=["stability_freq", "mean_importance"], ascending=[False, False])

    out_rf.parent.mkdir(parents=True, exist_ok=True)
    rf_df.to_csv(out_rf, index=False)

    print("#############################")
    print(f"Saved RF stability: {out_rf}")
    print("#############################")

    # Combined output (compatible schema)
    rf_score_norm = _minmax_norm(rf_mean)
    keep_mask = rf_freq >= freq_thr

    combined_df = pd.DataFrame({
        "feature": feat_names,
        "combined_score": rf_score_norm.values,
        "rf_mean_importance": rf_mean.values,
        "rf_stability_freq": rf_freq.values,
        "kept_by_threshold": keep_mask.values.astype(int),
    })

    combined_df = combined_df.sort_values(
        by=["kept_by_threshold", "combined_score"],
        ascending=[False, False],
    )

    out_combined.parent.mkdir(parents=True, exist_ok=True)
    combined_df.to_csv(out_combined, index=False)

    print("#############################")
    print(f"Saved combined stability ranking: {out_combined}")
    print("Top 20 combined:")
    print(combined_df.head(20).to_string(index=False))
    print("#############################")


if __name__ == "__main__":
    main()
