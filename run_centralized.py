# run_centralized.py
# Run GRU experiments with feature-mode aware logic.
#
# Behavior:
# - FEATURE_MODE == "all":
#     run Top-K sweep using rank_path
# - FEATURE_MODE == "vitals":
#     run a single GRU using the full vitals parquet (no feature selection)
# - FEATURE_MODE == "vitals+demo":
#     run a single GRU using vitals + baseline parquet (no feature selection)
#
# Output CSV columns (as requested):
# - AUROC/AUPRC only for TRAIN and TEST (no VAL columns)
# - Threshold-based metrics only for TEST
# - cpu_peak_mib and runtime_sec are the final two columns

from __future__ import annotations
from typing import List, Optional

import pandas as pd

import config
from train_eval_gru import TrainConfig, train_and_eval


def _round_3dp(df: pd.DataFrame) -> pd.DataFrame:
    df2 = df.copy()
    for c in df2.columns:
        if pd.api.types.is_numeric_dtype(df2[c]):
            df2[c] = df2[c].round(3)
    return df2


def _feature_mode() -> str:
    mode = str(getattr(config, "FEATURE_MODE", "all")).strip().lower()
    valid = {"all", "vitals", "vitals+demo"}
    if mode not in valid:
        raise ValueError(f"Unsupported config.FEATURE_MODE={mode!r}. Expected one of {sorted(valid)}")
    return mode


def run_sweep(
    ks: List[Optional[int]],
    rank_path: Optional[str] = None,
) -> pd.DataFrame:

    disease = config.DISEASE
    cfg = TrainConfig()
    feature_mode = _feature_mode()

    rows = []

    if feature_mode == "all":
        run_ks = ks
        run_rank_path = rank_path
        if run_rank_path is None:
            raise ValueError("FEATURE_MODE='all' requires rank_path for Top-K GRU sweep.")
    else:
        # No feature selection for vitals or vitals+demo
        run_ks = [None]
        run_rank_path = None

    for k in run_ks:
        if feature_mode == "all":
            k_name = "all" if k is None else str(int(k))
            print("#############################")
            print(f"GRU run | FEATURE_MODE={feature_mode} | top_k={k_name}")
            print(f"rank_path={run_rank_path}")
            print("#############################")
        else:
            k_name = "none"
            print("#############################")
            print(f"GRU run | FEATURE_MODE={feature_mode} | no feature selection")
            print("#############################")

        out = train_and_eval(
            disease=disease,
            cfg=cfg,
            top_k=k,
            rank_path=run_rank_path,
        )

        row = {
            "feature_mode": out["extra"].get("feature_mode", feature_mode),
            "top_k": k_name,
            "n_features": out["extra"]["n_features"],

            # AUROC/AUPRC: train + test only
            "train_auroc": out["train"]["auroc"],
            "train_auprc": out["train"]["auprc"],
            "test_auroc": out["test"]["auroc"],
            "test_auprc": out["test"]["auprc"],

            # TEST threshold + decision metrics
            "threshold": out["test"].get("threshold", out["extra"].get("threshold", float("nan"))),
            "test_accuracy": out["test"].get("accuracy", float("nan")),
            "test_precision": out["test"].get("precision", float("nan")),
            "test_recall": out["test"].get("recall", float("nan")),
            "test_f1": out["test"].get("f1", float("nan")),
            "test_fpr": out["test"].get("fpr", float("nan")),
            "test_tn": out["test"].get("tn", float("nan")),
            "test_fp": out["test"].get("fp", float("nan")),
            "test_fn": out["test"].get("fn", float("nan")),
            "test_tp": out["test"].get("tp", float("nan")),

            # Class counts (test)
            "test_n_pos": out["test"].get("n_pos", float("nan")),
            "test_n_neg": out["test"].get("n_neg", float("nan")),

            # Must be final 2 columns
            "cpu_peak_mib": out["extra"]["cpu_peak_mib"],
            "runtime_sec": out["extra"]["runtime_sec"],
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    col_order = [
        "feature_mode",
        "top_k",
        "n_features",
        "train_auroc",
        "train_auprc",
        "test_auroc",
        "test_auprc",
        "threshold",
        "test_accuracy",
        "test_precision",
        "test_recall",
        "test_f1",
        "test_fpr",
        "test_tn",
        "test_fp",
        "test_fn",
        "test_tp",
        "test_n_pos",
        "test_n_neg",
        "cpu_peak_mib",
        "runtime_sec",
    ]
    df = df[col_order]

    df_out = _round_3dp(df)

    out_path = config.gru_results_path(disease)
    df_out.to_csv(out_path, index=False)

    print("#############################")
    print("Summary:")
    print(df_out.to_string(index=False))
    print("----------------------------------------")
    print(f"Saved results CSV: {out_path}")
    print("#############################")

    return df_out


if __name__ == "__main__":
    feature_mode = _feature_mode()

    if feature_mode == "all":
        ks = [60]
        rank_path = str(config.stability_combined_path(config.DISEASE))
    else:
        ks = [None]
        rank_path = None

    run_sweep(ks=ks, rank_path=rank_path)
