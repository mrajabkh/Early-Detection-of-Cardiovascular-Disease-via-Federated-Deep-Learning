# run_gru_sweep.py
# Run GRU experiments with top-K feature limits
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


def run_sweep(
    ks: List[Optional[int]],
    rank_path: Optional[str] = None,
) -> pd.DataFrame:

    disease = config.DISEASE
    cfg = TrainConfig()

    rows = []

    for k in ks:
        k_name = "all" if k is None else str(int(k))

        print("#############################")
        print(f"GRU run | top_k={k_name}")
        print(f"rank_path={rank_path}")
        print("#############################")

        out = train_and_eval(disease=disease, cfg=cfg, top_k=k, rank_path=rank_path)

        row = {
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
    ks = [20, 40, 60, 80, 100, 120]
    rank_path = str(config.stability_combined_path(config.DISEASE))
    run_sweep(ks=ks, rank_path=rank_path)