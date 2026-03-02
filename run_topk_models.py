# run_topk_models.py
# Run baseline ML models on Top-K selected features and save results into Outputs/<run_name>/.
# Uses the patient-level split already stored in samples.csv (split column).
# Ranking: stability_combined only (no SHAP fallback).
# Optional: peak RAM via memory_profiler if installed.
#
# NOTE (quick hack): Only outputs TEST split rows (no VAL/TRAIN rows written).
#
# Location: Project/Code/run_topk_models.py

from __future__ import annotations

import time
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

import config


# -----------------------------
# Optional memory_profiler
# -----------------------------
try:
    from memory_profiler import memory_usage  # type: ignore
except Exception:
    memory_usage = None


# -----------------------------
# Model factory (mirrors your old choices)
# -----------------------------
def build_models() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Returns:
      - models_scaled: models that should use scaled inputs
      - models_imputed: models that can run on imputed (unscaled) inputs
    """
    models_scaled: Dict[str, Any] = {}
    models_imputed: Dict[str, Any] = {}

    if config.MODEL_ENABLED.get("log_reg", False):
        models_scaled["Logistic Regression"] = LogisticRegression(
            max_iter=5000,
            class_weight="balanced",
            n_jobs=-1,
        )

    if config.MODEL_ENABLED.get("knn", False):
        models_scaled["kNN"] = KNeighborsClassifier(
            n_neighbors=15,
            weights="distance",
            n_jobs=-1,
        )

    if config.MODEL_ENABLED.get("svm", False):
        models_scaled["SVM"] = SVC(
            kernel="rbf",
            probability=True,
            class_weight="balanced",
        )

    if config.MODEL_ENABLED.get("decision_tree", False):
        models_imputed["Decision Tree"] = DecisionTreeClassifier(
            max_depth=None,
            random_state=int(getattr(config, "SPLIT_RANDOM_STATE", 42)),
            class_weight="balanced",
        )

    if config.MODEL_ENABLED.get("random_forest", False):
        params = dict(getattr(config, "RF_TUNED_PARAMS", {}))
        models_imputed["Random Forest"] = RandomForestClassifier(**params)

    if config.MODEL_ENABLED.get("naive_bayes", False):
        models_imputed["Naive Bayes"] = GaussianNB()

    # Optional third-party models (only if enabled AND installed)
    if config.MODEL_ENABLED.get("xgboost", False):
        try:
            from xgboost import XGBClassifier  # type: ignore
        except Exception:
            print("WARNING: xgboost enabled but not installed. Skipping XGBoost.")
        else:
            models_imputed["XGBoost"] = XGBClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="binary:logistic",
                eval_metric="logloss",
                n_jobs=-1,
                random_state=int(getattr(config, "SPLIT_RANDOM_STATE", 42)),
                tree_method="hist",
            )

    if config.MODEL_ENABLED.get("lightgbm", False):
        try:
            from lightgbm import LGBMClassifier  # type: ignore
        except Exception:
            print("WARNING: lightgbm enabled but not installed. Skipping LightGBM.")
        else:
            params = dict(getattr(config, "LGBM_TUNED_PARAMS", {}))
            params.pop("objective", None)
            models_imputed["LightGBM"] = LGBMClassifier(**params)

    if config.MODEL_ENABLED.get("catboost", False):
        try:
            from catboost import CatBoostClassifier  # type: ignore
        except Exception:
            print("WARNING: catboost enabled but not installed. Skipping CatBoost.")
        else:
            models_imputed["CatBoost"] = CatBoostClassifier(
                iterations=300,
                learning_rate=0.05,
                depth=6,
                loss_function="Logloss",
                eval_metric="Logloss",
                random_seed=int(getattr(config, "SPLIT_RANDOM_STATE", 42)),
                verbose=False,
                thread_count=-1,
                class_weights=None,
            )

    return models_scaled, models_imputed


# -----------------------------
# Ranking loader (stability_combined only)
# -----------------------------
def load_feature_ranking(valid_feature_names: List[str]) -> List[str]:
    feature_set = set(valid_feature_names)

    if not bool(getattr(config, "USE_STABILITY_RANKING", True)):
        raise RuntimeError(
            "USE_STABILITY_RANKING is False, but this script only supports stability_combined ranking."
        )

    stab_path = config.stability_combined_path(config.DISEASE)
    if not stab_path.exists():
        raise FileNotFoundError(f"stability_combined ranking not found: {stab_path}")

    df = pd.read_csv(stab_path)
    if "feature" not in df.columns:
        raise ValueError("stability_combined CSV must contain a 'feature' column.")

    ranked_all = df["feature"].astype(str).tolist()
    ranked = [f for f in ranked_all if f in feature_set]

    if not ranked:
        raise RuntimeError("No overlap between stability_combined features and features.parquet columns.")

    missing = [f for f in ranked_all[:50] if f not in feature_set]
    if missing:
        print("WARNING: rank_path contains features not present in features.parquet")
        print(f"Example missing features (first 20): {missing[:20]}")

    print(f"Using stability_combined ranking: {stab_path}")
    return ranked


# -----------------------------
# Data loading / split building
# -----------------------------
def _load_samples() -> pd.DataFrame:
    samples_path = config.samples_path(config.DISEASE)
    df = pd.read_csv(samples_path)

    need = {"patientunitstayid", "t_end", "label", "split"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"samples.csv missing columns: {sorted(missing)}")

    df["patientunitstayid"] = pd.to_numeric(df["patientunitstayid"], errors="coerce")
    df["t_end"] = pd.to_numeric(df["t_end"], errors="coerce")
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df["split"] = df["split"].astype(str).str.lower()

    df = df.dropna(subset=["patientunitstayid", "t_end", "label", "split"]).copy()
    df["patientunitstayid"] = df["patientunitstayid"].astype(np.int64)
    df["t_end"] = df["t_end"].astype(np.int64)
    df["label"] = df["label"].astype(np.int64)

    return df


def _load_features() -> pd.DataFrame:
    features_path = config.features_path(config.DISEASE)
    df = pd.read_parquet(features_path)

    need = {"patientunitstayid", "t_end"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"features.parquet missing columns: {sorted(missing)}")

    df["patientunitstayid"] = pd.to_numeric(df["patientunitstayid"], errors="coerce")
    df["t_end"] = pd.to_numeric(df["t_end"], errors="coerce")
    df = df.dropna(subset=["patientunitstayid", "t_end"]).copy()
    df["patientunitstayid"] = df["patientunitstayid"].astype(np.int64)
    df["t_end"] = df["t_end"].astype(np.int64)

    return df


def _merge_samples_features(samples: pd.DataFrame, feats: pd.DataFrame) -> pd.DataFrame:
    merged = samples.merge(
        feats,
        on=["patientunitstayid", "t_end"],
        how="left",
        sort=False,
        copy=False,
    )
    return merged


def _get_feature_columns(merged: pd.DataFrame) -> List[str]:
    non_feat = {"patientunitstayid", "t_end", "label", "split", "t_event", "lead_time_mins"}
    feat_cols = [c for c in merged.columns if c not in non_feat]

    # Coerce object cols to numeric (anything not parseable -> NaN)
    for c in feat_cols:
        if merged[c].dtype == object:
            merged[c] = pd.to_numeric(merged[c], errors="coerce")

    return feat_cols


def _build_split_arrays(
    merged: pd.DataFrame,
    feat_cols: List[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    train_df = merged[merged["split"] == "train"].copy()
    val_df = merged[merged["split"] == "val"].copy()
    test_df = merged[merged["split"] == "test"].copy()

    # Quick hack: still requires val to exist because we reuse the existing pipeline,
    # but we won't evaluate on val or output val rows.
    if train_df.empty or val_df.empty or test_df.empty:
        raise RuntimeError("One of train/val/test is empty. Check your samples.csv splits.")

    X_train_df = train_df[feat_cols]
    X_val_df = val_df[feat_cols]
    X_test_df = test_df[feat_cols]

    y_train = train_df["label"].to_numpy(dtype=np.int64)
    y_val = val_df["label"].to_numpy(dtype=np.int64)
    y_test = test_df["label"].to_numpy(dtype=np.int64)

    # Drop columns that are all-null in TRAIN
    all_null = X_train_df.columns[X_train_df.isna().all(axis=0)].tolist()
    if all_null:
        print("#############################")
        print("Dropping all-null TRAIN feature columns")
        print(f"Dropped: {len(all_null)}")
        print(f"Examples: {all_null[:20]}")
        print("#############################")
        keep_cols = [c for c in feat_cols if c not in all_null]
        X_train_df = X_train_df[keep_cols]
        X_val_df = X_val_df[keep_cols]
        X_test_df = X_test_df[keep_cols]
        feat_cols = keep_cols

    # Impute (fit on train only)
    imputer = SimpleImputer(strategy=getattr(config, "IMPUTE_STRATEGY", "median"))
    X_train_imp = imputer.fit_transform(X_train_df).astype(np.float32, copy=False)
    X_val_imp = imputer.transform(X_val_df).astype(np.float32, copy=False)
    X_test_imp = imputer.transform(X_test_df).astype(np.float32, copy=False)

    # Scale (fit on train only)
    scaler = StandardScaler(with_mean=True, with_std=True)
    X_train_scl = scaler.fit_transform(X_train_imp).astype(np.float32, copy=False)
    X_val_scl = scaler.transform(X_val_imp).astype(np.float32, copy=False)
    X_test_scl = scaler.transform(X_test_imp).astype(np.float32, copy=False)

    return (
        X_train_imp, y_train,
        X_val_imp, y_val,
        X_test_imp, y_test,
        X_train_scl, X_val_scl, X_test_scl,
        feat_cols,
    )


def _make_feature_index_map(feature_names: List[str]) -> Dict[str, int]:
    return {f: i for i, f in enumerate(feature_names)}


# -----------------------------
# Evaluation
# -----------------------------
def _fit_predict_metrics(
    model: Any,
    X_train: np.ndarray,
    X_eval: np.ndarray,
    y_train: np.ndarray,
    y_eval: np.ndarray,
) -> Dict[str, Any]:
    start_time = time.time()

    model.fit(X_train, y_train)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_eval)[:, 1]
    else:
        # decision_function -> minmax to [0,1]
        y_dec = model.decision_function(X_eval)
        y_prob = (y_dec - y_dec.min()) / (y_dec.max() - y_dec.min() + 1e-9)

    y_pred = (y_prob >= 0.5).astype(int)

    end_time = time.time()

    # Metrics (guard if eval contains only one class)
    if len(np.unique(y_eval)) < 2:
        auroc = float("nan")
        auprc = float("nan")
    else:
        auroc = float(roc_auc_score(y_eval, y_prob))
        auprc = float(average_precision_score(y_eval, y_prob))

    accuracy = float(accuracy_score(y_eval, y_pred))
    precision = float(precision_score(y_eval, y_pred, zero_division=0))
    recall = float(recall_score(y_eval, y_pred, zero_division=0))
    f1 = float(f1_score(y_eval, y_pred, zero_division=0))

    tn, fp, fn, tp = confusion_matrix(y_eval, y_pred).ravel()

    return {
        "AUROC": float(auroc),
        "AUPRC": float(auprc),
        "Accuracy": float(accuracy),
        "Precision": float(precision),
        "Recall": float(recall),
        "F1": float(f1),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "Time_seconds": float(end_time - start_time),
    }


def evaluate_with_peak_ram(
    model: Any,
    X_train: np.ndarray,
    X_eval: np.ndarray,
    y_train: np.ndarray,
    y_eval: np.ndarray,
    interval_sec: float = 0.1,
) -> Dict[str, Any]:
    if memory_usage is None:
        out = _fit_predict_metrics(model, X_train, X_eval, y_train, y_eval)
        out["Peak_RAM_MB"] = float("nan")
        return out

    def _runner():
        return _fit_predict_metrics(model, X_train, X_eval, y_train, y_eval)

    mem_trace, result = memory_usage(
        (_runner, (), {}),
        interval=interval_sec,
        retval=True,
        max_usage=False,
    )

    peak_mb = float(np.max(mem_trace)) if len(mem_trace) else float("nan")
    result["Peak_RAM_MB"] = peak_mb
    return result


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    features_path = config.features_path(config.DISEASE)
    samples_path = config.samples_path(config.DISEASE)
    out_results = config.results_path(config.DISEASE)

    print("#############################")
    print("Top-K model runs (uses existing samples split)")
    print("#############################")
    print(f"Features: {features_path}")
    print(f"Samples:  {samples_path}")
    print(f"Output:   {out_results}")

    samples_df = _load_samples()
    feats_df = _load_features()
    merged = _merge_samples_features(samples_df, feats_df)

    feat_cols = _get_feature_columns(merged)

    (
        X_train_imp, y_train,
        X_val_imp, y_val,
        X_test_imp, y_test,
        X_train_scl, X_val_scl, X_test_scl,
        valid_feature_names,
    ) = _build_split_arrays(merged, feat_cols)

    print("#############################")
    print(f"Train: n={len(y_train)} pos={int(y_train.sum())} neg={int((y_train==0).sum())}")
    print(f"Val:   n={len(y_val)} pos={int(y_val.sum())} neg={int((y_val==0).sum())}")
    print(f"Test:  n={len(y_test)} pos={int(y_test.sum())} neg={int((y_test==0).sum())}")
    print(f"Valid features: {len(valid_feature_names)}")
    print("#############################")

    feature_index_map = _make_feature_index_map(valid_feature_names)
    ranked_features = load_feature_ranking(valid_feature_names)

    models_scaled, models_imputed = build_models()
    if not models_scaled and not models_imputed:
        raise RuntimeError("No models enabled in config.MODEL_ENABLED.")

    all_results: List[Dict[str, Any]] = []

    topk_list = list(getattr(config, "TOPK_LIST", [20, 40, 60, 80, 100]))
    for topk in topk_list:
        K = min(int(topk), len(ranked_features))
        top_k_features = ranked_features[:K]
        top_k_indices = [feature_index_map[f] for f in top_k_features]

        # Slice arrays
        Xtr_imp_k = X_train_imp[:, top_k_indices]
        Xte_imp_k = X_test_imp[:, top_k_indices]

        Xtr_scl_k = X_train_scl[:, top_k_indices]
        Xte_scl_k = X_test_scl[:, top_k_indices]

        print("#############################")
        print(f"Top-K = {K}")
        print(f"Train (imp): {Xtr_imp_k.shape}, Test (imp): {Xte_imp_k.shape}")
        print(f"Train (scl): {Xtr_scl_k.shape}, Test (scl): {Xte_scl_k.shape}")
        print("#############################")

        # Evaluate each model ONLY on TEST (train is used only for fitting)
        for name, model in models_scaled.items():
            full_name_test = f"{name} (top-{K})"
            print(f"Running (TEST): {full_name_test}")
            metrics_test = evaluate_with_peak_ram(
                model=model,
                X_train=Xtr_scl_k,
                X_eval=Xte_scl_k,
                y_train=y_train,
                y_eval=y_test,
                interval_sec=0.1,
            )
            all_results.append(
                {
                    "Model": full_name_test,
                    "TopK": K,
                    "Split": "test",
                    **metrics_test,
                }
            )

        for name, model in models_imputed.items():
            full_name_test = f"{name} (top-{K})"
            print(f"Running (TEST): {full_name_test}")
            metrics_test = evaluate_with_peak_ram(
                model=model,
                X_train=Xtr_imp_k,
                X_eval=Xte_imp_k,
                y_train=y_train,
                y_eval=y_test,
                interval_sec=0.1,
            )
            all_results.append(
                {
                    "Model": full_name_test,
                    "TopK": K,
                    "Split": "test",
                    **metrics_test,
                }
            )

    results_df = pd.DataFrame(all_results)

    # Sort: TopK then AUROC desc (Split is always test now)
    results_df = results_df.sort_values(by=["TopK", "AUROC"], ascending=[True, False])

    # Round numeric metrics to 3 d.p. (keep confusion matrix counts as ints)
    float_cols = results_df.select_dtypes(include=[np.number]).columns.tolist()
    int_like_cols = {"TopK", "TN", "FP", "FN", "TP"}
    round_cols = [c for c in float_cols if c not in int_like_cols]
    for c in round_cols:
        results_df[c] = results_df[c].astype(float).round(3)

    for c in ["TopK", "TN", "FP", "FN", "TP"]:
        if c in results_df.columns:
            results_df[c] = results_df[c].astype(int)

    out_results.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(out_results, index=False, float_format="%.3f")

    print("#############################")
    print(f"Saved results to: {out_results}")
    print("#############################")
    print(results_df.head(12).to_string(index=False))


if __name__ == "__main__":
    main()