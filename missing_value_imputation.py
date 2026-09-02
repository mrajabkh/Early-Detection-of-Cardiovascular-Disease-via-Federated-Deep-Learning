from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd


ID_COLUMNS = {
    "patientunitstayid",
    "t_end",
    "label",
    "split",
    "t_event",
    "lead_time_mins",
}
TEMPORAL_PREFIXES = ("vp_", "va_", "rch_", "nch_", "rc_", "inf_", "io_")


def is_count_column(column: str) -> bool:
    return column.endswith("_count") or column.endswith("_count_in_window")


def classify_feature_columns(feature_columns: Iterable[str]) -> Dict[str, list[str]]:
    columns = [str(c) for c in feature_columns if str(c) not in ID_COLUMNS]
    lab_columns = [c for c in columns if c.startswith("lab_")]
    if lab_columns:
        preview = ", ".join(lab_columns[:20])
        raise ValueError(
            "Laboratory features are present, so the 120-minute vital-sign carry-forward "
            f"limit has not been applied blindly. Laboratory columns include: {preview}"
        )

    count_columns = [c for c in columns if is_count_column(c)]
    missing_indicator_columns = [c for c in columns if c.endswith("_missing")]
    temporal_value_columns = [
        c
        for c in columns
        if c.startswith(TEMPORAL_PREFIXES)
        and c not in count_columns
        and c not in missing_indicator_columns
    ]
    static_columns = [
        c
        for c in columns
        if c not in temporal_value_columns
        and c not in count_columns
        and c not in missing_indicator_columns
    ]
    return {
        "temporal_value": temporal_value_columns,
        "count": count_columns,
        "missing_indicator": missing_indicator_columns,
        "static": static_columns,
    }


def _set_single_observation_std_to_zero(
    df: pd.DataFrame,
    temporal_columns: Iterable[str],
) -> int:
    changed = 0
    for column in temporal_columns:
        if not column.endswith("_std"):
            continue
        count_column = f"{column[:-4]}_count"
        if count_column not in df.columns:
            continue
        mask = df[column].isna() & (pd.to_numeric(df[count_column], errors="coerce") == 1)
        changed += int(mask.sum())
        if mask.any():
            df.loc[mask, column] = 0.0
    return changed


def causal_forward_fill(
    df: pd.DataFrame,
    temporal_columns: Iterable[str],
    max_age_mins: int = 120,
) -> Tuple[pd.DataFrame, int, int]:
    out = df.sort_values(["patientunitstayid", "t_end"]).copy()
    columns = list(temporal_columns)
    std_zeroed = _set_single_observation_std_to_zero(out, columns)
    forward_filled = 0

    patient_ids = out["patientunitstayid"]
    times = pd.to_numeric(out["t_end"], errors="coerce")
    for column in columns:
        values = pd.to_numeric(out[column], errors="coerce")
        missing_before = values.isna()
        if not missing_before.any():
            continue

        previous_value = values.groupby(patient_ids, sort=False).ffill()
        observed_time = times.where(values.notna())
        previous_time = observed_time.groupby(patient_ids, sort=False).ffill()
        age = times - previous_time
        eligible = missing_before & previous_value.notna() & age.ge(0) & age.le(max_age_mins)
        if eligible.any():
            out.loc[eligible, column] = previous_value.loc[eligible]
            forward_filled += int(eligible.sum())

    return out, forward_filled, std_zeroed


def fit_training_medians(
    df: pd.DataFrame,
    feature_columns: Iterable[str],
) -> Dict[str, float]:
    train = df[df["split"].astype(str).str.lower() == "train"]
    if train.empty:
        raise ValueError("Cannot fit imputation medians: no training rows are available.")

    medians: Dict[str, float] = {}
    all_missing: list[str] = []
    for column in feature_columns:
        values = pd.to_numeric(train[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
        median = values.median(skipna=True)
        if pd.isna(median):
            all_missing.append(str(column))
        else:
            medians[str(column)] = float(median)
    if all_missing:
        raise ValueError(
            "Training medians are undefined for entirely missing feature columns: "
            + ", ".join(all_missing[:20])
        )
    return medians


def apply_training_medians(
    df: pd.DataFrame,
    feature_columns: Iterable[str],
    medians: Dict[str, float],
) -> Tuple[pd.DataFrame, int]:
    out = df.copy()
    median_filled = 0
    for column in feature_columns:
        values = pd.to_numeric(out[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
        missing = values.isna()
        median_filled += int(missing.sum())
        out[column] = values.fillna(float(medians[str(column)]))
    return out, median_filled


def impute_from_training_split(
    df: pd.DataFrame,
    feature_columns: Iterable[str],
    max_age_mins: int = 120,
    fitted_medians: Dict[str, float] | None = None,
) -> Tuple[pd.DataFrame, Dict[str, float], Dict[str, int]]:
    feature_columns = list(feature_columns)
    categories = classify_feature_columns(feature_columns)
    out, forward_filled, std_zeroed = causal_forward_fill(
        df,
        categories["temporal_value"],
        max_age_mins=max_age_mins,
    )
    medians = fitted_medians if fitted_medians is not None else fit_training_medians(out, feature_columns)
    if set(medians) != set(feature_columns):
        raise ValueError("Stored training medians do not match the current feature columns.")
    out, median_filled = apply_training_medians(out, feature_columns, medians)

    values = out[feature_columns].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Non-finite feature values remain after causal/median imputation.")

    return out, medians, {
        "forward_filled": int(forward_filled),
        "single_observation_std_zeroed": int(std_zeroed),
        "median_filled": int(median_filled),
    }


def save_medians(
    path: Path,
    medians: Dict[str, float],
    max_age_mins: int,
    categories: Dict[str, list[str]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "method": "causal_forward_fill_120m_then_train_median",
        "max_forward_fill_age_mins": int(max_age_mins),
        "medians": medians,
        "column_categories": categories,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
