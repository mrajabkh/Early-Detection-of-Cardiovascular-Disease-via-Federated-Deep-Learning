# xai.py
# Explainability tool for the GRU model.
#
# What it does:
# - loads the saved GRU checkpoint, metadata, and test_predictions.csv
# - lets you choose a patient by TP/TN/FP/FN, positive/negative, or exact patient id
# - prints patient-level prediction summary
# - computes SHAP values over the full sequence
# - prints top feature-time SHAP contributions
# - saves:
#     1) temporal SHAP line plot
#     2) timestamp occlusion line plot
#     3) global top-15 SHAP beeswarm plot
#     4) global top-15 SHAP bar plot
#     5) grouped SHAP bar plot
#     6) full feature ranking CSV
#     7) variable ranking CSV
#     8) grouped ranking CSV
#
# Notes:
# - SHAP explains a patient-level max-risk score:
#     max valid timestep logit over the sequence
# - patient-level prediction summary still uses max valid timestep probability
# - padded rows are assumed to be all zeros after normalization
# - for SHAP wrapper length inference, valid timesteps are inferred from non-zero rows
# - global beeswarm uses signed SHAP values and aggregated feature values
# - global feature ranking uses mean absolute SHAP

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import config
from gru_model import GRURisk
from sequence_dataset_gru import PatientSequenceDataset
from trgru_model import TrGRURisk

try:
    import shap
except Exception as e:
    shap = None
    _shap_import_error = e
else:
    _shap_import_error = None


#############################
# Display names and grouping
#############################
FEATURE_DISPLAY_NAMES = {
    "rch_fio2_max": "FiO2 (max)",
    "rch_fio2_mean": "FiO2 (mean)",
    "rch_fio2_min": "FiO2 (min)",
    "rch_fio2_last": "FiO2 (last)",
    "pt_admissionweight": "Admission weight",
    "pt_age": "Age",
    "pt_admissionheight": "Admission height",
    "lab_labresult_last": "Lab result (last)",
    "lab_labresult_max": "Lab result (max)",
    "lab_labresult_mean": "Lab result (mean)",
    "lab_labresult_min": "Lab result (min)",
    "apv_bedcount": "Bed count",
    "vp_temperature_last": "Temperature (last)",
    "vp_temperature_mean": "Temperature (mean)",
    "vp_temperature_max": "Temperature (max)",
    "vp_temperature_min": "Temperature (min)",
    "nch_temperature_last": "Temperature (nurse-charted, last)",
    "nch_temperature_mean": "Temperature (nurse-charted, mean)",
    "nch_temperature_min": "Temperature (nurse-charted, min)",
    "nch_temperature_max": "Temperature (nurse-charted, max)",
    "vp_sao2_min": "SpO2 (min)",
    "vp_sao2_mean": "SpO2 (mean)",
    "vp_sao2_last": "SpO2 (last)",
    "vp_sao2_max": "SpO2 (max)",
    "va_noninvasivesystolic_mean": "SBP (mean)",
    "va_noninvasivesystolic_max": "SBP (max)",
    "va_noninvasivesystolic_min": "SBP (min)",
    "va_noninvasivesystolic_last": "SBP (last)",
    "nch_non_invasive_bp_last": "SBP (nurse-charted, last)",
    "va_noninvasivemean_last": "MAP (last)",
    "io_nettotal_last": "Net fluid balance (last)",
    "io_nettotal_min": "Net fluid balance (min)",
    "io_nettotal_max": "Net fluid balance (max)",
    "io_nettotal_mean": "Net fluid balance (mean)",
    "nch_o2_l_max": "O2 flow (max)",
    "nch_o2_l_mean": "O2 flow (mean)",
    "nch_o2_l_min": "O2 flow (min)",
    "io_outputtotal_max": "Output total (max)",
    "io_outputtotal_last": "Output total (last)",
    "io_outputtotal_mean": "Output total (mean)",
    "vp_heartrate_last": "Heart rate (last)",
    "vp_heartrate_mean": "Heart rate (mean)",
    "vp_heartrate_min": "Heart rate (min)",
    "vp_heartrate_max": "Heart rate (max)",
    "nch_heart_rate_last": "Heart rate (nurse-charted, last)",
    "nch_heart_rate_mean": "Heart rate (nurse-charted, mean)",
    "nch_heart_rate_min": "Heart rate (nurse-charted, min)",
    "nch_heart_rate_max": "Heart rate (nurse-charted, max)",
    "pt_unitvisitnumber": "Unit visit number",
    "nch_glasgow_coma_score_last": "GCS (last)",
    "rch_peep_count": "PEEP count",
    "vp_respiration_min": "Respiratory rate (min)",
    "vp_respiration_last": "Respiratory rate (last)",
    "vp_respiration_mean": "Respiratory rate (mean)",
    "vp_respiration_max": "Respiratory rate (max)",
    "io_cellvaluenumeric_last": "Cell value numeric (last)",
    "nch_pain_score_goal_count": "Pain score goal count",
    "drug_vasopressor_any": "Vasopressor used",
    "med_count_in_window": "Medication count",
    "treatment_count_in_window": "Treatment count",
}

FEATURE_TO_VARIABLE = {
    "rch_fio2_max": "FiO2",
    "rch_fio2_mean": "FiO2",
    "rch_fio2_min": "FiO2",
    "rch_fio2_last": "FiO2",
    "pt_admissionweight": "Admission weight",
    "pt_age": "Age",
    "pt_admissionheight": "Admission height",
    "pt_unitvisitnumber": "Unit visit number",
    "apv_bedcount": "Bed count",
    "lab_labresult_last": "Lab result",
    "lab_labresult_max": "Lab result",
    "lab_labresult_mean": "Lab result",
    "lab_labresult_min": "Lab result",
    "io_cellvaluenumeric_last": "Cell value numeric",
    "vp_temperature_last": "Temperature",
    "vp_temperature_mean": "Temperature",
    "vp_temperature_max": "Temperature",
    "vp_temperature_min": "Temperature",
    "nch_temperature_last": "Temperature",
    "nch_temperature_mean": "Temperature",
    "nch_temperature_min": "Temperature",
    "nch_temperature_max": "Temperature",
    "vp_sao2_min": "SpO2",
    "vp_sao2_mean": "SpO2",
    "vp_sao2_last": "SpO2",
    "vp_sao2_max": "SpO2",
    "va_noninvasivesystolic_mean": "Systolic blood pressure",
    "va_noninvasivesystolic_max": "Systolic blood pressure",
    "va_noninvasivesystolic_min": "Systolic blood pressure",
    "va_noninvasivesystolic_last": "Systolic blood pressure",
    "nch_non_invasive_bp_last": "Systolic blood pressure",
    "va_noninvasivemean_last": "Mean arterial pressure",
    "io_nettotal_last": "Net fluid balance",
    "io_nettotal_min": "Net fluid balance",
    "io_nettotal_max": "Net fluid balance",
    "io_nettotal_mean": "Net fluid balance",
    "io_outputtotal_last": "Output total",
    "io_outputtotal_max": "Output total",
    "io_outputtotal_mean": "Output total",
    "nch_o2_l_max": "O2 flow",
    "nch_o2_l_mean": "O2 flow",
    "nch_o2_l_min": "O2 flow",
    "vp_heartrate_last": "Heart rate",
    "vp_heartrate_mean": "Heart rate",
    "vp_heartrate_min": "Heart rate",
    "vp_heartrate_max": "Heart rate",
    "nch_heart_rate_last": "Heart rate",
    "nch_heart_rate_mean": "Heart rate",
    "nch_heart_rate_min": "Heart rate",
    "nch_heart_rate_max": "Heart rate",
    "nch_glasgow_coma_score_last": "Glasgow Coma Scale",
    "nch_pain_score_goal_count": "Pain score goal",
    "rch_peep_count": "PEEP",
    "vp_respiration_min": "Respiratory rate",
    "vp_respiration_last": "Respiratory rate",
    "vp_respiration_mean": "Respiratory rate",
    "vp_respiration_max": "Respiratory rate",
    "drug_vasopressor_any": "EXCLUDE",
    "med_count_in_window": "EXCLUDE",
    "treatment_count_in_window": "EXCLUDE",
}

VARIABLE_TO_GROUP = {
    "Age": "Baseline and admission",
    "Admission weight": "Baseline and admission",
    "Admission height": "Baseline and admission",
    "Unit visit number": "Baseline and admission",
    "Bed count": "Baseline and admission",
    "Heart rate": "Cardiovascular",
    "Systolic blood pressure": "Cardiovascular",
    "Diastolic blood pressure": "Cardiovascular",
    "Mean arterial pressure": "Cardiovascular",
    "Central venous pressure": "Cardiovascular",
    "Arterial systolic pressure": "Cardiovascular",
    "Arterial diastolic pressure": "Cardiovascular",
    "Pulmonary artery systolic pressure": "Cardiovascular",
    "Pulmonary artery diastolic pressure": "Cardiovascular",
    "FiO2": "Respiratory and oxygenation",
    "SpO2": "Respiratory and oxygenation",
    "O2 flow": "Respiratory and oxygenation",
    "PEEP": "Respiratory and oxygenation",
    "Respiratory rate": "Respiratory and oxygenation",
    "End-tidal CO2": "Respiratory and oxygenation",
    "Temperature": "Temperature",
    "Glasgow Coma Scale": "Neurological",
    "Pain score goal": "Neurological",
    "Lab result": "Labs",
    "Cell value numeric": "Labs",
    "Net fluid balance": "Fluids and output",
    "Output total": "Fluids and output",
}

# Prefix rules make grouping cover every derived statistic belonging to a
# physiological variable, including min/max/mean/std/count/last and the
# corresponding missingness indicators. Exact mappings above remain useful
# for non-family features and human-readable labels.
FEATURE_PREFIX_TO_VARIABLE = {
    "vp_heartrate_": "Heart rate",
    "vp_cvp_": "Central venous pressure",
    "vp_systemicsystolic_": "Arterial systolic pressure",
    "vp_systemicdiastolic_": "Arterial diastolic pressure",
    "vp_systemicmean_": "Mean arterial pressure",
    "vp_pasystolic_": "Pulmonary artery systolic pressure",
    "vp_padiastolic_": "Pulmonary artery diastolic pressure",
    "va_noninvasivesystolic_": "Systolic blood pressure",
    "va_noninvasivediastolic_": "Diastolic blood pressure",
    "va_noninvasivemean_": "Mean arterial pressure",
    "vp_sao2_": "SpO2",
    "vp_respiration_": "Respiratory rate",
    "vp_etco2_": "End-tidal CO2",
    "rch_fio2_": "FiO2",
    "rch_peep_": "PEEP",
    "nch_o2_l_": "O2 flow",
    "vp_temperature_": "Temperature",
    "nch_temperature_": "Temperature",
}


#############################
# Helpers
#############################
def _feature_mode() -> str:
    return str(getattr(config, "FEATURE_MODE", "all")).strip().lower()


def _feature_mode_tag() -> str:
    return _feature_mode().replace("+", "_")


def _run_tag_from_metadata(meta: Dict) -> str:
    feature_mode = str(meta.get("feature_mode", _feature_mode())).strip().lower()
    top_k = meta.get("top_k", None)
    if feature_mode == "all":
        return "all" if top_k is None else f"topk{int(top_k)}"
    return f"feat{feature_mode.replace('+', '_')}"


def _choose_model_type() -> Optional[str]:
    print()
    print("Choose explainability model type:")
    print("1) Centralized")
    print("2) Federated")
    print("3) Quit")

    choice = input("Selection: ").strip()
    if choice == "1":
        return "centralized"
    if choice == "2":
        return "federated"
    if choice == "3":
        return None

    print("Invalid selection.")
    return None


def _infer_artifact_paths(disease: config.DiseaseSpec, model_type: str = "centralized") -> Tuple[Path, Path, Path, Path]:
    out_dir = config.run_dir(disease)
    if model_type == "federated":
        artifact_root = out_dir / "Federated"
    else:
        artifact_root = out_dir

    if not artifact_root.exists():
        raise FileNotFoundError(f"Run directory does not exist: {artifact_root}")

    metadata_pattern = "metadata__fedavg__*.json" if model_type == "federated" else "metadata__*.json"
    metadata_files = sorted(artifact_root.glob(metadata_pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    if not metadata_files:
        raise FileNotFoundError(f"No {metadata_pattern} found in {artifact_root}")

    metadata_path = metadata_files[0]
    with open(metadata_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    run_tag = _run_tag_from_metadata(meta)
    model_tag = f"fedavg__{run_tag}" if model_type == "federated" else run_tag

    model_path = artifact_root / f"model__{model_tag}.pt"
    preds_path = artifact_root / f"test_predictions__{model_tag}.csv"

    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    if not preds_path.exists():
        raise FileNotFoundError(f"Predictions CSV not found: {preds_path}")

    xai_dir = artifact_root / "XAI"
    xai_dir.mkdir(parents=True, exist_ok=True)

    return model_path, metadata_path, preds_path, xai_dir


def _load_checkpoint_and_metadata(model_path: Path, metadata_path: Path, device: str):
    ckpt = torch.load(model_path, map_location=device)
    with open(metadata_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return ckpt, meta


def _build_model_from_checkpoint(ckpt: Dict, device: str) -> GRURisk:
    model = GRURisk(
        input_dim=int(ckpt["input_dim"]),
        hidden_dim=int(ckpt["hidden_dim"]),
        num_layers=int(ckpt["num_layers"]),
        dropout=float(ckpt["dropout"]),
        use_layernorm=bool(ckpt.get("use_layernorm", True)),
        use_attention_pooling=bool(ckpt.get("use_attention_pooling", True)),
    ).to(device)

    state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()
    return model


def _get_test_dataset(disease: config.DiseaseSpec, meta: Dict, samples_path: Optional[str] = None) -> PatientSequenceDataset:
    feature_mode = str(meta.get("feature_mode", _feature_mode())).strip().lower()
    top_k = meta.get("top_k", None) if feature_mode == "all" else None
    rank_path = str(config.stability_combined_path(disease)) if feature_mode == "all" else None
    return PatientSequenceDataset(
        split="test",
        disease=disease,
        max_len=int(meta["max_len"]),
        seed=int(meta.get("seed", 42)),
        normalize=True,
        top_k=top_k,
        rank_path=rank_path,
        samples_path=samples_path,
    )


def _get_train_dataset(disease: config.DiseaseSpec, meta: Dict, samples_path: Optional[str] = None) -> PatientSequenceDataset:
    feature_mode = str(meta.get("feature_mode", _feature_mode())).strip().lower()
    top_k = meta.get("top_k", None) if feature_mode == "all" else None
    rank_path = str(config.stability_combined_path(disease)) if feature_mode == "all" else None
    return PatientSequenceDataset(
        split="train",
        disease=disease,
        max_len=int(meta["max_len"]),
        seed=int(meta.get("seed", 42)),
        normalize=True,
        top_k=top_k,
        rank_path=rank_path,
        samples_path=samples_path,
    )


def _dataset_pid_to_index(ds: PatientSequenceDataset) -> Dict[int, int]:
    return {int(pid): i for i, pid in enumerate(ds.pids.tolist())}


def _pad_sequence_to_max_len(x: torch.Tensor, max_len: int) -> torch.Tensor:
    t, d = x.shape
    if t > max_len:
        x = x[-max_len:, :]
        t = x.shape[0]
    out = torch.zeros((max_len, d), dtype=torch.float32)
    out[:t, :] = x
    return out


def _row_activity_lengths(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    active = x.abs().sum(dim=2) > eps
    lengths = active.sum(dim=1).clamp(min=1).long()
    return lengths


def _time_label(t_idx: int, valid_len: int) -> str:
    step_hours = int(getattr(config, "STEP_HOURS", 1))
    hours_before = (valid_len - 1 - t_idx) * step_hours
    return f"-{hours_before}h"


def _format_top_pairs(
    shap_values_tf: np.ndarray,
    feature_names: List[str],
    valid_len: int,
    top_n: int = 10,
) -> List[Tuple[str, float]]:
    pairs: List[Tuple[str, float]] = []

    for t in range(valid_len):
        for f, feat in enumerate(feature_names):
            val = float(shap_values_tf[t, f])
            label = f"{feat} @ {_time_label(t, valid_len)}"
            pairs.append((label, val))

    pairs.sort(key=lambda x: abs(x[1]), reverse=True)
    return pairs[:top_n]


def _coerce_shap_values(shap_values) -> np.ndarray:
    vals = shap_values
    if isinstance(vals, list):
        vals = vals[0]
    vals = np.asarray(vals)

    while vals.ndim > 3 and vals.shape[-1] == 1:
        vals = vals[..., 0]

    if vals.ndim == 4 and vals.shape[1] == 1:
        vals = vals[:, 0, :, :]

    if vals.ndim != 3:
        raise ValueError(f"Unexpected SHAP output shape: {vals.shape}")

    return vals


def _hours_span_from_length(length: int) -> int:
    step_hours = int(getattr(config, "STEP_HOURS", 1))
    return int(length) * step_hours


def _positive_label_timestep_labels(y_seq: torch.Tensor, length: int) -> List[str]:
    y_np = y_seq[:length].detach().cpu().numpy().astype(np.int64)
    out = []
    for t, val in enumerate(y_np):
        if int(val) == 1:
            out.append(_time_label(t, length))
    return out


def _safe_float_or_none(series: pd.Series):
    vals = pd.to_numeric(series, errors="coerce").dropna()
    if len(vals) == 0:
        return None
    return float(vals.iloc[0])


def _safe_minmax(series: pd.Series) -> Tuple[Optional[float], Optional[float]]:
    vals = pd.to_numeric(series, errors="coerce").dropna()
    if len(vals) == 0:
        return None, None
    return float(vals.min()), float(vals.max())


def _patient_event_info(ds: PatientSequenceDataset, patient_id: int) -> Dict[str, object]:
    sub = ds.df.loc[ds.df["patientunitstayid"] == int(patient_id)].copy()
    if sub.empty:
        return {
            "final_t_end": None,
            "t_event": None,
            "time_to_event_from_final_t_end": None,
            "lead_time_min": None,
            "lead_time_max": None,
        }

    sub = sub.sort_values("t_end").reset_index(drop=True)

    final_t_end = _safe_float_or_none(pd.Series([sub["t_end"].iloc[-1]]))

    t_event = None
    if "t_event" in sub.columns:
        vals = pd.to_numeric(sub["t_event"], errors="coerce").dropna()
        if len(vals) > 0:
            unique_vals = vals.unique().tolist()
            t_event = float(unique_vals[0])

    time_to_event = None
    if t_event is not None and final_t_end is not None:
        time_to_event = float(t_event - final_t_end)

    lead_min = None
    lead_max = None
    if "lead_time_mins" in sub.columns:
        lead_min, lead_max = _safe_minmax(sub["lead_time_mins"])

    return {
        "final_t_end": final_t_end,
        "t_event": t_event,
        "time_to_event_from_final_t_end": time_to_event,
        "lead_time_min": lead_min,
        "lead_time_max": lead_max,
    }


def _display_name(feature: str) -> str:
    return FEATURE_DISPLAY_NAMES.get(feature, feature)


def _aggregate_patient_shap_over_time_signed(
    shap_values_tf: np.ndarray,
    valid_len: int,
) -> np.ndarray:
    return shap_values_tf[:valid_len, :].mean(axis=0)


def _aggregate_patient_feature_values_over_time(
    x_seq: torch.Tensor,
    valid_len: int,
) -> np.ndarray:
    return x_seq[:valid_len, :].detach().cpu().numpy().mean(axis=0)


def _build_global_summary_matrices(
    wrapper: nn.Module,
    background_x: torch.Tensor,
    test_ds: PatientSequenceDataset,
    max_len: int,
    device: str,
    max_patients: Optional[int] = None,
    batch_size: int = 16,
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    shap_rows: List[np.ndarray] = []
    feature_rows: List[np.ndarray] = []
    patient_ids: List[int] = []

    n = len(test_ds) if max_patients is None else min(len(test_ds), int(max_patients))

    if shap is None:
        raise ImportError(f"The shap package failed to import: {repr(_shap_import_error)}")
    explainer = shap.GradientExplainer(wrapper, background_x)

    for start in range(0, n, batch_size):
        records = [test_ds[idx] for idx in range(start, min(start + batch_size, n))]
        padded = torch.stack(
            [_pad_sequence_to_max_len(record[0].float(), max_len=max_len) for record in records]
        ).to(device)
        batch_values = _coerce_shap_values(explainer.shap_values(padded))

        for record, shap_values_tf in zip(records, batch_values):
            x_seq, _, length_t, pid_t = record
            length = int(length_t.item())
            shap_rows.append(_aggregate_patient_shap_over_time_signed(shap_values_tf, length))
            feature_rows.append(_aggregate_patient_feature_values_over_time(x_seq.float(), length))
            patient_ids.append(int(pid_t.item()))

    if not shap_rows:
        raise ValueError("No test patients were available for global SHAP summary generation.")

    shap_mat = np.vstack(shap_rows).astype(np.float64)
    feature_mat = np.vstack(feature_rows).astype(np.float64)
    return shap_mat, feature_mat, patient_ids


def _rank_features(global_shap_signed_matrix: np.ndarray, feature_names: List[str]) -> pd.DataFrame:
    mean_abs = np.abs(global_shap_signed_matrix).mean(axis=0)
    mean_signed = global_shap_signed_matrix.mean(axis=0)

    df = pd.DataFrame(
        {
            "feature": feature_names,
            "display_name": [_display_name(f) for f in feature_names],
            "mean_abs_shap": mean_abs,
            "mean_signed_shap": mean_signed,
        }
    )
    df = df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    return df


def _build_variable_importance_df(feature_ranking_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for _, r in feature_ranking_df.iterrows():
        feature = str(r["feature"])
        variable = FEATURE_TO_VARIABLE.get(feature, None)
        if variable is None:
            variable = next(
                (
                    mapped_variable
                    for prefix, mapped_variable in FEATURE_PREFIX_TO_VARIABLE.items()
                    if feature.startswith(prefix)
                ),
                None,
            )

        if variable is None or variable == "EXCLUDE":
            continue

        group = VARIABLE_TO_GROUP.get(variable, None)
        if group is None:
            continue

        rows.append(
            {
                "feature": feature,
                "variable": variable,
                "group": group,
                "mean_abs_shap": float(r["mean_abs_shap"]),
                "mean_signed_shap": float(r["mean_signed_shap"]),
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=["variable", "group", "mean_abs_shap", "mean_signed_shap", "n_features"]
        )

    df = pd.DataFrame(rows)

    variable_df = (
        df.groupby(["variable", "group"], as_index=False)
        .agg(
            mean_abs_shap=("mean_abs_shap", "sum"),
            mean_signed_shap=("mean_signed_shap", "sum"),
            n_features=("feature", "count"),
        )
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    return variable_df


def _group_shap_importance(variable_importance_df: pd.DataFrame) -> pd.DataFrame:
    if variable_importance_df.empty:
        return pd.DataFrame(columns=["group", "group_mean_variable_importance", "n_variables"])

    grouped = (
        variable_importance_df.groupby("group", as_index=False)
        .agg(
            group_mean_variable_importance=("mean_abs_shap", "mean"),
            n_variables=("variable", "count"),
        )
        .sort_values("group_mean_variable_importance", ascending=False)
        .reset_index(drop=True)
    )
    return grouped


#############################
# Model wrappers
#############################
def _model_forward_logits_ts(model_out) -> torch.Tensor:
    if isinstance(model_out, dict):
        return model_out["logits_ts"]
    return model_out


class PatientMaxRiskWrapper(nn.Module):
    """
    Wraps the GRU model so SHAP explains a patient-level score:
    the max valid timestep logit over the sequence.
    """

    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.base_model = base_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lengths = _row_activity_lengths(x)
        out = self.base_model(x, lengths)
        logits_ts = _model_forward_logits_ts(out)

        bsz, t_max = logits_ts.shape
        idx = torch.arange(t_max, device=x.device)[None, :]
        valid = idx < lengths[:, None]

        neg_inf = torch.full_like(logits_ts, -1e9)
        logits_masked = torch.where(valid, logits_ts, neg_inf)

        score = logits_masked.max(dim=1).values.unsqueeze(1)
        return score


#############################
# Patient summary
#############################
@torch.no_grad()
def _predict_patient_summary(
    model: nn.Module,
    x_seq: torch.Tensor,
    y_seq: torch.Tensor,
    length: int,
    threshold: float,
    device: str,
) -> Dict[str, float | int | str]:
    x = x_seq.unsqueeze(0).to(device)
    lengths = torch.tensor([int(length)], dtype=torch.long)

    out = model(x, lengths)
    logits_ts = _model_forward_logits_ts(out)[0, :length]
    probs_ts = torch.sigmoid(logits_ts)

    probs_np = probs_ts.detach().cpu().numpy().astype(np.float64)
    y_np = y_seq[:length].detach().cpu().numpy().astype(np.int64)

    pred_prob = float(probs_np.max())
    peak_idx = int(np.argmax(probs_np))
    true_label = int((y_np == 1).any())
    pred_label = int(pred_prob >= float(threshold))

    if true_label == 1 and pred_label == 1:
        outcome_type = "TP"
    elif true_label == 0 and pred_label == 0:
        outcome_type = "TN"
    elif true_label == 0 and pred_label == 1:
        outcome_type = "FP"
    else:
        outcome_type = "FN"

    return {
        "true_label": true_label,
        "pred_prob": pred_prob,
        "pred_label": pred_label,
        "outcome_type": outcome_type,
        "peak_timestep_idx": peak_idx,
        "peak_timestep_label": _time_label(peak_idx, int(length)),
        "threshold": float(threshold),
    }


#############################
# Background + SHAP
#############################
def _build_background_tensor(
    train_ds: PatientSequenceDataset,
    max_len: int,
    n_background: int,
    seed: int,
) -> torch.Tensor:
    rng = random.Random(seed)
    n = len(train_ds)
    take = min(n_background, n)

    idxs = list(range(n))
    rng.shuffle(idxs)
    idxs = idxs[:take]

    xs = []
    for idx in idxs:
        x_seq, y_seq, length, pid = train_ds[idx]
        x_pad = _pad_sequence_to_max_len(x_seq.float(), max_len=max_len)
        xs.append(x_pad)

    if not xs:
        raise ValueError("No background samples available.")

    return torch.stack(xs, dim=0)


def _compute_shap_values(
    wrapper: nn.Module,
    background_x: torch.Tensor,
    patient_x: torch.Tensor,
) -> np.ndarray:
    if shap is None:
        raise ImportError(
            "The shap package is not installed or failed to import. "
            f"Original error: {repr(_shap_import_error)}"
        )

    explainer = shap.GradientExplainer(wrapper, background_x)
    shap_values = explainer.shap_values(patient_x)
    shap_values = _coerce_shap_values(shap_values)
    return shap_values[0]


#############################
# Temporal occlusion
#############################
@torch.no_grad()
def _temporal_occlusion_importance(
    model: nn.Module,
    x_seq: torch.Tensor,
    length: int,
    device: str,
) -> np.ndarray:
    x = x_seq.clone().float()
    base_summary = _predict_patient_summary(
        model=model,
        x_seq=x,
        y_seq=torch.zeros(length, dtype=torch.long),
        length=length,
        threshold=0.5,
        device=device,
    )
    base_prob = float(base_summary["pred_prob"])

    scores = []
    for t in range(length):
        x_masked = x.clone()
        x_masked[t, :] = 0.0

        masked_summary = _predict_patient_summary(
            model=model,
            x_seq=x_masked,
            y_seq=torch.zeros(length, dtype=torch.long),
            length=length,
            threshold=0.5,
            device=device,
        )
        masked_prob = float(masked_summary["pred_prob"])
        scores.append(base_prob - masked_prob)

    return np.asarray(scores, dtype=np.float64)


#############################
# Plotting
#############################
def _plot_temporal_shap(
    shap_values_tf: np.ndarray,
    valid_len: int,
    patient_id: int,
    xai_dir: Path,
) -> Path:
    temporal = np.abs(shap_values_tf[:valid_len, :]).sum(axis=1)
    labels = [_time_label(t, valid_len) for t in range(valid_len)]

    save_path = xai_dir / f"temporal_shap__patient_{patient_id}.png"

    plt.figure(figsize=(9, 4.5))
    plt.plot(range(valid_len), temporal)
    plt.xticks(range(valid_len), labels, rotation=45)
    plt.xlabel("Time before endpoint")
    plt.ylabel("Mean absolute SHAP")
    plt.title(f"Temporal SHAP importance: patient {patient_id}")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

    return save_path


def _plot_temporal_occlusion(
    occlusion_scores: np.ndarray,
    valid_len: int,
    patient_id: int,
    xai_dir: Path,
) -> Path:
    labels = [_time_label(t, valid_len) for t in range(valid_len)]
    save_path = xai_dir / f"temporal_occlusion__patient_{patient_id}.png"

    plt.figure(figsize=(9, 4.5))
    plt.plot(range(valid_len), occlusion_scores)
    plt.xticks(range(valid_len), labels, rotation=45)
    plt.xlabel("Time before endpoint")
    plt.ylabel("Delta in max predicted risk when masked")
    plt.title(f"Timestamp occlusion importance: patient {patient_id}")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

    return save_path


def _plot_global_shap_beeswarm(
    global_shap_signed_matrix: np.ndarray,
    global_feature_value_matrix: np.ndarray,
    feature_names: List[str],
    xai_dir: Path,
    top_n: int = 15,
    model_label: str = "GRU",
) -> Path:
    mean_abs = np.abs(global_shap_signed_matrix).mean(axis=0)
    top_idx = np.argsort(mean_abs)[::-1][:top_n]

    shap_plot_data = global_shap_signed_matrix[:, top_idx]
    feature_plot_data = global_feature_value_matrix[:, top_idx]
    plot_feature_names = [_display_name(feature_names[i]) for i in top_idx]

    save_path = xai_dir / f"global_shap_beeswarm_top_{top_n}.png"

    plt.figure(figsize=(10, 6.5))
    shap.summary_plot(
        shap_plot_data,
        features=feature_plot_data,
        feature_names=plot_feature_names,
        show=False,
        plot_size=None,
    )
    fig = plt.gcf()
    axes = fig.axes
    if axes:
        axes[0].set_xlabel("SHAP value")
        axes[0].set_title(f"SHAP summary plot for {model_label} model predictions")
        axes[0].axvline(0, linestyle="--", linewidth=1, color="gray")
    if len(axes) > 1:
        axes[-1].set_ylabel("Feature value (low to high)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    return save_path


def _plot_global_shap_bar(
    feature_ranking_df: pd.DataFrame,
    xai_dir: Path,
    top_n: int = 15,
    model_label: str = "GRU",
) -> Path:
    df = feature_ranking_df.head(top_n).iloc[::-1].copy()
    save_path = xai_dir / f"global_shap_bar_top_{top_n}.png"

    plt.figure(figsize=(10, 6.5))
    plt.barh(df["display_name"], df["mean_abs_shap"])
    plt.xlabel("Mean |SHAP value|")
    plt.ylabel("")
    plt.title(f"Global feature importance based on SHAP values ({model_label} model)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    return save_path


def _plot_grouped_shap_bar(
    grouped_df: pd.DataFrame,
    xai_dir: Path,
) -> Optional[Path]:
    if grouped_df.empty:
        return None

    save_path = xai_dir / "grouped_shap_importance.png"
    df = grouped_df.iloc[::-1].copy()

    plt.figure(figsize=(9, 5.5))
    plt.barh(df["group"], df["group_mean_variable_importance"])
    plt.xlabel("Mean variable importance within group")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    return save_path


#############################
# Patient selection
#############################
def _sample_one(df: pd.DataFrame) -> int:
    if df.empty:
        raise ValueError("No patients available for that selection.")
    return int(df.sample(n=1, random_state=None)["patient_id"].iloc[0])


def _choose_patient_id(pred_df: pd.DataFrame) -> Optional[int]:
    print()
    print("Choose patient selection mode:")
    print("1) Random TP")
    print("2) Random TN")
    print("3) Random FP")
    print("4) Random FN")
    print("5) Random positive patient")
    print("6) Random negative patient")
    print("7) Enter patient ID")
    print("8) Quit")

    choice = input("Selection: ").strip()

    if choice == "1":
        return _sample_one(pred_df[pred_df["outcome_type"] == "TP"])
    if choice == "2":
        return _sample_one(pred_df[pred_df["outcome_type"] == "TN"])
    if choice == "3":
        return _sample_one(pred_df[pred_df["outcome_type"] == "FP"])
    if choice == "4":
        return _sample_one(pred_df[pred_df["outcome_type"] == "FN"])
    if choice == "5":
        return _sample_one(pred_df[pred_df["true_label"] == 1])
    if choice == "6":
        return _sample_one(pred_df[pred_df["true_label"] == 0])
    if choice == "7":
        raw = input("Enter patient_id: ").strip()
        try:
            return int(raw)
        except Exception:
            print("Invalid patient_id.")
            return None
    if choice == "8":
        return None

    print("Invalid selection.")
    return None


#############################
# Main CLI
#############################
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate GRU or TrGRU explanations.")
    parser.add_argument("--architecture", choices=("gru", "trgru"), default=None)
    return parser.parse_args()


def _architecture_paths(architecture: str, setting: str) -> Tuple[Path, Path]:
    run_dir = config.run_dir(config.DISEASE)
    if architecture == "gru":
        checkpoint = (
            run_dir / "model__featvitals_demo.pt"
            if setting == "centralized"
            else run_dir / "Federated" / "model__fedavg__featvitals_demo.pt"
        )
        output = run_dir / "GRU" / "Diagrams" / "SHAP" / setting.capitalize()
    else:
        checkpoint = (
            run_dir / "TrGRU" / "Centralized" / "trgru_model.pt"
            if setting == "centralized"
            else run_dir / "TrGRU" / "Federated" / "model.pt"
        )
        output = run_dir / "TrGRU" / "Diagrams" / "SHAP" / setting.capitalize()
    return checkpoint, output


def _load_architecture_checkpoint(architecture: str, checkpoint: Path, device: str):
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    if architecture == "gru":
        model = _build_model_from_checkpoint(ckpt, device)
        meta = {
            "max_len": int(ckpt.get("max_len", 128)),
            "seed": int(ckpt.get("seed", 42)),
            "feature_mode": str(ckpt.get("feature_mode", config.FEATURE_MODE)),
            "top_k": ckpt.get("top_k"),
            "feature_names": list(ckpt["feature_names"]),
        }
        return model, meta

    cfg = ckpt["config"]
    model = TrGRURisk(
        input_dim=len(ckpt["feature_names"]),
        d_model=int(cfg["d_model"]),
        nhead=int(cfg["nhead"]),
        transformer_layers=int(cfg["transformer_layers"]),
        dim_feedforward=int(cfg["dim_feedforward"]),
        gru_hidden_dim=int(cfg["gru_hidden_dim"]),
        gru_layers=int(cfg["gru_layers"]),
        dropout=float(cfg["dropout"]),
        max_len=int(cfg["max_len"]),
        mlp_hidden_dim=int(cfg["mlp_hidden_dim"]),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    meta = {
        "max_len": int(cfg["max_len"]),
        "seed": int(cfg.get("seed", 42)),
        "feature_mode": config.FEATURE_MODE,
        "top_k": None,
        "feature_names": list(ckpt["feature_names"]),
    }
    return model, meta


def _run_all_global_explanations(architecture: str, device: str) -> None:
    label = "GRU" if architecture == "gru" else "TrGRU"
    for setting in ("centralized", "federated"):
        checkpoint, output = _architecture_paths(architecture, setting)
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
        output.mkdir(parents=True, exist_ok=True)
        model, meta = _load_architecture_checkpoint(architecture, checkpoint, device)
        train_ds = _get_train_dataset(config.DISEASE, meta)
        test_ds = _get_test_dataset(config.DISEASE, meta)
        feature_names = list(meta["feature_names"])
        if list(train_ds.feature_cols) != feature_names or list(test_ds.feature_cols) != feature_names:
            raise ValueError("Checkpoint and current feature order do not match; refusing invalid SHAP.")

        seed = int(meta["seed"])
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        background = _build_background_tensor(
            train_ds, int(meta["max_len"]),
            int(getattr(config, "SHAP_BACKGROUND_SIZE", 64)), seed,
        ).to(device)
        wrapper = PatientMaxRiskWrapper(model).to(device).eval()
        shap_matrix, feature_matrix, patient_ids = _build_global_summary_matrices(
            wrapper, background, test_ds, int(meta["max_len"]), device,
            max_patients=None, batch_size=16,
        )
        ranking = _rank_features(shap_matrix, feature_names)
        variables = _build_variable_importance_df(ranking)
        grouped = _group_shap_importance(variables)
        ranking.to_csv(output / "global_shap_feature_ranking.csv", index=False)
        variables.to_csv(output / "variable_shap_importance.csv", index=False)
        grouped.to_csv(output / "grouped_shap_importance.csv", index=False)
        pd.DataFrame({"patient_id": patient_ids}).to_csv(
            output / "shap_patient_ids.csv", index=False
        )
        _plot_global_shap_beeswarm(
            shap_matrix, feature_matrix, feature_names, output, top_n=15,
            model_label=label,
        )
        _plot_grouped_shap_bar(grouped, output)
        (output / "generation_metadata.json").write_text(
            json.dumps(
                {
                    "architecture": architecture,
                    "setting": setting,
                    "checkpoint": str(checkpoint),
                    "background_size": int(getattr(config, "SHAP_BACKGROUND_SIZE", 64)),
                    "patients_used": len(patient_ids),
                    "seed": seed,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"Saved full-test {label} {setting} SHAP diagrams to: {output}")


def main():
    if shap is None:
        raise ImportError(
            "The shap package is required for this script. "
            f"Original import error: {repr(_shap_import_error)}"
        )

    disease = config.DISEASE
    device = "cuda" if torch.cuda.is_available() else "cpu"

    args = _parse_args()
    if args.architecture is not None:
        _run_all_global_explanations(args.architecture, device)
        return

    model_type = _choose_model_type()
    if model_type is None:
        print("Exiting.")
        return

    model_path, metadata_path, preds_path, xai_dir = _infer_artifact_paths(disease, model_type=model_type)
    ckpt, meta = _load_checkpoint_and_metadata(model_path, metadata_path, device=device)
    model = _build_model_from_checkpoint(ckpt, device=device)

    threshold = float(meta["threshold"])
    max_len = int(meta["max_len"])
    feature_names = list(meta["feature_names"])

    pred_df = pd.read_csv(preds_path)
    pred_df["patient_id"] = pred_df["patient_id"].astype(np.int64)

    print("========================================")
    print("GRU Explainability CLI")
    print("========================================")
    print(f"Model type       : {model_type}")
    print(f"Model checkpoint : {model_path}")
    print(f"Metadata         : {metadata_path}")
    print(f"Predictions      : {preds_path}")
    print(f"XAI output dir   : {xai_dir}")
    print(f"Threshold        : {threshold:.4f}")
    print(f"Device           : {device}")
    print()

    print("Loading datasets...")
    samples_path = None
    if model_type == "federated":
        samples_path = str(config.samples_path(disease))

    test_ds = _get_test_dataset(disease, meta, samples_path=samples_path)
    train_ds = _get_train_dataset(disease, meta, samples_path=samples_path)

    if list(test_ds.feature_cols) != feature_names:
        raise ValueError(
            "Feature order mismatch between metadata and current dataset.\n"
            "This would break SHAP feature naming."
        )

    pid_to_idx = _dataset_pid_to_index(test_ds)

    print("Preparing SHAP background...")
    background_x = _build_background_tensor(
        train_ds=train_ds,
        max_len=max_len,
        n_background=int(getattr(config, "SHAP_BACKGROUND_SIZE", 64)),
        seed=int(meta.get("seed", 42)),
    ).to(device)

    wrapper = PatientMaxRiskWrapper(model).to(device)
    wrapper.eval()

    print("Computing global SHAP summaries...")
    global_shap_signed_matrix, global_feature_value_matrix, global_patient_ids = _build_global_summary_matrices(
        wrapper=wrapper,
        background_x=background_x,
        test_ds=test_ds,
        max_len=max_len,
        device=device,
        max_patients=getattr(config, "SHAP_GLOBAL_MAX_PATIENTS", None),
    )

    feature_ranking_df = _rank_features(global_shap_signed_matrix, feature_names)
    feature_ranking_path = xai_dir / "global_shap_feature_ranking.csv"
    feature_ranking_df.to_csv(feature_ranking_path, index=False)

    variable_importance_df = _build_variable_importance_df(feature_ranking_df)
    variable_ranking_path = xai_dir / "variable_shap_importance.csv"
    variable_importance_df.to_csv(variable_ranking_path, index=False)

    grouped_df = _group_shap_importance(variable_importance_df)
    grouped_path = xai_dir / "grouped_shap_importance.csv"
    grouped_df.to_csv(grouped_path, index=False)

    global_beeswarm_path = _plot_global_shap_beeswarm(
        global_shap_signed_matrix=global_shap_signed_matrix,
        global_feature_value_matrix=global_feature_value_matrix,
        feature_names=feature_names,
        xai_dir=xai_dir,
        top_n=15,
    )

    global_bar_path = _plot_global_shap_bar(
        feature_ranking_df=feature_ranking_df,
        xai_dir=xai_dir,
        top_n=15,
    )

    grouped_bar_path = _plot_grouped_shap_bar(
        grouped_df=grouped_df,
        xai_dir=xai_dir,
    )

    print(f"Saved global beeswarm plot : {global_beeswarm_path}")
    print(f"Saved global bar plot      : {global_bar_path}")
    if grouped_bar_path is not None:
        print(f"Saved grouped bar plot     : {grouped_bar_path}")
    else:
        print("Grouped bar plot was not created because no grouped features were available.")
    print(f"Saved full ranking CSV     : {feature_ranking_path}")
    print(f"Saved variable ranking CSV : {variable_ranking_path}")
    print(f"Saved grouped ranking CSV  : {grouped_path}")
    print(f"Global SHAP patients used  : {len(global_patient_ids)}")
    print()

    while True:
        patient_id = _choose_patient_id(pred_df)
        if patient_id is None:
            print("Exiting.")
            break

        if patient_id not in pid_to_idx:
            print(f"Patient {patient_id} was not found in the current test dataset.")
            continue

        ds_idx = pid_to_idx[patient_id]
        x_seq, y_seq, length_t, pid_t = test_ds[ds_idx]
        length = int(length_t.item())
        patient_id = int(pid_t.item())

        summary = _predict_patient_summary(
            model=model,
            x_seq=x_seq.float(),
            y_seq=y_seq,
            length=length,
            threshold=threshold,
            device=device,
        )

        event_info = _patient_event_info(test_ds, patient_id)
        pos_labels = _positive_label_timestep_labels(y_seq, length)
        history_span_h = _hours_span_from_length(length)

        x_pad = _pad_sequence_to_max_len(x_seq.float(), max_len=max_len).unsqueeze(0).to(device)

        print()
        print("========================================")
        print(f"Patient ID                : {patient_id}")
        print(f"Observed timesteps        : {length}")
        print(f"Observed history span     : {history_span_h}h")
        print(f"True label                : {summary['true_label']}")
        print(f"Predicted probability     : {summary['pred_prob']:.4f}")
        print(f"Predicted class           : {summary['pred_label']}")
        print(f"Outcome type              : {summary['outcome_type']}")
        print(f"Threshold                 : {summary['threshold']:.4f}")
        print(f"Peak model risk timestep  : {summary['peak_timestep_label']}")
        print(f"Last observed window end/Prediction made at min     : {event_info['final_t_end']}")
        print(f"Cardiac arrest at minute   : {event_info['t_event']}")
        print(f"Time between model prediction and event   : {event_info['time_to_event_from_final_t_end']}")
        print(f"Lead time min (mins)      : {event_info['lead_time_min']}")
        print(f"Lead time max (mins)      : {event_info['lead_time_max']}")
        if pos_labels:
            print(f"Positive label timesteps  : {', '.join(pos_labels)}")
        else:
            print("Positive label timesteps  : none")
        print("========================================")

        print("Computing SHAP...")
        shap_values_tf = _compute_shap_values(
            wrapper=wrapper,
            background_x=background_x,
            patient_x=x_pad,
        )

        top_pairs = _format_top_pairs(
            shap_values_tf=shap_values_tf,
            feature_names=feature_names,
            valid_len=length,
            top_n=int(getattr(config, "SHAP_TOP_N", 10)),
        )

        print()
        print("Top feature-time SHAP contributions:")
        for i, (label, value) in enumerate(top_pairs, start=1):
            print(f"{i:2d}. {label:<40} {value:+.4f}")

        print()
        print("Computing timestamp occlusion...")
        occlusion_scores = _temporal_occlusion_importance(
            model=model,
            x_seq=x_seq.float(),
            length=length,
            device=device,
        )

        shap_plot_path = _plot_temporal_shap(
            shap_values_tf=shap_values_tf,
            valid_len=length,
            patient_id=patient_id,
            xai_dir=xai_dir,
        )
        occlusion_plot_path = _plot_temporal_occlusion(
            occlusion_scores=occlusion_scores,
            valid_len=length,
            patient_id=patient_id,
            xai_dir=xai_dir,
        )

        print()
        print(f"Saved temporal SHAP plot     : {shap_plot_path}")
        print(f"Saved temporal occlusion plot: {occlusion_plot_path}")
        print()

        again = input("Explain another patient? (y/n): ").strip().lower()
        if again != "y":
            print("Done.")
            break


if __name__ == "__main__":
    main()
