# xai.py
# CLI explainability tool for the centralized GRU model.
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
#
# Notes:
# - SHAP explains a patient-level max-risk score:
#     max valid timestep logit over the sequence
# - patient-level prediction summary still uses max valid timestep probability
# - padded rows are assumed to be all zeros after normalization
# - for SHAP wrapper length inference, valid timesteps are inferred from non-zero rows

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

import config
from sequence_dataset_gru import PatientSequenceDataset
from gru_risk import GRURisk

try:
    import shap
except Exception as e:
    shap = None
    _shap_import_error = e
else:
    _shap_import_error = None


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


def _infer_artifact_paths(disease: config.DiseaseSpec) -> Tuple[Path, Path, Path, Path]:
    out_dir = config.run_dir(disease)
    if not out_dir.exists():
        raise FileNotFoundError(f"Run directory does not exist: {out_dir}")

    metadata_files = sorted(out_dir.glob("metadata__*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not metadata_files:
        raise FileNotFoundError(f"No metadata__*.json found in {out_dir}")

    metadata_path = metadata_files[0]
    with open(metadata_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    run_tag = _run_tag_from_metadata(meta)

    model_path = out_dir / f"model__{run_tag}.pt"
    preds_path = out_dir / f"test_predictions__{run_tag}.csv"

    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    if not preds_path.exists():
        raise FileNotFoundError(f"Predictions CSV not found: {preds_path}")

    xai_dir = out_dir / "XAI"
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


def _get_test_dataset(disease: config.DiseaseSpec, meta: Dict) -> PatientSequenceDataset:
    return PatientSequenceDataset(
        split="test",
        disease=disease,
        max_len=int(meta["max_len"]),
        seed=int(meta.get("seed", 42)),
        normalize=True,
        top_k=meta.get("top_k", None),
        rank_path=str(config.stability_combined_path(config.DISEASE)),
    )


def _get_train_dataset(disease: config.DiseaseSpec, meta: Dict) -> PatientSequenceDataset:
    return PatientSequenceDataset(
        split="train",
        disease=disease,
        max_len=int(meta["max_len"]),
        seed=int(meta.get("seed", 42)),
        normalize=True,
        top_k=meta.get("top_k", None),
        rank_path=str(config.stability_combined_path(config.DISEASE)),
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
    active = (x.abs().sum(dim=2) > eps)
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


def _safe_minmax(series: pd.Series) -> Tuple[float | None, float | None]:
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


#############################
# Patient selection
#############################
def _sample_one(df: pd.DataFrame) -> int:
    if df.empty:
        raise ValueError("No patients available for that selection.")
    return int(df.sample(n=1, random_state=None)["patient_id"].iloc[0])


def _choose_patient_id(pred_df: pd.DataFrame) -> int | None:
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
def main():
    if shap is None:
        raise ImportError(
            "The shap package is required for this script. "
            f"Original import error: {repr(_shap_import_error)}"
        )

    disease = config.DISEASE
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_path, metadata_path, preds_path, xai_dir = _infer_artifact_paths(disease)
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
    print(f"Model checkpoint : {model_path}")
    print(f"Metadata         : {metadata_path}")
    print(f"Predictions      : {preds_path}")
    print(f"XAI output dir   : {xai_dir}")
    print(f"Threshold        : {threshold:.4f}")
    print(f"Device           : {device}")
    print()

    print("Loading datasets...")
    test_ds = _get_test_dataset(disease, meta)
    train_ds = _get_train_dataset(disease, meta)

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