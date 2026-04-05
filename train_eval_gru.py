# train_eval_gru.py
# Train + evaluate a supervised GRU early detection model on window-sequence data.
# Reports AUROC and AUPRC on train/val/test splits.
# Saves ROC and PR curves for the TEST split.
#
# NEW:
# - Uses pos_weight in BCEWithLogitsLoss during TRAINING to handle class imbalance.
# - pos_weight computed from TRAIN split masked labels: n_neg / n_pos
# - Optional clipping via config.POS_WEIGHT_MAX (if not present, no clipping).
#
# NEW (metrics parity with ML table, but only for TEST):
# - Selects a classification threshold on the VALIDATION split by maximizing F1.
# - Applies that fixed threshold to TEST to compute: Accuracy, Precision, Recall, F1,
#   confusion matrix (TN/FP/FN/TP) and FPR.
#
# NEW (focal loss):
# - Training uses masked focal loss on per-timestep logits.
#
# NEW (attention pooling aux head):
# - Model may return dict containing logits_ts and optional logit_seq.
# - If logit_seq exists, we can add an auxiliary sequence-level loss.
#   This gives you attention weights for XAI later without changing evaluation.
#
# NEW (XAI prep):
# - Saves trained model checkpoint to the normal output folder from config.
# - Saves metadata.json with feature names and model config.
# - Saves test_predictions.csv with one row per patient so explainability can
#   sample TP / TN / FP / FN cases later.

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
import json
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

import config
from sequence_dataset_gru import PatientSequenceDataset, pad_collate
from gru_risk import GRURisk

try:
    from memory_profiler import memory_usage
except Exception:
    memory_usage = None


#############################
# Attention aux loss config
#############################
# Keep small so it doesn't hijack training
ATTN_AUX_ENABLED = True
ATTN_AUX_WEIGHT = 0.1


#############################
# Feature mode helpers
#############################
def _feature_mode() -> str:
    return str(getattr(config, "FEATURE_MODE", "all")).strip().lower()


def _feature_mode_tag() -> str:
    return _feature_mode().replace("+", "_")


def _run_tag(top_k: int | None) -> str:
    feature_mode = _feature_mode()
    if feature_mode == "all":
        return "all" if top_k is None else f"topk{int(top_k)}"
    return f"feat{_feature_mode_tag()}"


#############################
# Metrics
#############################
def roc_auc_score_manual(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = y_true.astype(np.int64)
    y_score = y_score.astype(np.float64)

    pos = y_true == 1
    neg = y_true == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(y_score)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(len(y_score)) + 1

    sorted_scores = y_score[order]
    sorted_pos = pos[order]

    i = 0
    sum_ranks_pos = 0.0
    while i < len(sorted_scores):
        j = i
        while j + 1 < len(sorted_scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        avg_rank = 0.5 * (ranks[order[i]] + ranks[order[j]])
        sum_ranks_pos += int(sorted_pos[i : j + 1].sum()) * avg_rank
        i = j + 1

    return (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def avg_precision_score_manual(y_true: np.ndarray, y_score: np.ndarray) -> float:
    n_pos = int((y_true == 1).sum())
    if n_pos == 0:
        return float("nan")

    order = np.argsort(-y_score)
    y_sorted = y_true[order]

    tp = 0
    fp = 0
    ap = 0.0
    prev_rec = 0.0

    for i in range(len(y_sorted)):
        if y_sorted[i] == 1:
            tp += 1
        else:
            fp += 1
        rec = tp / n_pos
        prec = tp / max(tp + fp, 1)
        if rec > prev_rec:
            ap += prec * (rec - prev_rec)
            prev_rec = rec

    return ap


def _precision_recall_curve_manual(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    y_true = y_true.astype(np.int64)
    y_score = y_score.astype(np.float64)

    n_pos = int((y_true == 1).sum())
    if n_pos == 0:
        return np.array([0.0, 1.0], dtype=np.float64), np.array([1.0, 0.0], dtype=np.float64)

    order = np.argsort(-y_score)
    y_sorted = y_true[order]

    tp = np.cumsum(y_sorted == 1)
    fp = np.cumsum(y_sorted == 0)

    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / n_pos

    recall = np.concatenate([np.array([0.0]), recall, np.array([1.0])])
    precision = np.concatenate([np.array([1.0]), precision, np.array([precision[-1] if len(precision) else 0.0])])

    return recall, precision


def _roc_curve_manual(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    y_true = y_true.astype(np.int64)
    y_score = y_score.astype(np.float64)

    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return np.array([0.0, 1.0], dtype=np.float64), np.array([0.0, 1.0], dtype=np.float64)

    order = np.argsort(-y_score)
    y_sorted = y_true[order]

    tp = np.cumsum(y_sorted == 1)
    fp = np.cumsum(y_sorted == 0)

    tpr = tp / float(n_pos)
    fpr = fp / float(n_neg)

    fpr = np.concatenate([np.array([0.0]), fpr, np.array([1.0])])
    tpr = np.concatenate([np.array([0.0]), tpr, np.array([1.0])])

    return fpr, tpr


def _model_forward_logits_ts(model_out) -> torch.Tensor:
    if isinstance(model_out, dict):
        return model_out["logits_ts"]
    return model_out


def _model_forward_logit_seq(model_out) -> torch.Tensor | None:
    if isinstance(model_out, dict):
        return model_out.get("logit_seq", None)
    return None


def _flatten_loader_probs(model, loader, device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_y: List[np.ndarray] = []
    all_p: List[np.ndarray] = []

    with torch.no_grad():
        for x, y, mask, lengths, pids in loader:
            x = x.to(device)
            y = y.to(device)
            mask = mask.to(device)

            out = model(x, lengths)
            logits = _model_forward_logits_ts(out)

            probs = torch.sigmoid(logits)
            valid = mask > 0.5

            yt = y[valid].detach().cpu().numpy().astype(np.int64)
            pt = probs[valid].detach().cpu().numpy().astype(np.float64)

            if yt.size > 0:
                all_y.append(yt)
                all_p.append(pt)

    if not all_y:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float64)

    return np.concatenate(all_y), np.concatenate(all_p)


def _pick_threshold_max_f1(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    y_true = y_true.astype(np.int64)
    y_prob = y_prob.astype(np.float64)

    n_pos = int((y_true == 1).sum())
    if n_pos == 0:
        return 0.5

    order = np.argsort(-y_prob)
    y_sorted = y_true[order]
    p_sorted = y_prob[order]

    tp = np.cumsum(y_sorted == 1)
    fp = np.cumsum(y_sorted == 0)

    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / float(n_pos)

    denom = precision + recall
    f1 = np.where(denom > 0, 2.0 * precision * recall / denom, 0.0)

    best_idx = int(np.argmax(f1))
    best_t = float(p_sorted[best_idx])

    if best_t < 0.0:
        best_t = 0.0
    if best_t > 1.0:
        best_t = 1.0

    return best_t


def _threshold_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> Dict[str, float]:
    y_true = y_true.astype(np.int64)
    y_prob = y_prob.astype(np.float64)
    thr = float(threshold)

    y_pred = (y_prob >= thr).astype(np.int64)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    acc = (tp + tn) / max(tp + tn + fp + fn, 1)

    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)

    denom = prec + rec
    f1 = (2.0 * prec * rec / denom) if denom > 0 else 0.0

    fpr = fp / max(fp + tn, 1)

    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "fpr": float(fpr),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
        "tp": float(tp),
        "threshold": float(thr),
    }


#############################
# Loss
#############################
def masked_bce(logits, targets, mask, pos_weight=None):
    loss_fn = nn.BCEWithLogitsLoss(reduction="none", pos_weight=pos_weight)
    loss = loss_fn(logits, targets.float())
    loss = loss * mask.float()
    return loss.sum() / mask.sum().clamp_min(1.0)


def masked_focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    gamma: float = 2.0,
    alpha: float | None = None,
    pos_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    bce = nn.functional.binary_cross_entropy_with_logits(
        logits,
        targets.float(),
        reduction="none",
        pos_weight=pos_weight,
    )

    p = torch.sigmoid(logits)
    pt = torch.where(targets > 0.5, p, 1.0 - p)

    focal_factor = (1.0 - pt).clamp_min(0.0).pow(gamma)
    loss = focal_factor * bce

    if alpha is not None:
        a = float(alpha)
        alpha_t = torch.where(
            targets > 0.5,
            torch.tensor(a, device=logits.device, dtype=loss.dtype),
            torch.tensor(1.0 - a, device=logits.device, dtype=loss.dtype),
        )
        loss = alpha_t * loss

    loss = loss * mask.float()
    return loss.sum() / mask.sum().clamp_min(1.0)


def _compute_seq_targets(y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    valid = mask > 0.5
    y_pos = (y > 0.5) & valid
    y_seq = y_pos.any(dim=1).float()
    return y_seq


def _compute_pos_weight_from_loader(train_loader: DataLoader, device: str) -> torch.Tensor | None:
    n_pos = 0
    n_neg = 0

    for x, y, mask, lengths, pids in train_loader:
        y_np = y.detach().cpu().numpy().astype(np.int64)
        m_np = mask.detach().cpu().numpy().astype(np.float32)

        valid = m_np > 0.5
        if not np.any(valid):
            continue

        yv = y_np[valid]
        n_pos += int((yv == 1).sum())
        n_neg += int((yv == 0).sum())

    if n_pos == 0:
        return None

    pw = float(n_neg) / float(n_pos)

    if hasattr(config, "POS_WEIGHT_MAX") and config.POS_WEIGHT_MAX is not None:
        pw_max = float(config.POS_WEIGHT_MAX)
        pw = float(np.clip(pw, 1.0, pw_max))

    return torch.tensor([pw], dtype=torch.float32, device=device)


#############################
# Config
#############################
@dataclass
class TrainConfig:
    max_len: int = 128
    batch_size: int = 32
    hidden_dim: int = 128
    num_layers: int = 1
    dropout: float = 0.2
    lr: float = 1e-3
    weight_decay: float = 1e-5
    epochs: int = 20
    patience: int = 3
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


#############################
# Eval
#############################
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_y, all_s = [], []
    total_loss = 0.0

    for x, y, mask, lengths, pids in loader:
        x = x.to(device)
        y = y.to(device)
        mask = mask.to(device)

        out = model(x, lengths)
        logits = _model_forward_logits_ts(out)

        loss = masked_bce(logits, y, mask)
        total_loss += float(loss.item())

        probs = torch.sigmoid(logits)
        valid = mask > 0.5
        yt = y[valid].detach().cpu().numpy().astype(np.int64)
        ys = probs[valid].detach().cpu().numpy().astype(np.float64)

        if yt.size:
            all_y.append(yt)
            all_s.append(ys)

    if not all_y:
        return {
            "loss": float("nan"),
            "auroc": float("nan"),
            "auprc": float("nan"),
            "n_pos": float("nan"),
            "n_neg": float("nan"),
        }

    y_true = np.concatenate(all_y)
    y_score = np.concatenate(all_s)

    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())

    return {
        "loss": total_loss / max(len(loader), 1),
        "auroc": roc_auc_score_manual(y_true, y_score),
        "auprc": avg_precision_score_manual(y_true, y_score),
        "n_pos": float(n_pos),
        "n_neg": float(n_neg),
    }


def _save_test_curves(
    model,
    test_loader,
    disease: config.DiseaseSpec,
    top_k: int | None,
    device: str,
) -> Dict[str, str | None]:
    out_dir = config.run_dir(disease)

    curves_dir = out_dir / "Curves"
    curves_dir.mkdir(parents=True, exist_ok=True)

    run_tag = _run_tag(top_k)

    y_true, y_prob = _flatten_loader_probs(model, test_loader, device=device)
    if y_true.size == 0:
        return {"roc_path": None, "pr_path": None}

    roc_path = str(curves_dir / f"roc_test__{run_tag}.png")
    pr_path = str(curves_dir / f"pr_test__{run_tag}.png")

    fpr, tpr = _roc_curve_manual(y_true, y_prob)
    plt.figure()
    plt.plot(fpr, tpr)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC curve (TEST, {run_tag})")
    plt.tight_layout()
    plt.savefig(roc_path, dpi=200)
    plt.close()

    recall, precision = _precision_recall_curve_manual(y_true, y_prob)
    plt.figure()
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"PR curve (TEST, {run_tag})")
    plt.tight_layout()
    plt.savefig(pr_path, dpi=200)
    plt.close()

    return {"roc_path": roc_path, "pr_path": pr_path}


#############################
# XAI artifact helpers
#############################
def _collect_patient_level_predictions(model, loader, device, threshold: float) -> pd.DataFrame:
    model.eval()
    rows: List[Dict[str, float | int | str]] = []

    with torch.no_grad():
        for x, y, mask, lengths, pids in loader:
            x = x.to(device)
            y = y.to(device)
            mask = mask.to(device)

            out = model(x, lengths)
            logits = _model_forward_logits_ts(out)
            probs = torch.sigmoid(logits)

            probs_np = probs.detach().cpu().numpy().astype(np.float64)
            y_np = y.detach().cpu().numpy().astype(np.int64)
            mask_np = mask.detach().cpu().numpy().astype(np.float32)
            pids_np = pids.detach().cpu().numpy().astype(np.int64)

            for i in range(len(pids_np)):
                valid = mask_np[i] > 0.5
                if not np.any(valid):
                    continue

                seq_probs = probs_np[i, valid]
                seq_y = y_np[i, valid]

                pred_prob = float(seq_probs.max())
                true_label = int((seq_y == 1).any())
                pred_label = int(pred_prob >= float(threshold))

                if true_label == 1 and pred_label == 1:
                    outcome_type = "TP"
                elif true_label == 0 and pred_label == 0:
                    outcome_type = "TN"
                elif true_label == 0 and pred_label == 1:
                    outcome_type = "FP"
                else:
                    outcome_type = "FN"

                rows.append(
                    {
                        "patient_id": int(pids_np[i]),
                        "true_label": int(true_label),
                        "pred_prob": float(pred_prob),
                        "pred_label": int(pred_label),
                        "outcome_type": outcome_type,
                        "threshold": float(threshold),
                    }
                )

    if not rows:
        return pd.DataFrame(
            columns=["patient_id", "true_label", "pred_prob", "pred_label", "outcome_type", "threshold"]
        )

    return pd.DataFrame(rows)


def _save_xai_artifacts(
    model,
    disease: config.DiseaseSpec,
    cfg: TrainConfig,
    top_k: int | None,
    threshold: float,
    feature_names: List[str],
    test_pred_df: pd.DataFrame,
) -> Dict[str, str]:
    out_dir = config.run_dir(disease)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_tag = _run_tag(top_k)

    model_path = str(out_dir / f"model__{run_tag}.pt")
    metadata_path = str(out_dir / f"metadata__{run_tag}.json")
    predictions_path = str(out_dir / f"test_predictions__{run_tag}.csv")

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "input_dim": int(len(feature_names)),
        "hidden_dim": int(cfg.hidden_dim),
        "num_layers": int(cfg.num_layers),
        "dropout": float(cfg.dropout),
        "max_len": int(cfg.max_len),
        "feature_names": list(feature_names),
        "threshold": float(threshold),
        "feature_mode": _feature_mode(),
        "top_k": None if top_k is None else int(top_k),
        "use_layernorm": True,
        "use_attention_pooling": True,
    }
    torch.save(checkpoint, model_path)

    metadata = {
        "input_dim": int(len(feature_names)),
        "hidden_dim": int(cfg.hidden_dim),
        "num_layers": int(cfg.num_layers),
        "dropout": float(cfg.dropout),
        "max_len": int(cfg.max_len),
        "batch_size": int(cfg.batch_size),
        "lr": float(cfg.lr),
        "weight_decay": float(cfg.weight_decay),
        "epochs": int(cfg.epochs),
        "patience": int(cfg.patience),
        "seed": int(cfg.seed),
        "device": str(cfg.device),
        "threshold": float(threshold),
        "feature_mode": _feature_mode(),
        "top_k": None if top_k is None else int(top_k),
        "feature_names": list(feature_names),
        "use_layernorm": True,
        "use_attention_pooling": True,
        "attn_aux_enabled": bool(ATTN_AUX_ENABLED),
        "attn_aux_weight": float(ATTN_AUX_WEIGHT),
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    test_pred_df.to_csv(predictions_path, index=False)

    return {
        "model_path": model_path,
        "metadata_path": metadata_path,
        "predictions_path": predictions_path,
    }


#############################
# Train + Eval
#############################
def train_and_eval(
    disease: config.DiseaseSpec,
    cfg: TrainConfig,
    top_k: int | None = None,
    rank_path: str | None = None,
) -> Dict[str, Dict]:

    t0 = time.perf_counter()

    def _run():
        train_ds = PatientSequenceDataset(
            split="train",
            disease=disease,
            max_len=cfg.max_len,
            seed=cfg.seed,
            normalize=True,
            top_k=top_k,
            rank_path=rank_path,
        )
        val_ds = PatientSequenceDataset(
            split="val",
            disease=disease,
            max_len=cfg.max_len,
            seed=cfg.seed,
            normalize=True,
            top_k=top_k,
            rank_path=rank_path,
        )
        test_ds = PatientSequenceDataset(
            split="test",
            disease=disease,
            max_len=cfg.max_len,
            seed=cfg.seed,
            normalize=True,
            top_k=top_k,
            rank_path=rank_path,
        )

        train_loader = DataLoader(train_ds, cfg.batch_size, True, collate_fn=pad_collate)
        val_loader = DataLoader(val_ds, cfg.batch_size, False, collate_fn=pad_collate)
        test_loader = DataLoader(test_ds, cfg.batch_size, False, collate_fn=pad_collate)

        model = GRURisk(
            input_dim=len(train_ds.feature_cols),
            hidden_dim=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout,
            use_layernorm=True,
            use_attention_pooling=True,
        ).to(cfg.device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

        best_val = -1.0
        best_state = None
        bad_epochs = 0

        for epoch in range(cfg.epochs):
            model.train()
            for x, y, mask, lengths, pids in train_loader:
                x = x.to(cfg.device)
                y = y.to(cfg.device)
                mask = mask.to(cfg.device)

                optimizer.zero_grad()

                out = model(x, lengths)
                logits_ts = _model_forward_logits_ts(out)

                loss_ts = masked_focal_loss(
                    logits_ts,
                    y,
                    mask,
                    gamma=2.0,
                    alpha=None,
                    pos_weight=None,
                )

                loss = loss_ts

                if ATTN_AUX_ENABLED:
                    logit_seq = _model_forward_logit_seq(out)
                    if logit_seq is not None:
                        y_seq = _compute_seq_targets(y, mask)
                        loss_seq = nn.functional.binary_cross_entropy_with_logits(
                            logit_seq,
                            y_seq,
                            reduction="mean",
                        )
                        loss = loss + float(ATTN_AUX_WEIGHT) * loss_seq

                loss.backward()
                optimizer.step()

            val = evaluate(model, val_loader, cfg.device)
            if val["auroc"] > best_val:
                best_val = val["auroc"]
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= cfg.patience:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)

        train_metrics = evaluate(model, train_loader, cfg.device)
        val_metrics = evaluate(model, val_loader, cfg.device)
        test_metrics = evaluate(model, test_loader, cfg.device)

        yv_true, yv_prob = _flatten_loader_probs(model, val_loader, device=cfg.device)
        chosen_threshold = 0.5 if yv_true.size == 0 else _pick_threshold_max_f1(yv_true, yv_prob)

        yt_true, yt_prob = _flatten_loader_probs(model, test_loader, device=cfg.device)
        if yt_true.size == 0:
            test_metrics.update(
                {
                    "accuracy": float("nan"),
                    "precision": float("nan"),
                    "recall": float("nan"),
                    "f1": float("nan"),
                    "fpr": float("nan"),
                    "tn": float("nan"),
                    "fp": float("nan"),
                    "fn": float("nan"),
                    "tp": float("nan"),
                    "threshold": float(chosen_threshold),
                }
            )
        else:
            test_metrics.update(_threshold_metrics(yt_true, yt_prob, chosen_threshold))

        curve_paths = _save_test_curves(
            model=model,
            test_loader=test_loader,
            disease=disease,
            top_k=top_k,
            device=cfg.device,
        )

        test_pred_df = _collect_patient_level_predictions(
            model=model,
            loader=test_loader,
            device=cfg.device,
            threshold=float(chosen_threshold),
        )

        artifact_paths = _save_xai_artifacts(
            model=model,
            disease=disease,
            cfg=cfg,
            top_k=top_k,
            threshold=float(chosen_threshold),
            feature_names=list(train_ds.feature_cols),
            test_pred_df=test_pred_df,
        )

        return {
            "train": train_metrics,
            "val": val_metrics,
            "test": test_metrics,
            "threshold": float(chosen_threshold),
            "n_features": len(train_ds.feature_cols),
            "feature_mode": _feature_mode(),
            "curve_paths": curve_paths,
            "artifact_paths": artifact_paths,
            "pos_weight": None,
        }

    if memory_usage is not None:
        mem, out = memory_usage((_run, (), {}), retval=True, interval=0.1)
        cpu_peak = float(max(mem))
    else:
        out = _run()
        cpu_peak = float("nan")

    runtime = time.perf_counter() - t0

    out["extra"] = {
        "runtime_sec": runtime,
        "cpu_peak_mib": cpu_peak,
        "n_features": out["n_features"],
        "feature_mode": out.get("feature_mode", None),
        "roc_path": out.get("curve_paths", {}).get("roc_path", None),
        "pr_path": out.get("curve_paths", {}).get("pr_path", None),
        "model_path": out.get("artifact_paths", {}).get("model_path", None),
        "metadata_path": out.get("artifact_paths", {}).get("metadata_path", None),
        "predictions_path": out.get("artifact_paths", {}).get("predictions_path", None),
        "pos_weight": out.get("pos_weight", None),
        "threshold": out.get("threshold", None),
    }

    return out