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
# - Keeps train/val AUROC/AUPRC available internally, but you can choose what to log.
#
# NEW (focal loss option):
# - Adds masked focal loss (logits version) to focus training on hard examples.
# - Training uses focal loss by default here (pos_weight disabled for focal).
# - Eval loss still uses BCE (ranking metrics unchanged). You can switch eval loss
#   to focal too if you want, but it doesn't affect AUROC/AUPRC.

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
import time

import numpy as np
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
    """
    Returns (fpr, tpr) arrays, including (0,0) and (1,1) endpoints.
    """
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


def _flatten_loader_probs(model, loader, device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_y: List[np.ndarray] = []
    all_p: List[np.ndarray] = []

    with torch.no_grad():
        for x, y, mask, lengths in loader:
            x = x.to(device)
            y = y.to(device)
            mask = mask.to(device)

            logits = model(x, lengths)
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
    """
    Choose threshold that maximizes F1 on (y_true, y_prob).
    Returns t such that y_pred = (y_prob >= t).
    """
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
    """
    Compute threshold-based metrics + confusion matrix + FPR.
    """
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
    """
    Focal loss for binary classification using logits, with timestep masking.

    - gamma: focusing parameter (2.0 is common)
    - alpha: optional class balance factor:
        * if alpha is not None: alpha for positives, (1-alpha) for negatives
    - pos_weight: optional (like BCEWithLogitsLoss pos_weight). Suggest starting with None
      when using focal, then later try a damped pos_weight if needed.
    """
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


def _compute_pos_weight_from_loader(train_loader: DataLoader, device: str) -> torch.Tensor | None:
    """
    Compute pos_weight = n_neg / n_pos from TRAIN masked labels.
    Optional clipping: if config.POS_WEIGHT_MAX exists, clip to that max.
    If no positives found, returns None.
    """
    n_pos = 0
    n_neg = 0

    for x, y, mask, lengths in train_loader:
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

    for x, y, mask, lengths in loader:
        x = x.to(device)
        y = y.to(device)
        mask = mask.to(device)

        logits = model(x, lengths)
        loss = masked_bce(logits, y, mask)  # keep BCE for reporting
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
    """
    Save ROC and PR plots for TEST split.
    """
    out_dir = config.run_dir(disease)

    curves_dir = out_dir / "Curves"
    curves_dir.mkdir(parents=True, exist_ok=True)

    topk_tag = "all" if top_k is None else str(int(top_k))

    y_true, y_prob = _flatten_loader_probs(model, test_loader, device=device)
    if y_true.size == 0:
        return {"roc_path": None, "pr_path": None}

    roc_path = str(curves_dir / f"roc_test__topk{topk_tag}.png")
    pr_path = str(curves_dir / f"pr_test__topk{topk_tag}.png")

    #############################
    # ROC
    #############################
    fpr, tpr = _roc_curve_manual(y_true, y_prob)

    plt.figure()
    plt.plot(fpr, tpr)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC curve (TEST, top_k={topk_tag})")
    plt.tight_layout()
    plt.savefig(roc_path, dpi=200)
    plt.close()

    #############################
    # PR
    #############################
    recall, precision = _precision_recall_curve_manual(y_true, y_prob)

    plt.figure()
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"PR curve (TEST, top_k={topk_tag})")
    plt.tight_layout()
    plt.savefig(pr_path, dpi=200)
    plt.close()

    return {"roc_path": roc_path, "pr_path": pr_path}


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
        ).to(cfg.device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

        #############################
        # pos_weight (TRAIN only)
        # NOTE: For focal loss, start with pos_weight disabled (None). We'll ablate later if needed.
        #############################
        use_pos_weight = True
        if hasattr(config, "USE_POS_WEIGHT"):
            use_pos_weight = bool(config.USE_POS_WEIGHT)

        pos_weight = None
        if use_pos_weight:
            pos_weight = _compute_pos_weight_from_loader(train_loader, device=cfg.device)

            print("#############################")
            print("Pos weight configuration")
            print(f"USE_POS_WEIGHT: {use_pos_weight}")
            if pos_weight is None:
                print("pos_weight: None (no positives found in train masked labels)")
            else:
                print(f"pos_weight: {float(pos_weight.item()):.6f}")
                if hasattr(config, "POS_WEIGHT_MAX") and config.POS_WEIGHT_MAX is not None:
                    print(f"POS_WEIGHT_MAX: {float(config.POS_WEIGHT_MAX):.6f}")
            print("#############################")

        best_val = -1.0
        best_state = None
        bad_epochs = 0

        for epoch in range(cfg.epochs):
            model.train()
            for x, y, mask, lengths in train_loader:
                x = x.to(cfg.device)
                y = y.to(cfg.device)
                mask = mask.to(cfg.device)

                optimizer.zero_grad()

                # FOCAL LOSS (default)
                loss = masked_focal_loss(
                    model(x, lengths),
                    y,
                    mask,
                    gamma=2.0,
                    alpha=None,
                    pos_weight=None,
                )

                loss.backward()
                optimizer.step()

            val = evaluate(model, val_loader, cfg.device)
            if val["auroc"] > best_val:
                best_val = val["auroc"]
                best_state = model.state_dict()
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= cfg.patience:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)

        # Compute baseline metrics (AUROC/AUPRC) for splits
        train_metrics = evaluate(model, train_loader, cfg.device)
        val_metrics = evaluate(model, val_loader, cfg.device)
        test_metrics = evaluate(model, test_loader, cfg.device)

        # ----------------------------
        # Choose threshold on VAL to maximize F1 (NO TEST LEAKAGE)
        # ----------------------------
        yv_true, yv_prob = _flatten_loader_probs(model, val_loader, device=cfg.device)
        if yv_true.size == 0:
            chosen_threshold = 0.5
        else:
            chosen_threshold = _pick_threshold_max_f1(yv_true, yv_prob)

        # ----------------------------
        # Add threshold-based metrics ONLY for TEST (as requested)
        # ----------------------------
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

        return {
            "train": train_metrics,
            "val": val_metrics,
            "test": test_metrics,
            "threshold": float(chosen_threshold),
            "n_features": len(train_ds.feature_cols),
            "curve_paths": curve_paths,
            "pos_weight": (float(pos_weight.item()) if pos_weight is not None else None),
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
        "roc_path": out.get("curve_paths", {}).get("roc_path", None),
        "pr_path": out.get("curve_paths", {}).get("pr_path", None),
        "pos_weight": out.get("pos_weight", None),
        "threshold": out.get("threshold", None),
    }

    return out