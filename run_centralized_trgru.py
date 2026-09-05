"""Train and evaluate centralized TrGRU on the same data as run_centralized.py."""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import config
from sequence_dataset_gru import PatientSequenceDataset, pad_collate
from train_eval_gru import (
    _flatten_loader_probs,
    _pick_threshold_max_f1,
    _resolve_pos_weight,
    _threshold_metrics,
    avg_precision_score_manual,
    masked_focal_loss,
    roc_auc_score_manual,
)
from trgru_model import TrGRURisk


@dataclass
class TrGRUConfig:
    max_len: int = 128
    batch_size: int = 32
    d_model: int = 128
    nhead: int = 4
    transformer_layers: int = 3
    dim_feedforward: int = 256
    gru_hidden_dim: int = 128
    gru_layers: int = 2
    mlp_hidden_dim: int = 64
    dropout: float = 0.2
    lr: float = 3e-4
    weight_decay: float = 1e-5
    epochs: int = 25
    patience: int = 5
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the adapted TrGRU model.")
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--transformer-layers", type=int, default=3)
    parser.add_argument("--dim-feedforward", type=int, default=256)
    parser.add_argument("--gru-hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--validation-only", action="store_true")
    return parser.parse_args()


def _metrics(model, loader, device: str) -> dict[str, float]:
    y_true, y_prob = _flatten_loader_probs(model, loader, device)
    return {
        "auroc": roc_auc_score_manual(y_true, y_prob),
        "auprc": avg_precision_score_manual(y_true, y_prob),
        "n_pos": float((y_true == 1).sum()),
        "n_neg": float((y_true == 0).sum()),
    }


def main() -> None:
    args = _parse_args()
    cfg = TrGRUConfig(
        d_model=args.d_model,
        nhead=args.nhead,
        transformer_layers=args.transformer_layers,
        dim_feedforward=args.dim_feedforward,
        gru_hidden_dim=args.gru_hidden_dim,
        dropout=args.dropout,
        lr=args.lr,
        epochs=args.epochs,
        patience=args.patience,
    )
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    requested_splits = ("train", "val") if args.validation_only else ("train", "val", "test")
    datasets = {
        split: PatientSequenceDataset(
            split=split,
            disease=config.DISEASE,
            max_len=cfg.max_len,
            seed=cfg.seed,
            normalize=True,
        )
        for split in requested_splits
    }
    loaders = {
        "train": DataLoader(datasets["train"], cfg.batch_size, True, collate_fn=pad_collate),
        "train_eval": DataLoader(datasets["train"], cfg.batch_size, False, collate_fn=pad_collate),
        "val": DataLoader(datasets["val"], cfg.batch_size, False, collate_fn=pad_collate),
    }
    if not args.validation_only:
        loaders["test"] = DataLoader(
            datasets["test"], cfg.batch_size, False, collate_fn=pad_collate
        )

    model = TrGRURisk(
        input_dim=len(datasets["train"].feature_cols),
        d_model=cfg.d_model,
        nhead=cfg.nhead,
        transformer_layers=cfg.transformer_layers,
        dim_feedforward=cfg.dim_feedforward,
        gru_hidden_dim=cfg.gru_hidden_dim,
        gru_layers=cfg.gru_layers,
        dropout=cfg.dropout,
        max_len=cfg.max_len,
        mlp_hidden_dim=cfg.mlp_hidden_dim,
    ).to(cfg.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    pos_weight = _resolve_pos_weight(loaders["train"], cfg.device)

    best_val = -np.inf
    best_state = None
    bad_epochs = 0
    started = time.perf_counter()
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running_loss = 0.0
        for x, y, mask, lengths, _ in loaders["train"]:
            x, y, mask, lengths = (
                x.to(cfg.device), y.to(cfg.device), mask.to(cfg.device), lengths.to(cfg.device)
            )
            optimizer.zero_grad()
            logits = model(x, lengths)["logits_ts"]
            loss = masked_focal_loss(logits, y, mask, gamma=2.0, pos_weight=pos_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += float(loss.item())

        val = _metrics(model, loaders["val"], cfg.device)
        print(
            f"Epoch {epoch:02d} | loss={running_loss / max(len(loaders['train']), 1):.4f} "
            f"| val_AUROC={val['auroc']:.4f} | val_AUPRC={val['auprc']:.4f}"
        )
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

    train = _metrics(model, loaders["train_eval"], cfg.device)
    val = _metrics(model, loaders["val"], cfg.device)
    test = None
    threshold = None
    if not args.validation_only:
        test = _metrics(model, loaders["test"], cfg.device)
        val_y, val_prob = _flatten_loader_probs(model, loaders["val"], cfg.device)
        threshold = _pick_threshold_max_f1(val_y, val_prob)
        test_y, test_prob = _flatten_loader_probs(model, loaders["test"], cfg.device)
        test.update(_threshold_metrics(test_y, test_prob, threshold))

    out_dir = config.run_dir(config.DISEASE) / "TrGRU" / "Centralized"
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "model": "TrGRU adapted replication",
        "feature_mode": config.FEATURE_MODE,
        "n_features": len(datasets["train"].feature_cols),
        "train_auroc": train["auroc"],
        "train_auprc": train["auprc"],
        "val_auroc": val["auroc"],
        "val_auprc": val["auprc"],
        "test_auroc": np.nan if test is None else test["auroc"],
        "test_auprc": np.nan if test is None else test["auprc"],
        "threshold": threshold,
        "test_accuracy": np.nan if test is None else test["accuracy"],
        "test_precision": np.nan if test is None else test["precision"],
        "test_recall": np.nan if test is None else test["recall"],
        "test_f1": np.nan if test is None else test["f1"],
        "test_fpr": np.nan if test is None else test["fpr"],
        "test_tn": np.nan if test is None else test["tn"],
        "test_fp": np.nan if test is None else test["fp"],
        "test_fn": np.nan if test is None else test["fn"],
        "test_tp": np.nan if test is None else test["tp"],
        "runtime_sec": time.perf_counter() - started,
    }
    pd.DataFrame([result]).to_csv(out_dir / "trgru_results.csv", index=False)
    if not args.validation_only:
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "config": asdict(cfg),
                "feature_names": datasets["train"].feature_cols,
                "threshold": threshold,
            },
            out_dir / "trgru_model.pt",
        )
    (out_dir / "trgru_metadata.json").write_text(
        json.dumps({"config": asdict(cfg), "results": result}, indent=2), encoding="utf-8"
    )
    print(pd.DataFrame([result]).to_string(index=False))
    print(f"Saved TrGRU outputs to: {out_dir}")


if __name__ == "__main__":
    main()
