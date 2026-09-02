"""Train independent TrGRU models on each federated node, sequentially."""

from __future__ import annotations

import argparse
import random
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import config
from run_centralized_trgru import TrGRUConfig, _metrics
from run_federated import _write_node_samples_csvs
from sequence_dataset_gru import PatientSequenceDataset, pad_collate
from train_eval_gru import (
    _flatten_loader_probs,
    _pick_threshold_max_f1,
    _resolve_pos_weight,
    _threshold_metrics,
    masked_focal_loss,
)
from trgru_model import TrGRURisk


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train local-only TrGRU node models.")
    parser.add_argument("--node", type=int, default=None, help="Run one node; default runs all nodes.")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--patience", type=int, default=5)
    return parser.parse_args()


def _dataset(split: str, path: Path, cfg: TrGRUConfig) -> PatientSequenceDataset:
    return PatientSequenceDataset(
        split=split,
        disease=config.DISEASE,
        max_len=cfg.max_len,
        seed=cfg.seed,
        normalize=True,
        samples_path=path,
    )


def _model(input_dim: int, cfg: TrGRUConfig) -> TrGRURisk:
    return TrGRURisk(
        input_dim=input_dim,
        d_model=cfg.d_model,
        nhead=cfg.nhead,
        transformer_layers=cfg.transformer_layers,
        dim_feedforward=cfg.dim_feedforward,
        gru_hidden_dim=cfg.gru_hidden_dim,
        gru_layers=cfg.gru_layers,
        dropout=cfg.dropout,
        max_len=cfg.max_len,
        mlp_hidden_dim=cfg.mlp_hidden_dim,
    )


def _train_node(node_id: int, samples_path: Path, out_dir: Path, cfg: TrGRUConfig) -> dict:
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    datasets = {s: _dataset(s, samples_path, cfg) for s in ("train", "val", "test")}
    loaders = {
        "train": DataLoader(datasets["train"], cfg.batch_size, True, collate_fn=pad_collate),
        "train_eval": DataLoader(datasets["train"], cfg.batch_size, False, collate_fn=pad_collate),
        "val": DataLoader(datasets["val"], cfg.batch_size, False, collate_fn=pad_collate),
        "test": DataLoader(datasets["test"], cfg.batch_size, False, collate_fn=pad_collate),
    }
    model = _model(len(datasets["train"].feature_cols), cfg).to(cfg.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    pos_weight = _resolve_pos_weight(loaders["train"], cfg.device)
    best_val = -np.inf
    best_state = None
    bad_epochs = 0
    started = time.perf_counter()

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        for x, y, mask, lengths, _ in loaders["train"]:
            x, y, mask, lengths = (
                x.to(cfg.device), y.to(cfg.device), mask.to(cfg.device), lengths.to(cfg.device)
            )
            optimizer.zero_grad()
            logits = model(x, lengths)["logits_ts"]
            loss = masked_focal_loss(logits, y, mask, gamma=2.0, pos_weight=pos_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        val = _metrics(model, loaders["val"], cfg.device)
        print(f"Node {node_id} epoch {epoch:02d} | val_AUROC={val['auroc']:.4f} | val_AUPRC={val['auprc']:.4f}")
        if np.isfinite(val["auroc"]) and val["auroc"] > best_val:
            best_val = float(val["auroc"])
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
    test = _metrics(model, loaders["test"], cfg.device)
    val_y, val_prob = _flatten_loader_probs(model, loaders["val"], cfg.device)
    threshold = _pick_threshold_max_f1(val_y, val_prob)
    test_y, test_prob = _flatten_loader_probs(model, loaders["test"], cfg.device)
    test.update(_threshold_metrics(test_y, test_prob, threshold))

    node_out = out_dir / f"node_{node_id}"
    node_out.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": asdict(cfg),
            "feature_names": datasets["train"].feature_cols,
            "threshold": threshold,
            "node_id": node_id,
            "model_type": "local_trgru",
        },
        node_out / "model.pt",
    )
    return {
        "node_id": node_id,
        "n_train_patients": len(datasets["train"]),
        "n_val_patients": len(datasets["val"]),
        "n_test_patients": len(datasets["test"]),
        "n_features": len(datasets["train"].feature_cols),
        "train_auroc": train["auroc"],
        "train_auprc": train["auprc"],
        "val_auroc": val["auroc"],
        "val_auprc": val["auprc"],
        "test_auroc": test["auroc"],
        "test_auprc": test["auprc"],
        "threshold": threshold,
        "test_accuracy": test["accuracy"],
        "test_precision": test["precision"],
        "test_recall": test["recall"],
        "test_f1": test["f1"],
        "test_fpr": test["fpr"],
        "runtime_sec": time.perf_counter() - started,
    }


def main() -> None:
    args = _parse_args()
    cfg = TrGRUConfig(epochs=args.epochs, patience=args.patience)
    samples_path = config.samples_path(config.DISEASE)
    out_dir = config.run_dir(config.DISEASE) / "TrGRU" / "Local"
    node_dir = out_dir / "samples"
    out_dir.mkdir(parents=True, exist_ok=True)
    node_ids, node_paths, _ = _write_node_samples_csvs(samples_path, node_dir)
    selected = node_ids if args.node is None else [args.node]
    missing = [node_id for node_id in selected if node_id not in node_paths]
    if missing:
        raise ValueError(f"Unknown node(s) {missing}; available nodes are {node_ids}")

    results_path = out_dir / "local_trgru_results.csv"
    existing = pd.read_csv(results_path) if results_path.exists() else pd.DataFrame()
    rows = []
    for node_id in selected:
        row = _train_node(node_id, node_paths[node_id], out_dir, cfg)
        rows.append(row)
        combined = pd.concat([existing, pd.DataFrame(rows)], ignore_index=True)
        combined = combined.drop_duplicates(subset=["node_id"], keep="last").sort_values("node_id")
        combined.to_csv(results_path, index=False)
        print(pd.DataFrame([row]).to_string(index=False))

    print(f"Saved local TrGRU results to: {results_path}")


if __name__ == "__main__":
    main()
