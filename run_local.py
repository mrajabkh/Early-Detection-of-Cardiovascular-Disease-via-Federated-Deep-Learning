from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import config
import xai
from run_federated import _write_node_samples_csvs
from gru_model import GRURisk
from sequence_dataset_gru import PatientSequenceDataset, pad_collate
from train_eval_gru import (
    TrainConfig,
    evaluate,
    masked_focal_loss,
    _resolve_pos_weight,
    _flatten_loader_probs,
    _model_forward_logits_ts,
    _pick_threshold_max_f1,
    _threshold_metrics,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train local GRUs for all nodes or one selected node."
    )
    parser.add_argument(
        "--node",
        type=int,
        default=None,
        help="Run one node; omission runs every node sequentially.",
    )
    parser.add_argument(
        "--samples-with-node",
        type=Path,
        default=None,
        help=(
            "Path to the samples CSV containing node_id. Defaults to config.samples_path(config.DISEASE)."
        ),
    )
    parser.add_argument("--epochs", type=int, default=20, help="Local node training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Local node batch size")
    parser.add_argument("--output-dir", type=Path, default=None, help="Optional output directory for JSON results")
    parser.add_argument(
        "--compare-saved",
        action="store_true",
        help="Also evaluate saved centralized/federated GRUs on every node.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Retrain nodes already present in local_gru_results.csv.",
    )
    return parser.parse_args()


def _find_latest_metadata(artifact_root: Path, pattern: str, exclude_prefix: Optional[str] = None) -> Optional[Path]:
    files = sorted(artifact_root.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    if exclude_prefix is not None:
        files = [f for f in files if not f.name.startswith(exclude_prefix)]
    return files[0] if files else None


def _load_saved_model(model_type: str, device: str) -> Optional[Dict[str, object]]:
    if model_type == "centralized":
        artifact_root = config.run_dir(config.DISEASE)
        metadata_path = _find_latest_metadata(artifact_root, "metadata__*.json", exclude_prefix="metadata__fedavg__")
        if metadata_path is None:
            print("WARNING: No centralized metadata file found.")
            return None
        with metadata_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
        run_tag = xai._run_tag_from_metadata(meta)
        model_path = artifact_root / f"model__{run_tag}.pt"
    elif model_type == "federated":
        artifact_root = config.run_dir(config.DISEASE) / "Federated"
        metadata_path = _find_latest_metadata(artifact_root, "metadata__fedavg__*.json")
        if metadata_path is None:
            print("WARNING: No federated metadata file found.")
            return None
        with metadata_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
        run_tag = xai._run_tag_from_metadata(meta)
        model_path = artifact_root / f"model__fedavg__{run_tag}.pt"
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    if not model_path.exists():
        print(f"WARNING: Saved checkpoint not found: {model_path}")
        return None

    ckpt = torch.load(model_path, map_location=device)
    model = xai._build_model_from_checkpoint(ckpt, device=device)
    return {
        "label": f"{model_type}_saved",
        "model": model,
        "meta": meta,
        "model_path": str(model_path.resolve()),
    }


def _build_dataset_from_meta(split: str, samples_path: Path, meta: Dict, cfg: TrainConfig) -> PatientSequenceDataset:
    feature_mode = str(getattr(config, "FEATURE_MODE", "all")).strip().lower()
    top_k = meta.get("top_k", None) if feature_mode == "all" else None
    rank_path = str(config.stability_combined_path(config.DISEASE)) if top_k is not None else None
    return PatientSequenceDataset(
        split=split,
        disease=config.DISEASE,
        max_len=int(meta["max_len"]),
        seed=int(meta.get("seed", cfg.seed)),
        normalize=True,
        top_k=top_k,
        rank_path=rank_path,
        samples_path=samples_path,
    )


def _evaluate_saved_model(saved: Dict[str, object], samples_path: Path, cfg: TrainConfig) -> Dict[str, object]:
    model = saved["model"]
    meta = saved["meta"]

    val_ds = _build_dataset_from_meta("val", samples_path, meta, cfg)
    test_ds = _build_dataset_from_meta("test", samples_path, meta, cfg)

    val_loader = DataLoader(val_ds, cfg.batch_size, False, collate_fn=pad_collate)
    test_loader = DataLoader(test_ds, cfg.batch_size, False, collate_fn=pad_collate)

    val_metrics = evaluate(model, val_loader, cfg.device)
    test_metrics = evaluate(model, test_loader, cfg.device)

    yv_true, yv_prob = _flatten_loader_probs(model, val_loader, device=cfg.device)
    threshold = 0.5 if yv_true.size == 0 else _pick_threshold_max_f1(yv_true, yv_prob)
    yt_true, yt_prob = _flatten_loader_probs(model, test_loader, device=cfg.device)
    test_threshold_metrics = _threshold_metrics(yt_true, yt_prob, threshold) if yt_true.size else {}

    return {
        "label": saved["label"],
        "model_path": saved["model_path"],
        "train": None,
        "val": val_metrics,
        "test": test_metrics,
        "threshold": float(threshold),
        "test_threshold_metrics": test_threshold_metrics,
    }


def _build_loader(split: str, samples_path: Path, cfg: TrainConfig) -> DataLoader:
    feature_mode = str(getattr(config, "FEATURE_MODE", "all")).strip().lower()
    if feature_mode == "all":
        top_k = int(getattr(config, "DEFAULT_TOPK", 60))
        rank_path = str(config.stability_combined_path(config.DISEASE))
    else:
        top_k = None
        rank_path = None

    ds = PatientSequenceDataset(
        split=split,
        disease=config.DISEASE,
        max_len=cfg.max_len,
        seed=cfg.seed,
        normalize=True,
        top_k=top_k,
        rank_path=rank_path,
        samples_path=samples_path,
    )
    return DataLoader(ds, batch_size=cfg.batch_size, shuffle=(split == "train"), collate_fn=pad_collate)


def _train_local_node(train_samples_path: Path, cfg: TrainConfig) -> Dict[str, object]:
    train_loader = _build_loader("train", train_samples_path, cfg)
    val_loader = _build_loader("val", train_samples_path, cfg)
    test_loader = _build_loader("test", train_samples_path, cfg)

    feature_count = len(train_loader.dataset.feature_cols)
    model = GRURisk(
        input_dim=feature_count,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
        use_layernorm=True,
        use_attention_pooling=True,
    ).to(cfg.device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    pos_weight = _resolve_pos_weight(train_loader, cfg.device)
    best_val_auroc = -1.0
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
            loss = masked_focal_loss(logits_ts, y, mask, gamma=2.0, alpha=None, pos_weight=pos_weight)
            loss.backward()
            optimizer.step()

        val_metrics = evaluate(model, val_loader, cfg.device)
        if val_metrics["auroc"] > best_val_auroc:
            best_val_auroc = val_metrics["auroc"]
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
    threshold = 0.5 if yv_true.size == 0 else _pick_threshold_max_f1(yv_true, yv_prob)
    yt_true, yt_prob = _flatten_loader_probs(model, test_loader, device=cfg.device)
    test_threshold_metrics = _threshold_metrics(yt_true, yt_prob, threshold) if yt_true.size else {}

    return {
        "label": "local_node",
        "train": train_metrics,
        "val": val_metrics,
        "test": test_metrics,
        "threshold": float(threshold),
        "test_threshold_metrics": test_threshold_metrics,
    }


def _node_csv_paths(samples_with_node: Path) -> Dict[int, Path]:
    run_dir = config.run_dir(config.DISEASE)
    cache_dir = run_dir / "NodeRun"
    cache_dir.mkdir(parents=True, exist_ok=True)
    _, node_csv_paths, _ = _write_node_samples_csvs(samples_with_node, cache_dir)
    return node_csv_paths


def main() -> None:
    args = _parse_args()
    samples_with_node = args.samples_with_node or config.samples_path(config.DISEASE)
    samples_with_node = Path(samples_with_node)

    if not samples_with_node.exists():
        raise FileNotFoundError(f"Missing samples file: {samples_with_node.resolve()}")

    node_csv_paths = _node_csv_paths(samples_with_node)
    selected_nodes = sorted(node_csv_paths) if args.node is None else [args.node]
    missing_nodes = [node_id for node_id in selected_nodes if node_id not in node_csv_paths]
    if missing_nodes:
        raise ValueError(
            f"Node(s) {missing_nodes} not found. Available nodes: {sorted(node_csv_paths.keys())}"
        )

    cfg = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    output_dir = args.output_dir or (config.run_dir(config.DISEASE) / "GRU" / "Local")
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "local_gru_results.csv"
    existing = pd.read_csv(summary_path) if summary_path.exists() else pd.DataFrame()
    if args.node is None and not args.force and "node_id" in existing.columns:
        completed_nodes = set(existing["node_id"].astype(int).tolist())
        selected_nodes = [node_id for node_id in selected_nodes if node_id not in completed_nodes]
        if completed_nodes:
            print(f"Skipping completed nodes already in CSV: {sorted(completed_nodes)}")
    new_rows = []

    for node_id in selected_nodes:
        node_samples = node_csv_paths[node_id]
        print(f"Selected node: {node_id}")
        print(f"Node-specific samples: {node_samples}")
        print(f"Full centralized samples: {samples_with_node}\n")

        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg.seed)

        results: Dict[str, object] = {
            "node_id": node_id,
            "samples_with_node": str(samples_with_node.resolve()),
            "node_samples": str(node_samples.resolve()),
            "device": cfg.device,
            "config": {
                "epochs": cfg.epochs,
                "batch_size": cfg.batch_size,
                "lr": cfg.lr,
                "weight_decay": cfg.weight_decay,
                "hidden_dim": cfg.hidden_dim,
                "num_layers": cfg.num_layers,
                "dropout": cfg.dropout,
            },
            "runs": [],
        }

        print("Running local node-only model (trained from scratch)...\n")
        local_result = _train_local_node(node_samples, cfg)
        results["runs"].append(local_result)

        if args.compare_saved:
            centralized_saved = _load_saved_model("centralized", cfg.device)
            if centralized_saved is not None:
                print("Evaluating saved centralized model on node data...\n")
                results["runs"].append(_evaluate_saved_model(centralized_saved, node_samples, cfg))

            federated_saved = _load_saved_model("federated", cfg.device)
            if federated_saved is not None:
                print("Evaluating saved federated model on node data...\n")
                results["runs"].append(_evaluate_saved_model(federated_saved, node_samples, cfg))

        out_path = output_dir / f"node_{node_id}_comparison.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        tm = local_result["test_threshold_metrics"]
        new_rows.append(
            {
                "node_id": node_id,
                "train_auroc": local_result["train"]["auroc"],
                "train_auprc": local_result["train"]["auprc"],
                "val_auroc": local_result["val"]["auroc"],
                "val_auprc": local_result["val"]["auprc"],
                "test_auroc": local_result["test"]["auroc"],
                "test_auprc": local_result["test"]["auprc"],
                "threshold": local_result["threshold"],
                "test_accuracy": tm.get("accuracy", float("nan")),
                "test_precision": tm.get("precision", float("nan")),
                "test_recall": tm.get("recall", float("nan")),
                "test_f1": tm.get("f1", float("nan")),
                "test_fpr": tm.get("fpr", float("nan")),
            }
        )
        combined = pd.concat([existing, pd.DataFrame(new_rows)], ignore_index=True)
        combined = combined.drop_duplicates(subset=["node_id"], keep="last").sort_values("node_id")
        combined.to_csv(summary_path, index=False)

        print(f"Saved node results to {out_path.resolve()}")
        print(f"Node {node_id} test AUROC: {local_result['test']['auroc']:.4f}")
        print(f"Node {node_id} test AUPRC: {local_result['test']['auprc']:.4f}\n")

    print(f"Saved consolidated local GRU results to: {summary_path.resolve()}")


if __name__ == "__main__":
    main()
