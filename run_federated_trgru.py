"""Federated FedAvg/FedProx training for the adapted TrGRU model.

The run is resumable after every completed communication round. Test data are
evaluated only after all rounds finish and the best validation state is loaded.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

import config
from run_centralized_trgru import TrGRUConfig, _metrics
from run_federated import _avg_state_dict, _write_node_samples_csvs
from sequence_dataset_gru import PatientSequenceDataset, pad_collate
from train_eval_gru import (
    _flatten_loader_probs,
    _pick_threshold_max_f1,
    _resolve_pos_weight,
    _threshold_metrics,
    evaluate,
    masked_focal_loss,
)
from trgru_model import TrGRURisk


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run federated TrGRU with FedAvg/FedProx.")
    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--mu", type=float, default=0.01)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional isolated output directory (default: the main Federated folder).",
    )
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_model(input_dim: int, cfg: TrGRUConfig) -> TrGRURisk:
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


def _dataset(split: str, samples_path: Path, cfg: TrGRUConfig) -> PatientSequenceDataset:
    return PatientSequenceDataset(
        split=split,
        disease=config.DISEASE,
        max_len=cfg.max_len,
        seed=cfg.seed,
        normalize=True,
        samples_path=samples_path,
    )


def _train_client(
    model: TrGRURisk,
    loader: DataLoader,
    cfg: TrGRUConfig,
    local_epochs: int,
    global_parameters: dict[str, torch.Tensor],
    mu: float,
) -> None:
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    pos_weight = _resolve_pos_weight(loader, cfg.device)
    model.train()
    for _ in range(local_epochs):
        for x, y, mask, lengths, _ in loader:
            x, y, mask, lengths = (
                x.to(cfg.device), y.to(cfg.device), mask.to(cfg.device), lengths.to(cfg.device)
            )
            optimizer.zero_grad()
            logits = model(x, lengths)["logits_ts"]
            task_loss = masked_focal_loss(logits, y, mask, gamma=2.0, pos_weight=pos_weight)
            if np.isclose(mu, 0.0):
                loss = task_loss
            else:
                proximal_penalty = sum(
                    torch.sum((parameter - global_parameters[name]) ** 2)
                    for name, parameter in model.named_parameters()
                )
                loss = task_loss + 0.5 * float(mu) * proximal_penalty
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()


def _save_federated_auprc_plot(history: list[dict[str, float]], out_dir: Path) -> None:
    history_df = pd.DataFrame(history)
    required = {"round", "val_auprc"}
    if not required.issubset(history_df.columns):
        raise ValueError("Federated history must contain round and val_auprc")
    if not np.isfinite(history_df[list(required)].to_numpy(dtype=float)).all():
        raise ValueError("Federated AUPRC history contains non-finite values")
    diagrams_dir = (
        out_dir.parent / "Diagrams" / "Federated"
        if out_dir.name == "Federated"
        else out_dir / "Diagrams"
    )
    diagrams_dir.mkdir(parents=True, exist_ok=True)
    central_results = config.run_dir(config.DISEASE) / "TrGRU" / "Centralized" / "trgru_results.csv"
    plt.figure(figsize=(7, 5))
    plt.plot(history_df["round"], history_df["val_auprc"], marker="o", linewidth=2,
             label="Federated (validation)")
    if central_results.exists():
        baseline = float(pd.read_csv(central_results).iloc[0]["val_auprc"])
        plt.axhline(baseline, color="gray", linestyle="--", linewidth=1.5,
                    label="Centralised baseline (validation)")
    plt.xlabel("Federated round")
    plt.ylabel("AUPRC")
    plt.title("Validation AUPRC across communication rounds")
    plt.ylim(0.0, 1.0)
    plt.grid(alpha=0.3)
    plt.legend(loc="best")
    plt.tight_layout()
    path = diagrams_dir / "federated_validation_auprc.png"
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"Saved federated TrGRU AUPRC diagram to: {path}")


def _save_federated_plots(history: list[dict[str, float]], out_dir: Path) -> None:
    history_df = pd.DataFrame(history)
    required = {"round", "val_loss", "val_auroc"}
    missing = required - set(history_df.columns)
    if missing:
        raise ValueError(f"Federated history is missing plot columns: {sorted(missing)}")

    diagrams_dir = (
        out_dir.parent / "Diagrams" / "Federated"
        if out_dir.name == "Federated"
        else out_dir / "Diagrams"
    )
    diagrams_dir.mkdir(parents=True, exist_ok=True)
    central_results = (
        config.run_dir(config.DISEASE)
        / "TrGRU"
        / "Centralized"
        / "trgru_results.csv"
    )
    central_auroc = None
    if central_results.exists():
        central_auroc = float(pd.read_csv(central_results).iloc[0]["test_auroc"])

    plt.figure(figsize=(7, 5))
    plt.plot(history_df["round"], history_df["val_auroc"], marker="o", linewidth=2,
             label="Federated")
    if central_auroc is not None:
        plt.axhline(central_auroc, color="gray", linestyle="--", linewidth=1.5,
                    label="Centralised baseline")
    plt.xlabel("Federated round")
    plt.ylabel("AUROC")
    plt.title("Validation AUROC across communication rounds")
    plt.grid(alpha=0.3)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(diagrams_dir / "federated_validation_auroc.png", dpi=200)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.plot(history_df["round"], history_df["val_loss"], marker="o", linewidth=2,
             color="tab:red")
    plt.xlabel("Federated round")
    plt.ylabel("Loss")
    plt.title("Validation loss across communication rounds")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(diagrams_dir / "federated_validation_loss.png", dpi=200)
    plt.close()
    _save_federated_auprc_plot(history, out_dir)
    print(f"Saved federated TrGRU diagrams to: {diagrams_dir}")


def main() -> None:
    args = _parse_args()
    if args.mu < 0:
        raise ValueError("--mu must be non-negative")
    algorithm = "fedavg" if np.isclose(args.mu, 0.0) else "fedprox"
    cfg = TrGRUConfig(lr=args.lr, epochs=1, patience=1)
    _seed(cfg.seed)

    samples_path = config.samples_path(config.DISEASE)
    out_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else config.run_dir(config.DISEASE) / "TrGRU" / "Federated"
    )
    node_dir = out_dir / "nodes"
    out_dir.mkdir(parents=True, exist_ok=True)
    node_ids, node_paths, node_sizes = _write_node_samples_csvs(samples_path, node_dir)

    val_ds = _dataset("val", samples_path, cfg)
    test_ds = _dataset("test", samples_path, cfg)
    val_loader = DataLoader(val_ds, cfg.batch_size, False, collate_fn=pad_collate)
    test_loader = DataLoader(test_ds, cfg.batch_size, False, collate_fn=pad_collate)
    global_model = _build_model(len(val_ds.feature_cols), cfg).to(cfg.device)

    resume_path = out_dir / "resume_checkpoint.pt"
    start_round = 0
    best_val_auroc = -np.inf
    best_state = None
    history: list[dict[str, float]] = []
    if resume_path.exists() and not args.no_resume:
        saved = torch.load(resume_path, map_location="cpu", weights_only=False)
        if saved.get("algorithm") != algorithm or not np.isclose(
            float(saved.get("mu", np.nan)), float(args.mu)
        ):
            raise ValueError(
                "The existing resume checkpoint is not from this federated configuration. "
                "Delete the federated output folder or rerun with --no-resume."
            )
        global_model.load_state_dict(saved["global_state"])
        best_state = saved.get("best_state")
        best_val_auroc = float(saved.get("best_val_auroc", -np.inf))
        history = list(saved.get("history", []))
        start_round = int(saved["completed_round"])
        print(f"Resuming after round {start_round} from {resume_path}")

    started = time.perf_counter()
    for round_index in range(start_round, args.rounds):
        local_states = []
        local_weights = []
        global_parameters = {
            name: parameter.detach().clone()
            for name, parameter in global_model.named_parameters()
        }
        for node_id in node_ids:
            train_ds = _dataset("train", node_paths[node_id], cfg)
            train_loader = DataLoader(train_ds, cfg.batch_size, True, collate_fn=pad_collate)
            local_model = _build_model(len(train_ds.feature_cols), cfg).to(cfg.device)
            local_model.load_state_dict(global_model.state_dict())
            _train_client(
                local_model,
                train_loader,
                cfg,
                args.local_epochs,
                global_parameters,
                args.mu,
            )
            local_states.append({k: v.detach().cpu().clone() for k, v in local_model.state_dict().items()})
            local_weights.append(float(node_sizes[node_id]))
            del local_model
            if str(cfg.device).startswith("cuda"):
                torch.cuda.empty_cache()

        global_model.load_state_dict(_avg_state_dict(local_states, local_weights))
        val = evaluate(global_model, val_loader, cfg.device)
        history.append(
            {
                "round": float(round_index + 1),
                "val_loss": float(val["loss"]),
                "val_auroc": float(val["auroc"]),
                "val_auprc": float(val["auprc"]),
            }
        )
        if np.isfinite(val["auroc"]) and val["auroc"] > best_val_auroc:
            best_val_auroc = float(val["auroc"])
            best_state = {k: v.detach().cpu().clone() for k, v in global_model.state_dict().items()}

        torch.save(
            {
                "algorithm": algorithm,
                "mu": float(args.mu),
                "completed_round": round_index + 1,
                "global_state": global_model.state_dict(),
                "best_state": best_state,
                "best_val_auroc": best_val_auroc,
                "history": history,
            },
            resume_path,
        )
        pd.DataFrame(history).to_csv(out_dir / "validation_history.csv", index=False)
        print(
            f"Round {round_index + 1:02d}/{args.rounds} | "
            f"val_loss={val['loss']:.4f} | val_AUROC={val['auroc']:.4f} "
            f"| val_AUPRC={val['auprc']:.4f}"
        )

    if best_state is not None:
        global_model.load_state_dict(best_state)
    val = _metrics(global_model, val_loader, cfg.device)
    test = _metrics(global_model, test_loader, cfg.device)
    val_y, val_prob = _flatten_loader_probs(global_model, val_loader, cfg.device)
    threshold = _pick_threshold_max_f1(val_y, val_prob)
    test_y, test_prob = _flatten_loader_probs(global_model, test_loader, cfg.device)
    test.update(_threshold_metrics(test_y, test_prob, threshold))

    result = {
        "model": f"{algorithm}_trgru",
        "algorithm": algorithm,
        "mu": float(args.mu),
        "n_features": len(val_ds.feature_cols),
        "n_nodes": len(node_ids),
        "rounds": args.rounds,
        "local_epochs": args.local_epochs,
        "best_val_auroc": best_val_auroc,
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
        "test_tn": test["tn"],
        "test_fp": test["fp"],
        "test_fn": test["fn"],
        "test_tp": test["tp"],
        "runtime_sec_this_session": time.perf_counter() - started,
    }
    pd.DataFrame([result]).to_csv(out_dir / "federated_trgru_results.csv", index=False)
    torch.save(
        {
            "model_state_dict": global_model.state_dict(),
            "config": asdict(cfg),
            "feature_names": val_ds.feature_cols,
            "threshold": threshold,
            "model_type": f"{algorithm}_trgru",
            "algorithm": algorithm,
            "mu": float(args.mu),
        },
        out_dir / "model.pt",
    )
    (out_dir / "metadata.json").write_text(
        json.dumps({"config": asdict(cfg), "result": result}, indent=2), encoding="utf-8"
    )
    _save_federated_plots(history, out_dir)
    print(pd.DataFrame([result]).to_string(index=False))
    print(f"Saved federated TrGRU outputs to: {out_dir}")


if __name__ == "__main__":
    main()
