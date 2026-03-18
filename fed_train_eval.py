# fed_train_eval.py
# Simulated Federated Learning (FedAvg) for your GRU early detection model.
#
# Requirements in your run folder (config.run_dir(disease)):
# - samples_with_node.csv  (must include columns: patientunitstayid, t_end, label, split, node_id)
# - features.parquet       (as usual)
#
# This script:
# 1) Builds per-node samples CSVs (cached in Outputs/<run>/Federated/)
# 2) Runs FedAvg for a small set of reasonable default hyperparams
# 3) Evaluates on GLOBAL val each round, then GLOBAL test at the end (Option A)
# 4) Saves a single-row summary into federated_results.csv (appends if exists)

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Any

import time
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import config
from gru_risk import GRURisk
from sequence_dataset_gru import PatientSequenceDataset, pad_collate

# Reuse your existing training/eval utilities exactly
from train_eval_gru import (
    TrainConfig,
    evaluate,
    masked_focal_loss,
    _model_forward_logits_ts,
    _model_forward_logit_seq,
    _compute_seq_targets,
    _flatten_loader_probs,
    _pick_threshold_max_f1,
    _threshold_metrics,
    _save_test_curves,
    ATTN_AUX_ENABLED,
    ATTN_AUX_WEIGHT,
)


@dataclass
class FedConfig:
    rounds: int = 30
    local_epochs: int = 1
    frac_clients: float = 1.0  # keep 1.0 for now
    seed: int = 42


def _seed_everything(seed: int) -> None:
    seed = int(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Keep this reasonable: deterministic can slow you down a lot
    torch.backends.cudnn.benchmark = True


def _avg_state_dict(
    state_dicts: List[Dict[str, torch.Tensor]],
    weights: List[float],
) -> Dict[str, torch.Tensor]:
    if len(state_dicts) == 0:
        raise ValueError("No state_dicts to aggregate.")
    if len(state_dicts) != len(weights):
        raise ValueError("state_dicts and weights must have same length.")

    w = np.array(weights, dtype=np.float64)
    if not np.isfinite(w).all() or w.sum() <= 0:
        raise ValueError(f"Invalid aggregation weights: {w}")
    w = w / w.sum()

    keys = list(state_dicts[0].keys())
    out: Dict[str, torch.Tensor] = {}

    for k in keys:
        acc = None
        for sd, alpha in zip(state_dicts, w):
            t = sd[k].detach().cpu()
            if acc is None:
                acc = t.mul(float(alpha))
            else:
                acc.add_(t, alpha=float(alpha))
        out[k] = acc

    return out


def _write_node_samples_csvs(
    samples_with_node_csv: Path,
    out_dir: Path,
) -> Tuple[List[int], Dict[int, Path], Dict[int, int]]:
    df = pd.read_csv(samples_with_node_csv)
    need = {"patientunitstayid", "t_end", "label", "split", "node_id"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"samples_with_node.csv missing columns: {sorted(missing)}")

    node_ids = sorted(df["node_id"].dropna().unique().astype(int).tolist())
    if len(node_ids) == 0:
        raise RuntimeError("No node_id values found in samples_with_node.csv")

    out_dir.mkdir(parents=True, exist_ok=True)

    node_csv_paths: Dict[int, Path] = {}
    node_train_sizes: Dict[int, int] = {}

    for nid in node_ids:
        node_df = df[df["node_id"] == int(nid)].copy()
        if len(node_df) == 0:
            continue

        out_path = out_dir / f"samples_node{nid}.csv"
        node_df.to_csv(out_path, index=False)

        n_train = int((node_df["split"].astype(str).str.lower() == "train").sum())
        node_csv_paths[int(nid)] = out_path
        node_train_sizes[int(nid)] = n_train

    # sanity
    bad_nodes = [nid for nid, ntr in node_train_sizes.items() if ntr <= 0]
    if bad_nodes:
        raise RuntimeError(
            f"Some nodes have 0 TRAIN rows (cannot train): {bad_nodes}. "
            "Check your samples_with_node.csv split assignment."
        )

    return node_ids, node_csv_paths, node_train_sizes


def _local_train_one_client(
    model: nn.Module,
    train_loader: DataLoader,
    cfg: TrainConfig,
    local_epochs: int,
) -> None:
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    for _ in range(int(local_epochs)):
        for x, y, mask, lengths in train_loader:
            x = x.to(cfg.device)
            y = y.to(cfg.device)
            mask = mask.to(cfg.device)

            opt.zero_grad()

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
            opt.step()


def train_and_eval_fedavg(
    disease: config.DiseaseSpec,
    cfg: TrainConfig,
    fed: FedConfig,
    top_k: int | None = None,
    rank_path: str | None = None,
) -> Dict[str, Any]:
    _seed_everything(fed.seed)
    rng = np.random.default_rng(int(fed.seed))

    run_dir = config.run_dir(disease)
    samples_with_node_csv = run_dir / "samples_with_node.csv"
    if not samples_with_node_csv.exists():
        raise FileNotFoundError(
            f"Missing {samples_with_node_csv}.\n"
            "Generate it first (samples_with_node.csv + hospital_to_node.json)."
        )

    cache_dir = run_dir / "Federated"
    node_ids, node_csv_paths, node_train_sizes = _write_node_samples_csvs(samples_with_node_csv, cache_dir)

    # GLOBAL val/test (Option A)
    # Important: we pass samples_path=samples_with_node_csv so it uses the same file layout.
    val_ds = PatientSequenceDataset(
        split="val",
        disease=disease,
        max_len=cfg.max_len,
        seed=cfg.seed,
        normalize=True,
        top_k=top_k,
        rank_path=rank_path,
        samples_path=samples_with_node_csv,
    )
    test_ds = PatientSequenceDataset(
        split="test",
        disease=disease,
        max_len=cfg.max_len,
        seed=cfg.seed,
        normalize=True,
        top_k=top_k,
        rank_path=rank_path,
        samples_path=samples_with_node_csv,
    )

    val_loader = DataLoader(val_ds, cfg.batch_size, False, collate_fn=pad_collate)
    test_loader = DataLoader(test_ds, cfg.batch_size, False, collate_fn=pad_collate)

    # Build global model (feature set must be consistent)
    model_global = GRURisk(
        input_dim=len(val_ds.feature_cols),
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
        use_layernorm=True,
        use_attention_pooling=True,
    ).to(cfg.device)

    best_val_auroc = -1.0
    best_state = None
    history_rows: List[Dict[str, float]] = []

    t0 = time.perf_counter()

    for r in range(int(fed.rounds)):
        # client sampling (keep all by default)
        active = node_ids
        if fed.frac_clients < 1.0 and len(node_ids) > 1:
            m = max(1, int(round(float(fed.frac_clients) * len(node_ids))))
            active = rng.choice(node_ids, size=m, replace=False).tolist()

        local_states: List[Dict[str, torch.Tensor]] = []
        local_weights: List[float] = []

        for nid in active:
            train_ds = PatientSequenceDataset(
                split="train",
                disease=disease,
                max_len=cfg.max_len,
                seed=cfg.seed,
                normalize=True,  # node-local normalization (computed from node train split in that CSV)
                top_k=top_k,
                rank_path=rank_path,
                samples_path=node_csv_paths[int(nid)],
            )
            train_loader = DataLoader(train_ds, cfg.batch_size, True, collate_fn=pad_collate)

            local_model = GRURisk(
                input_dim=len(train_ds.feature_cols),
                hidden_dim=cfg.hidden_dim,
                num_layers=cfg.num_layers,
                dropout=cfg.dropout,
                use_layernorm=True,
                use_attention_pooling=True,
            ).to(cfg.device)

            local_model.load_state_dict(model_global.state_dict())

            _local_train_one_client(local_model, train_loader, cfg, fed.local_epochs)

            local_states.append({k: v.detach().cpu().clone() for k, v in local_model.state_dict().items()})
            local_weights.append(float(node_train_sizes[int(nid)]))

            del local_model
            if str(cfg.device).startswith("cuda"):
                torch.cuda.empty_cache()

        new_state = _avg_state_dict(local_states, local_weights)
        model_global.load_state_dict(new_state)

        val_metrics = evaluate(model_global, val_loader, cfg.device)
        va = float(val_metrics.get("auroc", float("nan")))
        vp = float(val_metrics.get("auprc", float("nan")))

        history_rows.append({"round": float(r + 1), "val_auroc": va, "val_auprc": vp})

        if np.isfinite(va) and va > best_val_auroc:
            best_val_auroc = va
            best_state = {k: v.detach().cpu().clone() for k, v in model_global.state_dict().items()}

        print(f"[Round {r+1}/{fed.rounds}] val_auroc={va:.4f} val_auprc={vp:.4f}")

    if best_state is not None:
        model_global.load_state_dict(best_state)

    # Threshold chosen on GLOBAL val, applied on GLOBAL test (parity with centralized)
    yv_true, yv_prob = _flatten_loader_probs(model_global, val_loader, device=cfg.device)
    chosen_threshold = 0.5 if yv_true.size == 0 else _pick_threshold_max_f1(yv_true, yv_prob)

    test_metrics = evaluate(model_global, test_loader, cfg.device)

    yt_true, yt_prob = _flatten_loader_probs(model_global, test_loader, device=cfg.device)
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
        model=model_global,
        test_loader=test_loader,
        disease=disease,
        top_k=top_k,
        device=cfg.device,
    )

    runtime = float(time.perf_counter() - t0)

    return {
        "fed": {
            "rounds": int(fed.rounds),
            "local_epochs": int(fed.local_epochs),
            "frac_clients": float(fed.frac_clients),
            "seed": int(fed.seed),
            "n_nodes": int(len(node_ids)),
            "node_ids": [int(x) for x in node_ids],
            "node_train_sizes": {int(k): int(v) for k, v in node_train_sizes.items()},
            "best_val_auroc": float(best_val_auroc),
            "runtime_sec": runtime,
        },
        "test": test_metrics,
        "threshold": float(chosen_threshold),
        "curve_paths": curve_paths,
        "history": history_rows,
        "n_features": int(len(val_ds.feature_cols)),
    }


def _save_fed_results_csv(
    out: Dict[str, Any],
    disease: config.DiseaseSpec,
    cfg: TrainConfig,
    fed: FedConfig,
    top_k: int | None,
    rank_path: str | None,
) -> Path:
    run_dir = config.run_dir(disease)
    csv_path = run_dir / "federated_results.csv"

    test = out.get("test", {}) or {}
    fed_info = out.get("fed", {}) or {}

    row = {
        "model": "fedavg_gru",
        "disease": getattr(disease, "name", str(disease)),
        "top_k": (int(top_k) if top_k is not None else ""),
        "rank_path": (str(rank_path) if rank_path is not None else ""),
        "rounds": int(fed_info.get("rounds", fed.rounds)),
        "local_epochs": int(fed_info.get("local_epochs", fed.local_epochs)),
        "frac_clients": float(fed_info.get("frac_clients", fed.frac_clients)),
        "n_nodes": int(fed_info.get("n_nodes", "" if "n_nodes" not in fed_info else fed_info["n_nodes"])),
        "seed": int(fed_info.get("seed", fed.seed)),
        "max_len": int(cfg.max_len),
        "batch_size": int(cfg.batch_size),
        "hidden_dim": int(cfg.hidden_dim),
        "num_layers": int(cfg.num_layers),
        "dropout": float(cfg.dropout),
        "lr": float(cfg.lr),
        "weight_decay": float(cfg.weight_decay),
        "n_features": int(out.get("n_features", -1)),
        "best_val_auroc": float(fed_info.get("best_val_auroc", float("nan"))),
        "test_loss": float(test.get("loss", float("nan"))),
        "test_auroc": float(test.get("auroc", float("nan"))),
        "test_auprc": float(test.get("auprc", float("nan"))),
        "test_accuracy": float(test.get("accuracy", float("nan"))),
        "test_precision": float(test.get("precision", float("nan"))),
        "test_recall": float(test.get("recall", float("nan"))),
        "test_f1": float(test.get("f1", float("nan"))),
        "test_fpr": float(test.get("fpr", float("nan"))),
        "threshold": float(out.get("threshold", float("nan"))),
        "runtime_sec": float(fed_info.get("runtime_sec", float("nan"))),
        "roc_path": (out.get("curve_paths", {}) or {}).get("roc_path", ""),
        "pr_path": (out.get("curve_paths", {}) or {}).get("pr_path", ""),
    }

    df_row = pd.DataFrame([row])

    if csv_path.exists():
        df_old = pd.read_csv(csv_path)
        df_new = pd.concat([df_old, df_row], ignore_index=True)
    else:
        df_new = df_row

    df_new.to_csv(csv_path, index=False)
    return csv_path


def _pick_disease_from_config() -> config.DiseaseSpec:
    d = getattr(config, "DISEASE", None)
    if d is not None:
        return d

    ds = getattr(config, "DISEASES", None)
    if isinstance(ds, dict) and len(ds) > 0:
        # pick first deterministically
        key = sorted(ds.keys())[0]
        print(f"WARNING: config.DISEASE not set. Using first disease from config.DISEASES: {key}")
        return ds[key]

    raise RuntimeError("Could not determine disease. Set config.DISEASE or config.DISEASES.")


if __name__ == "__main__":
    # Reasonable defaults for a first federated baseline.
    # You can tune later.
    disease = _pick_disease_from_config()

    cfg = TrainConfig(
        max_len=128,
        batch_size=32,
        hidden_dim=128,
        num_layers=1,
        dropout=0.2,
        lr=5e-4,          # slightly lower than 1e-3 for stability under non-IID
        weight_decay=1e-5,
        epochs=1,         # not used in FL loop (local_epochs controls)
        patience=3,       # not used in FL loop
        seed=42,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    fed = FedConfig(
        rounds=30,
        local_epochs=1,
        frac_clients=1.0,
        seed=42,
    )

    top_k = None
    rank_path = None

    print("============================================================")
    print("Federated GRU (FedAvg) run")
    print("Disease:", getattr(disease, "name", str(disease)))
    print("Run dir:", config.run_dir(disease))
    print("Device:", cfg.device)
    print(f"Rounds={fed.rounds}  LocalEpochs={fed.local_epochs}  LR={cfg.lr}")
    print("============================================================")

    out = train_and_eval_fedavg(
        disease=disease,
        cfg=cfg,
        fed=fed,
        top_k=top_k,
        rank_path=rank_path,
    )

    csv_path = _save_fed_results_csv(out, disease, cfg, fed, top_k=top_k, rank_path=rank_path)
    print("Saved CSV:", csv_path)
    print("Final TEST metrics:", out.get("test", {}))