"""Build and run the reproducible six-bin TrGRU prediction-horizon study."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import config
from run_centralized_trgru import TrGRUConfig, _metrics
from sequence_dataset_gru import PatientSequenceDataset, pad_collate
from train_eval_gru import (
    _flatten_loader_probs,
    _pick_threshold_max_f1,
    _resolve_pos_weight,
    _threshold_metrics,
    masked_focal_loss,
)
from trgru_model import TrGRURisk


TIME_BINS = tuple(config.PREDICTION_HORIZON_BINS)
KEY_COLUMNS = ["patientunitstayid", "t_end"]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare features and run all six TrGRU prediction-horizon bins."
    )
    parser.add_argument(
        "--force-features", action="store_true", help="Rebuild horizon features even if current."
    )
    parser.add_argument(
        "--force-training", action="store_true", help="Retrain completed horizon bins."
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fingerprint(paths: list[Path], extra: dict[str, object]) -> str:
    digest = hashlib.sha256(json.dumps(extra, sort_keys=True).encode("utf-8"))
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(_sha256(path).encode("ascii"))
    return digest.hexdigest()


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _validate_master(master: pd.DataFrame) -> None:
    required = {"patientunitstayid", "t_end", "t_event", "split"}
    missing = required - set(master.columns)
    if missing:
        raise ValueError(f"Fixed cohort is missing columns: {sorted(missing)}")
    if master.duplicated(KEY_COLUMNS).any():
        raise ValueError("Fixed cohort contains duplicate patient/time keys")
    split_counts = master.groupby("patientunitstayid")["split"].nunique()
    if (split_counts > 1).any():
        raise ValueError("Fixed cohort has patient leakage across data splits")
    invalid_splits = set(master["split"].dropna().unique()) - {"train", "val", "test"}
    if invalid_splits:
        raise ValueError(f"Fixed cohort contains invalid splits: {sorted(invalid_splits)}")


def _validate_protocol() -> None:
    if not TIME_BINS:
        raise ValueError("PREDICTION_HORIZON_BINS cannot be empty")
    names = [name for name, _, _ in TIME_BINS]
    if len(names) != len(set(names)):
        raise ValueError("Prediction-horizon bin names must be unique")
    if int(TIME_BINS[0][1]) != int(config.LEAD_TIME_MINS):
        raise ValueError("The first horizon bin must start at LEAD_TIME_MINS")
    previous_high = None
    for name, low, high in TIME_BINS:
        if int(low) >= int(high):
            raise ValueError(f"Invalid prediction-horizon bounds for {name}: ({low}, {high}]")
        if previous_high is not None and int(low) != int(previous_high):
            raise ValueError("Prediction-horizon bins must be contiguous and non-overlapping")
        previous_high = high
    if int(TIME_BINS[-1][2]) != int(config.COHORT_HORIZON_MINS):
        raise ValueError("The final horizon bin must end at COHORT_HORIZON_MINS")


def _relabel_bin(master: pd.DataFrame, low: int, high: int) -> pd.DataFrame:
    output = master.copy()
    t_event = pd.to_numeric(output["t_event"], errors="coerce")
    t_end = pd.to_numeric(output["t_end"], errors="coerce")
    lead = t_event - t_end
    output["lead_time_mins"] = lead
    output["label"] = (t_event.notna() & (lead > low) & (lead <= high)).astype("int64")
    ratio = float(config.PREDICTION_HORIZON_NEG_POS_MAX_RATIO)
    random_state = int(config.NEG_LIMITER_RANDOM_STATE)
    blocks: list[pd.DataFrame] = []
    for split in ("train", "val", "test"):
        part = output.loc[output["split"] == split].copy()
        if part.empty:
            continue
        positive = part.loc[part["label"] == 1]
        negative = part.loc[part["label"] == 0]
        if positive.empty:
            blocks.append(part)
            continue
        max_negative = int(np.floor(ratio * len(positive)))
        if len(negative) > max_negative:
            negative = negative.sample(n=max_negative, random_state=random_state)
            part = (
                pd.concat([positive, negative], ignore_index=True)
                .sample(frac=1.0, random_state=random_state)
                .reset_index(drop=True)
            )
        blocks.append(part)
    return pd.concat(blocks, ignore_index=True)


def _prepare_samples(root: Path) -> tuple[dict[str, Path], Path, dict[str, object]]:
    master_path = config.fixed_cohort_samples_path(config.DISEASE)
    if not master_path.exists():
        raise FileNotFoundError(
            f"Missing generated fixed cohort: {master_path}\n"
            "Run 'python3.11 prepare_data.py' once with FIXED_COHORT_ENABLED=True."
        )
    master = pd.read_csv(master_path)
    _validate_master(master)

    samples_dir = root / "Samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    frames: list[pd.DataFrame] = []
    positive_keys: set[tuple[int, int]] = set()
    bin_counts: dict[str, object] = {}

    for name, low, high in TIME_BINS:
        samples = _relabel_bin(master, low, high)
        positives = samples.loc[samples["label"] == 1, KEY_COLUMNS]
        keys = set(map(tuple, positives.to_numpy()))
        if positive_keys.intersection(keys):
            raise RuntimeError(f"Positive windows overlap across horizon bins: {name}")
        positive_keys.update(keys)

        for split, part in samples.groupby("split"):
            n_pos = int(part["label"].sum())
            n_neg = int(len(part) - n_pos)
            ratio = float(config.PREDICTION_HORIZON_NEG_POS_MAX_RATIO)
            if n_pos and n_neg > int(np.floor(ratio * n_pos)):
                raise RuntimeError(f"Negative cap failed for bin={name}, split={split}")

        path = samples_dir / f"samples__fixed12hcohort__bin_{name}.csv"
        samples.to_csv(path, index=False)
        paths[name] = path
        frames.append(samples)
        bin_counts[name] = {
            "low_mins_exclusive": low,
            "high_mins_inclusive": high,
            "windows": int(len(samples)),
            "positives": int(samples["label"].sum()),
            "negatives": int((samples["label"] == 0).sum()),
            "sha256": _sha256(path),
        }

    combined = pd.concat(frames, ignore_index=True)
    split_conflicts = combined.groupby(KEY_COLUMNS)["split"].nunique()
    if (split_conflicts > 1).any():
        raise RuntimeError("A horizon feature key has inconsistent split assignments")
    cohort = combined[KEY_COLUMNS + ["split"]].drop_duplicates(KEY_COLUMNS).copy()
    cohort.insert(2, "label", 0)  # Required by aggregation; labels are unused there.
    cohort_path = root / "samples__feature_cohort.csv"
    cohort.to_csv(cohort_path, index=False)

    manifest = {
        "source": str(master_path),
        "source_sha256": _sha256(master_path),
        "lead_time_mins": int(config.LEAD_TIME_MINS),
        "negative_to_positive_cap": float(config.PREDICTION_HORIZON_NEG_POS_MAX_RATIO),
        "feature_cohort_windows": int(len(cohort)),
        "feature_cohort_sha256": _sha256(cohort_path),
        "bins": bin_counts,
    }
    (root / "cohort_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return paths, cohort_path, manifest


def _expected_feature_paths(features_dir: Path) -> list[Path]:
    tag = config.disease_tag(config.DISEASE)
    return [
        features_dir / f"features__{tag}.parquet",
        features_dir / f"features_vitals__{tag}.parquet",
        features_dir / f"features_baseline__{tag}.parquet",
    ]


def _validate_feature_keys(cohort_path: Path, feature_paths: list[Path]) -> None:
    expected = pd.read_csv(cohort_path, usecols=KEY_COLUMNS).drop_duplicates()
    expected_index = pd.MultiIndex.from_frame(expected)
    for path in feature_paths:
        actual = pd.read_parquet(path, columns=KEY_COLUMNS)
        if actual.duplicated(KEY_COLUMNS).any():
            raise RuntimeError(f"Feature file contains duplicate keys: {path}")
        actual_index = pd.MultiIndex.from_frame(actual)
        missing = expected_index.difference(actual_index)
        extra = actual_index.difference(expected_index)
        if len(missing) or len(extra):
            raise RuntimeError(
                f"Feature-key mismatch for {path}: missing={len(missing)}, extra={len(extra)}"
            )


def _ensure_features(root: Path, cohort_path: Path, force: bool) -> tuple[Path, str]:
    code_dir = Path(__file__).resolve().parent
    features_dir = root / "Features"
    feature_paths = _expected_feature_paths(features_dir)
    fingerprint = _fingerprint(
        [
            cohort_path,
            code_dir / "aggregate_features.py",
            code_dir / "missing_value_imputation.py",
            code_dir / "config.py",
        ],
        {"feature_mode": config.FEATURE_MODE},
    )
    manifest_path = features_dir / "feature_manifest.json"
    saved_manifest = {}
    if manifest_path.exists():
        try:
            saved_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            saved_manifest = {}
    rebuild = (
        force
        or any(not path.exists() for path in feature_paths)
        or saved_manifest.get("fingerprint") != fingerprint
    )
    if not rebuild:
        try:
            _validate_feature_keys(cohort_path, feature_paths)
        except (OSError, ValueError, RuntimeError):
            print("Stored horizon features failed validation; rebuilding them.")
            rebuild = True
    if rebuild:
        features_dir.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            str(code_dir / "aggregate_features.py"),
            "--samples-path",
            str(cohort_path),
            "--output-dir",
            str(features_dir),
        ]
        print("Building prediction-horizon features from the generated cohort...")
        subprocess.run(command, check=True)
        _validate_feature_keys(cohort_path, feature_paths)
        manifest_path.write_text(
            json.dumps(
                {
                    "fingerprint": fingerprint,
                    "cohort": str(cohort_path),
                    "feature_files": [path.name for path in feature_paths],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    else:
        print(f"Using verified prediction-horizon features: {features_dir}")
    return features_dir, fingerprint


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


def _train_trgru(
    samples: Path,
    features: Path,
    output: Path,
    bin_name: str,
    low: int,
    high: int,
    feature_fingerprint: str,
) -> dict[str, object]:
    cfg = TrGRUConfig()
    _seed(cfg.seed)
    datasets = {
        split: PatientSequenceDataset(
            split,
            config.DISEASE,
            cfg.max_len,
            cfg.seed,
            True,
            samples_path=samples,
            features_dir=features,
        )
        for split in ("train", "val", "test")
    }
    loaders = {
        split: DataLoader(ds, cfg.batch_size, split == "train", collate_fn=pad_collate)
        for split, ds in datasets.items()
    }
    train_eval = DataLoader(datasets["train"], cfg.batch_size, False, collate_fn=pad_collate)
    model = _build_model(len(datasets["train"].feature_cols), cfg).to(cfg.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    pos_weight = _resolve_pos_weight(loaders["train"], cfg.device)
    best_val, best_state, bad_epochs = -np.inf, None, 0

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        for x, y, mask, lengths, _ in loaders["train"]:
            x, y, mask, lengths = (
                x.to(cfg.device),
                y.to(cfg.device),
                mask.to(cfg.device),
                lengths.to(cfg.device),
            )
            optimizer.zero_grad()
            loss = masked_focal_loss(
                model(x, lengths)["logits_ts"], y, mask, gamma=2.0, pos_weight=pos_weight
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        val = _metrics(model, loaders["val"], cfg.device)
        print(f"Epoch {epoch:02d}: val AUROC={val['auroc']:.4f}, AUPRC={val['auprc']:.4f}")
        if np.isfinite(val["auroc"]) and val["auroc"] > best_val:
            best_val = float(val["auroc"])
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.patience:
                break
    if best_state is None:
        raise RuntimeError(f"No finite validation AUROC was produced for bin {bin_name}")
    model.load_state_dict(best_state)

    train = _metrics(model, train_eval, cfg.device)
    val = _metrics(model, loaders["val"], cfg.device)
    test = _metrics(model, loaders["test"], cfg.device)
    val_y, val_prob = _flatten_loader_probs(model, loaders["val"], cfg.device)
    threshold = _pick_threshold_max_f1(val_y, val_prob)
    test_y, test_prob = _flatten_loader_probs(model, loaders["test"], cfg.device)
    test.update(_threshold_metrics(test_y, test_prob, threshold))

    sample_df = pd.read_csv(samples, usecols=["label"])
    result: dict[str, object] = {
        "time_bin": bin_name,
        "bin_low_mins_exclusive": low,
        "bin_high_mins_inclusive": high,
        "n_windows": int(len(sample_df)),
        "n_pos_windows": int(sample_df["label"].sum()),
        "n_neg_windows": int((sample_df["label"] == 0).sum()),
        "n_features": len(datasets["train"].feature_cols),
        **{f"train_{key}": value for key, value in train.items()},
        **{f"val_{key}": value for key, value in val.items()},
        **{f"test_{key}": value for key, value in test.items()},
    }
    output.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": asdict(cfg),
            "feature_names": datasets["train"].feature_cols,
            "threshold": threshold,
            "model_type": "centralized_trgru_prediction_horizon",
            "time_bin": bin_name,
        },
        output / "model.pt",
    )
    pd.DataFrame([result]).to_csv(output / "results.csv", index=False)
    metadata = {
        "config": asdict(cfg),
        "result": result,
        "samples": str(samples),
        "samples_sha256": _sha256(samples),
        "feature_fingerprint": feature_fingerprint,
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return result


def _training_fingerprint(feature_fingerprint: str) -> str:
    code_dir = Path(__file__).resolve().parent
    return _fingerprint(
        [
            code_dir / "run_prediction_horizons.py",
            code_dir / "run_centralized_trgru.py",
            code_dir / "trgru_model.py",
            code_dir / "sequence_dataset_gru.py",
            code_dir / "train_eval_gru.py",
        ],
        {"feature_fingerprint": feature_fingerprint, "model": asdict(TrGRUConfig())},
    )


def _load_completed(
    output: Path, samples: Path, feature_fingerprint: str, training_fingerprint: str
) -> dict[str, object] | None:
    metadata_path = output / "metadata.json"
    results_path = output / "results.csv"
    model_path = output / "model.pt"
    if not all(path.exists() for path in (metadata_path, results_path, model_path)):
        return None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        result = pd.read_csv(results_path)
    except (json.JSONDecodeError, OSError, ValueError, RuntimeError):
        return None
    if (
        metadata.get("samples_sha256") != _sha256(samples)
        or metadata.get("feature_fingerprint") != feature_fingerprint
        or metadata.get("training_fingerprint") != training_fingerprint
        or checkpoint.get("time_bin") != output.name
    ):
        return None
    required = {"time_bin", "test_auroc", "test_auprc", "n_features"}
    if len(result) != 1 or not required.issubset(result.columns):
        return None
    if not np.isfinite(result.loc[0, ["test_auroc", "test_auprc"]].astype(float)).all():
        return None
    return result.iloc[0].to_dict()


def main() -> None:
    args = _parse_args()
    _validate_protocol()
    root = config.prediction_horizon_dir(config.DISEASE)
    root.mkdir(parents=True, exist_ok=True)
    sample_paths, cohort_path, cohort_manifest = _prepare_samples(root)
    features_dir, feature_fingerprint = _ensure_features(root, cohort_path, args.force_features)
    training_fingerprint = _training_fingerprint(feature_fingerprint)

    results_path = root / "trgru_prediction_horizons.csv"
    rows: list[dict[str, object]] = []
    for name, low, high in TIME_BINS:
        output = root / "TrGRU" / name
        completed = None if args.force_training else _load_completed(
            output, sample_paths[name], feature_fingerprint, training_fingerprint
        )
        print(f"\n{'=' * 70}\nTrGRU prediction-horizon bin: {name}\n{'=' * 70}")
        if completed is not None:
            print("Using verified completed result.")
            row = completed
        else:
            row = _train_trgru(
                sample_paths[name],
                features_dir,
                output,
                name,
                low,
                high,
                feature_fingerprint,
            )
            metadata_path = output / "metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["training_fingerprint"] = training_fingerprint
            metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        rows.append(row)
        pd.DataFrame(rows).to_csv(results_path, index=False)

    final_manifest = {
        "cohort": cohort_manifest,
        "feature_fingerprint": feature_fingerprint,
        "training_fingerprint": training_fingerprint,
        "results": str(results_path),
    }
    (root / "run_manifest.json").write_text(
        json.dumps(final_manifest, indent=2), encoding="utf-8"
    )
    results = pd.DataFrame(rows)
    print(results.to_string(index=False))
    print(f"Saved prediction-horizon comparison: {results_path}")


if __name__ == "__main__":
    main()
