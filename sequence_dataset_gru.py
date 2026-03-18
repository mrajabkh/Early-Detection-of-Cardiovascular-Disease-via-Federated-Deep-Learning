# sequence_dataset_gru.py
# Build patient-level sequences from rolling-window feature parquet(s) + samples.csv
# Outputs (X, y, length) per patient for GRU/LSTM models.
#
# Uses the 'split' column in samples.csv to match your ML pipeline.
# Enforces patient-level splitting: a patient must not appear in multiple splits.
#
# IMPORTANT:
# - If samples.csv includes 't_event' and 'lead_time_mins', we keep them in df for evaluation,
#   but we EXCLUDE them from model features so they are not normalized or used as inputs.
#
# FEATURE MODE SUPPORT:
# - config.FEATURE_MODE == "all"
#     -> load full feature parquet
# - config.FEATURE_MODE == "vitals"
#     -> load vitals-only parquet
# - config.FEATURE_MODE == "vitals+demo"
#     -> load vitals parquet + full baseline parquet
#
# MEMORY FIX:
# - If top_k is provided, we only load ["patientunitstayid", "t_end"] + top_k ranked feature columns
#   from the full features parquet. This prevents loading a huge wide dataframe into RAM.
# - top_k / rank_path are only valid when FEATURE_MODE == "all"
#
# ROBUSTNESS FIX:
# - If rank CSV contains feature names not present in features.parquet (e.g. *_missing),
#   we filter them out using the parquet schema (without reading the full dataset).
#
# NEW (federation support):
# - PatientSequenceDataset accepts samples_path override so each node can point to its own samples CSV.

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple, Set, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

import config


def _read_samples(samples_path: Union[str, Path]) -> pd.DataFrame:
    df = pd.read_csv(str(samples_path))

    need = {"patientunitstayid", "t_end", "label", "split"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"samples.csv missing columns: {sorted(missing)}")

    optional_cols = []
    for c in ["t_event", "lead_time_mins"]:
        if c in df.columns:
            optional_cols.append(c)

    keep_cols = ["patientunitstayid", "t_end", "label", "split"] + optional_cols
    df = df[keep_cols].copy()

    df = df.replace([np.inf, -np.inf], np.nan)

    df["patientunitstayid"] = pd.to_numeric(df["patientunitstayid"], errors="coerce")
    df["t_end"] = pd.to_numeric(df["t_end"], errors="coerce")
    df["label"] = pd.to_numeric(df["label"], errors="coerce")

    if "t_event" in df.columns:
        df["t_event"] = pd.to_numeric(df["t_event"], errors="coerce")
    if "lead_time_mins" in df.columns:
        df["lead_time_mins"] = pd.to_numeric(df["lead_time_mins"], errors="coerce")

    df = df.dropna(subset=["patientunitstayid", "t_end", "label", "split"])

    df["patientunitstayid"] = df["patientunitstayid"].astype(np.int64)
    df["t_end"] = df["t_end"].astype(np.int64)
    df["label"] = df["label"].astype(np.int64)

    df["split"] = df["split"].astype(str).str.lower()
    bad = ~df["split"].isin(["train", "val", "test"])
    if bad.any():
        bad_vals = sorted(df.loc[bad, "split"].unique().tolist())
        raise ValueError(f"samples.csv has invalid split values: {bad_vals}")

    return df


def _parquet_schema_columns(features_path: str) -> Set[str]:
    try:
        import pyarrow.parquet as pq
    except Exception as e:
        raise ImportError(
            "pyarrow is required to read parquet schema for top_k column filtering. "
            f"Original error: {repr(e)}"
        )

    pf = pq.ParquetFile(features_path)
    return set(pf.schema.names)


def _load_ranked_features(rank_path: str, top_k: int, features_path: str) -> List[str]:
    r = pd.read_csv(rank_path)
    if "feature" not in r.columns:
        raise ValueError("rank_path must be a CSV with a 'feature' column")

    want = [x.strip() for x in r["feature"].astype(str).tolist()]
    want = want[: int(top_k)]

    existing = _parquet_schema_columns(features_path)
    kept = [c for c in want if c in existing]

    if len(kept) == 0:
        example = want[0] if want else "(none)"
        raise ValueError(
            "None of the requested top_k ranked features were found in features.parquet. "
            "This usually means your rank CSV was built on a different feature set "
            "(e.g. it includes *_missing columns) than the parquet you are loading. "
            f"Example requested feature: {example}"
        )

    if len(kept) < len(want):
        skipped = len(want) - len(kept)
        print("#############################")
        print("WARNING: rank_path contains features not present in features.parquet")
        print(f"Requested top_k: {int(top_k)}")
        print(f"Kept (found in parquet): {len(kept)}")
        print(f"Skipped (missing from parquet): {skipped}")
        print("Example missing features:")
        miss_examples = [c for c in want if c not in existing][:10]
        for m in miss_examples:
            print(f" - {m}")
        print("#############################")

    return kept


def _read_features(features_path: str, columns: Optional[List[str]] = None) -> pd.DataFrame:
    df = pd.read_parquet(features_path, columns=columns)

    need = {"patientunitstayid", "t_end"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"features parquet missing columns: {sorted(missing)}")

    df = df.replace([np.inf, -np.inf], np.nan)

    df["patientunitstayid"] = pd.to_numeric(df["patientunitstayid"], errors="coerce")
    df["t_end"] = pd.to_numeric(df["t_end"], errors="coerce")
    df = df.dropna(subset=["patientunitstayid", "t_end"])

    df["patientunitstayid"] = df["patientunitstayid"].astype(np.int64)
    df["t_end"] = df["t_end"].astype(np.int64)

    return df


def _select_feature_columns(df: pd.DataFrame) -> List[str]:
    drop = {
        "patientunitstayid",
        "t_end",
        "label",
        "split",
        "t_event",
        "lead_time_mins",
        "t_event_missing",
        "lead_time_mins_missing",
    }

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cols = [c for c in numeric_cols if c not in drop]
    if not cols:
        raise ValueError("No numeric feature columns found after filtering.")
    return cols


def _compute_norm_stats(df: pd.DataFrame, feature_cols: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    x = df[feature_cols].to_numpy(dtype=np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def _apply_norm(df: pd.DataFrame, feature_cols: List[str], mean: np.ndarray, std: np.ndarray) -> None:
    x = df[feature_cols].to_numpy(dtype=np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x = (x - mean[None, :]) / std[None, :]
    df.loc[:, feature_cols] = x


def _restrict_to_topk_features(
    df: pd.DataFrame,
    feature_cols: List[str],
    top_k: int,
    rank_path: Optional[str],
) -> List[str]:
    top_k = int(top_k)
    if top_k <= 0:
        raise ValueError("top_k must be a positive integer")

    ranked: Optional[List[str]] = None

    if rank_path is not None:
        r = pd.read_csv(rank_path)
        if "feature" not in r.columns:
            raise ValueError("rank_path must be a CSV with a 'feature' column")
        ranked = [f for f in r["feature"].astype(str).tolist() if f in feature_cols]

    if ranked is None or len(ranked) == 0:
        tmp = df[feature_cols].to_numpy(dtype=np.float32)
        tmp = np.nan_to_num(tmp, nan=0.0, posinf=0.0, neginf=0.0)
        var = tmp.var(axis=0)
        order = np.argsort(-var)
        ranked = [feature_cols[i] for i in order.tolist()]

    return ranked[:top_k]


def _assert_patient_split_consistent(samples: pd.DataFrame) -> None:
    grp = samples.groupby("patientunitstayid")["split"].nunique()
    bad = grp[grp > 1]
    if len(bad) > 0:
        example_pids = bad.index[:10].tolist()
        raise ValueError(
            "Patient leakage detected: some patientunitstayid appear in multiple splits. "
            f"Examples: {example_pids}. Fix samples.csv split assignment to be patient-level."
        )


def _feature_mode() -> str:
    mode = str(getattr(config, "FEATURE_MODE", "all")).strip().lower()
    valid = {"all", "vitals", "vitals+demo"}
    if mode not in valid:
        raise ValueError(f"Unsupported config.FEATURE_MODE={mode!r}. Expected one of {sorted(valid)}")
    return mode


def _load_features_for_mode(
    disease: config.DiseaseSpec,
    feature_mode: str,
    top_k: Optional[int],
    rank_path: Optional[str],
) -> pd.DataFrame:
    if feature_mode == "all":
        features_path = str(config.features_path(disease))

        feat_columns = None
        if top_k is not None:
            if rank_path is None:
                raise ValueError("top_k was set but rank_path is None")
            ranked = _load_ranked_features(rank_path, top_k=int(top_k), features_path=features_path)
            feat_columns = ["patientunitstayid", "t_end"] + ranked

        return _read_features(features_path, columns=feat_columns)

    if top_k is not None or rank_path is not None:
        raise ValueError(
            f"top_k/rank_path are only supported when FEATURE_MODE == 'all'. "
            f"Current FEATURE_MODE is {feature_mode!r}."
        )

    if feature_mode == "vitals":
        vitals_path = str(config.vitals_features_path(disease))
        return _read_features(vitals_path)

    if feature_mode == "vitals+demo":
        vitals_path = str(config.vitals_features_path(disease))
        baseline_path = str(config.baseline_features_path(disease))

        vitals = _read_features(vitals_path)
        baseline = _read_features(baseline_path)

        merged = vitals.merge(
            baseline,
            on=["patientunitstayid", "t_end"],
            how="inner",
            suffixes=("", "__dup"),
        )

        dup_cols = [c for c in merged.columns if c.endswith("__dup")]
        if dup_cols:
            merged = merged.drop(columns=dup_cols)

        return merged

    raise ValueError(f"Unsupported FEATURE_MODE: {feature_mode}")


class PatientSequenceDataset(Dataset):
    """
    Returns per-patient sequences:
      X: (T, D) float32
      y: (T,) int64
      length: int64 (<= max_len)
    """

    def __init__(
        self,
        split: str,
        disease: Optional[config.DiseaseSpec] = None,
        max_len: int = 128,
        seed: int = 42,
        normalize: bool = True,
        cached_norm_path: Optional[str] = None,
        top_k: Optional[int] = None,
        rank_path: Optional[str] = None,
        samples_path: Optional[Union[str, Path]] = None,
    ) -> None:
        super().__init__()
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be one of: train, val, test")

        if disease is None:
            disease = getattr(config, "DISEASE", None)
        if disease is None:
            raise ValueError("disease is None and config.DISEASE not found.")

        self.split = split
        self.max_len = int(max_len)
        self.seed = int(seed)

        if samples_path is None:
            samples_path_str = str(config.samples_path(disease))
        else:
            samples_path_str = str(Path(samples_path))

        feature_mode = _feature_mode()
        print("#############################")
        print(f"PatientSequenceDataset split={split}")
        print(f"FEATURE_MODE={feature_mode}")
        print("#############################")

        samples = _read_samples(samples_path_str)
        feats = _load_features_for_mode(
            disease=disease,
            feature_mode=feature_mode,
            top_k=top_k,
            rank_path=rank_path,
        )

        _assert_patient_split_consistent(samples)

        df = samples.merge(feats, on=["patientunitstayid", "t_end"], how="inner")
        if df.empty:
            raise ValueError("Merged dataset is empty. Check that samples.csv and feature parquet(s) align.")

        df = df.sort_values(["patientunitstayid", "t_end"]).reset_index(drop=True)

        feature_cols = _select_feature_columns(df)

        df = df[df["split"] == split].copy()
        df = df.sort_values(["patientunitstayid", "t_end"]).reset_index(drop=True)
        if df.empty:
            raise ValueError(f"No rows left after applying split='{split}'. Check samples.csv split column.")

        if top_k is not None:
            if feature_mode != "all":
                raise ValueError("top_k is only valid when FEATURE_MODE == 'all'")
            feature_cols = _restrict_to_topk_features(df, feature_cols, top_k=top_k, rank_path=rank_path)

        self.feature_cols = feature_cols
        self.feature_mode = feature_mode

        self.mean = None
        self.std = None
        if normalize:
            if cached_norm_path is not None:
                norm = np.load(cached_norm_path)
                self.mean = norm["mean"].astype(np.float32)
                self.std = norm["std"].astype(np.float32)
            else:
                df_train = samples.merge(feats, on=["patientunitstayid", "t_end"], how="inner")
                df_train = df_train.sort_values(["patientunitstayid", "t_end"]).reset_index(drop=True)
                df_train = df_train[df_train["split"] == "train"].copy()
                df_train = df_train.sort_values(["patientunitstayid", "t_end"]).reset_index(drop=True)

                mean, std = _compute_norm_stats(df_train, self.feature_cols)
                self.mean, self.std = mean, std

            _apply_norm(df, self.feature_cols, self.mean, self.std)

        self.df = df
        self.pids = df["patientunitstayid"].unique().astype(np.int64)

        self._groups: List[np.ndarray] = []
        pid_values = df["patientunitstayid"].to_numpy(dtype=np.int64)

        start = 0
        for pid in self.pids:
            while start < len(pid_values) and pid_values[start] != pid:
                start += 1
            end = start
            while end < len(pid_values) and pid_values[end] == pid:
                end += 1
            idx = np.arange(start, end, dtype=np.int64)
            self._groups.append(idx)
            start = end

    def __len__(self) -> int:
        return len(self.pids)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        idx = self._groups[i]
        sub = self.df.iloc[idx]

        x = sub[self.feature_cols].to_numpy(dtype=np.float32)
        y = sub["label"].to_numpy(dtype=np.int64)

        if len(y) > self.max_len:
            x = x[-self.max_len :]
            y = y[-self.max_len :]

        length = np.int64(len(y))

        return (
            torch.from_numpy(x),
            torch.from_numpy(y),
            torch.tensor(length, dtype=torch.long),
        )


def pad_collate(batch):
    xs, ys, lens = zip(*batch)
    lengths = torch.stack(lens, dim=0)

    bsz = len(xs)
    d = xs[0].shape[1]
    t_max = int(lengths.max().item())

    x_pad = torch.zeros((bsz, t_max, d), dtype=torch.float32)
    y_pad = torch.zeros((bsz, t_max), dtype=torch.long)
    mask = torch.zeros((bsz, t_max), dtype=torch.float32)

    for i in range(bsz):
        t = int(lengths[i].item())
        x_pad[i, :t, :] = xs[i]
        y_pad[i, :t] = ys[i]
        mask[i, :t] = 1.0

    return x_pad, y_pad, mask, lengths