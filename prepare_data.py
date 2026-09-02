# prepare_data.py
# Build samples.csv for a chosen disease definition with horizon labeling.
# Adds patient-level train/val/test split to avoid patient overlap across sets.
# Stores t_event (onset time) and time-to-event (lead_time_mins = onset_time - t_end).
#
# Updated for the new Cardiac Arrest baseline:
# - Horizon labeling with lead time:
#     positive if t_end + LEAD_TIME_MINS < onset_time <= t_end + HORIZON_MINS
# - Stride is hourly predictions (config.STRIDE_MINS)
# - Coverage is vitals-only (vitalPeriodic + vitalAperiodic)
# - Do NOT drop positive patients if vitals are sparse
# - Optional Neg limiter per split (config-controlled):
#     keep all positives, downsample negatives to Neg <= R * Pos
# - Node assignment:
#     pick top-K hospitals by CA-positive patient count as anchors (K from config)
#     print CA-positive patient counts for every hospital (desc),
#     and also show the (K+1)th hospital as "just missed" when available.
#
# Location: Project/Code/prepare_data.py

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import config


#############################
# Helpers
#############################
def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def print_counts(df: pd.DataFrame, title: str) -> None:
    n_total = int(len(df))
    n_pos = int(df["label"].sum()) if "label" in df.columns else 0
    n_neg = int((df["label"] == 0).sum()) if "label" in df.columns else 0
    ratio = (n_neg / n_pos) if n_pos > 0 else float("inf")
    prev = (n_pos / n_total) if n_total > 0 else 0.0

    print("#############################")
    print(title)
    print(f"Total windows: {n_total}")
    print(f"Pos windows:   {n_pos}")
    print(f"Neg windows:   {n_neg}")
    print(f"Neg:Pos ratio: {ratio:.2f}:1" if np.isfinite(ratio) else "Neg:Pos ratio: inf")
    print(f"Prevalence:    {prev:.4f}")
    print("#############################")


def read_diagnoses(data_dir: Path) -> pd.DataFrame:
    path = data_dir / "diagnosis.csv.gz"
    df = pd.read_csv(path, compression="infer", low_memory=False)
    required = {"diagnosisstring", "patientunitstayid", "diagnosisoffset"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"diagnosis.csv.gz missing columns: {sorted(missing)}")
    return df


def read_patients(data_dir: Path) -> pd.DataFrame:
    path = data_dir / "patient.csv.gz"
    df = pd.read_csv(path, compression="infer", low_memory=False, usecols=["patientunitstayid", "hospitalid"])
    required = {"patientunitstayid", "hospitalid"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"patient.csv.gz missing columns: {sorted(missing)}")
    return df


def split_diagnosisstring(diagnoses: pd.DataFrame) -> pd.DataFrame:
    parts = diagnoses["diagnosisstring"].astype(str).str.split("|", n=2, expand=True)
    diagnoses = diagnoses.copy()
    diagnoses["majorcategory"] = parts[0].str.strip()
    diagnoses["subcategory"] = parts[1].str.strip() if parts.shape[1] > 1 else ""
    diagnoses["diagnosisname"] = parts[2].str.strip() if parts.shape[1] > 2 else ""
    return diagnoses


def filter_disease_rows(diagnoses: pd.DataFrame, disease: config.DiseaseSpec) -> pd.DataFrame:
    df = split_diagnosisstring(diagnoses)
    major = disease.major.strip().lower()
    subcat = disease.subcategory.strip().lower() if disease.subcategory else None

    if subcat is None:
        out = df[df["majorcategory"].str.lower() == major]
    else:
        out = df[(df["majorcategory"].str.lower() == major) & (df["subcategory"].str.lower() == subcat)]
    return out


def compute_onset_times(disease_rows: pd.DataFrame) -> pd.Series:
    onset = disease_rows.groupby("patientunitstayid")["diagnosisoffset"].min()
    onset = pd.to_numeric(onset, errors="coerce").dropna()
    onset.index = onset.index.astype(int)
    onset = onset.astype(int)
    return onset


#############################
# Coverage (vitals only)
#############################
def compute_max_offsets_vitals_only(data_dir: Path, chunksize: int = 1_000_000) -> pd.Series:
    sources = [
        ("vitalPeriodic.csv.gz", "observationoffset", ["patientunitstayid", "observationoffset"]),
        ("vitalAperiodic.csv.gz", "observationoffset", ["patientunitstayid", "observationoffset"]),
    ]

    max_dict: Dict[int, int] = defaultdict(int)

    for fname, offset_col, usecols in sources:
        path = data_dir / fname
        if not path.exists():
            print(f"WARNING: missing vitals file for coverage, skipping: {fname}")
            continue

        print("#############################")
        print(f"Vitals coverage scan: {fname} ({offset_col})")
        print("#############################")

        reader = pd.read_csv(
            path,
            compression="infer",
            usecols=usecols,
            chunksize=chunksize,
            low_memory=False,
        )

        for chunk in reader:
            if "patientunitstayid" not in chunk.columns or offset_col not in chunk.columns:
                continue

            chunk["patientunitstayid"] = pd.to_numeric(chunk["patientunitstayid"], errors="coerce")
            chunk[offset_col] = pd.to_numeric(chunk[offset_col], errors="coerce")
            chunk = chunk.dropna(subset=["patientunitstayid", offset_col])
            if chunk.empty:
                continue

            chunk["patientunitstayid"] = chunk["patientunitstayid"].astype(int)
            grp = chunk.groupby("patientunitstayid")[offset_col].max()

            for pid, max_off in grp.items():
                if pd.isna(max_off):
                    continue
                max_off_int = int(max_off)
                if max_off_int > max_dict[pid]:
                    max_dict[pid] = max_off_int

    return pd.Series(max_dict, name="max_offset_vitals")


#############################
# Window generation + labeling
#############################
def generate_patient_windows(
    patient_id: int,
    max_offset_vitals: int,
    onset_time: Optional[int],
    min_history_mins: int,
    horizon_mins: int,
    stride_mins: int,
    lead_time_mins: int,
    require_full_horizon_for_negatives: bool,
    stop_after_event: bool,
) -> List[Tuple[int, int, int, float, float]]:
    """
    Creates hourly windows ending at t_end.

    Labeling (horizon + lead time):
      label=1 iff t_end + lead_time_mins < onset_time <= t_end + horizon_mins

    Coverage:
      - We determine the latest feasible t_end primarily from vitals coverage.
      - For NEGATIVE patients (onset_time is None), we optionally require full horizon coverage.
      - For POSITIVE patients, we DO NOT require full horizon coverage (do not drop positives if vitals are sparse).
    """
    if max_offset_vitals < min_history_mins:
        return []

    last_t_end = int(max_offset_vitals)

    if onset_time is None and require_full_horizon_for_negatives:
        last_t_end = last_t_end - int(horizon_mins)

    if last_t_end < min_history_mins:
        return []

    t_ends = list(range(int(min_history_mins), int(last_t_end) + 1, int(stride_mins)))
    t_event = float(onset_time) if onset_time is not None else float("nan")

    rows: List[Tuple[int, int, int, float, float]] = []
    for t_end in t_ends:
        if onset_time is not None and stop_after_event and t_end >= onset_time:
            break

        label = 0
        if onset_time is not None:
            if (t_end + lead_time_mins < onset_time) and (onset_time <= t_end + horizon_mins):
                label = 1

        lead_time = float(onset_time - t_end) if onset_time is not None else float("nan")
        rows.append((patient_id, int(t_end), int(label), t_event, lead_time))

    return rows


#############################
# Patient-level split
#############################
def add_patient_level_splits(samples_df: pd.DataFrame) -> pd.DataFrame:
    per_patient = (
        samples_df.groupby("patientunitstayid")["label"]
        .max()
        .astype(int)
        .rename("patient_has_positive")
        .reset_index()
    )

    pids = per_patient["patientunitstayid"].to_numpy()
    strat = per_patient["patient_has_positive"].to_numpy()

    test_size = float(getattr(config, "TEST_SIZE", 0.2))
    val_size = float(getattr(config, "VAL_SIZE", 0.15))
    rs = int(getattr(config, "SPLIT_RANDOM_STATE", 42))

    trainval_pids, test_pids = train_test_split(
        pids,
        test_size=test_size,
        random_state=rs,
        stratify=strat,
    )

    trainval_strat = per_patient.set_index("patientunitstayid").loc[trainval_pids, "patient_has_positive"].to_numpy()

    trainval_frac = 1.0 - test_size
    val_frac_of_trainval = val_size / trainval_frac
    val_frac_of_trainval = float(np.clip(val_frac_of_trainval, 0.01, 0.8))

    train_pids, val_pids = train_test_split(
        trainval_pids,
        test_size=val_frac_of_trainval,
        random_state=rs,
        stratify=trainval_strat,
    )

    train_set = set(int(x) for x in train_pids.tolist())
    val_set = set(int(x) for x in val_pids.tolist())

    out = samples_df.copy()
    out["split"] = "test"
    out.loc[out["patientunitstayid"].isin(train_set), "split"] = "train"
    out.loc[out["patientunitstayid"].isin(val_set), "split"] = "val"
    return out


#############################
# Neg limiter (config-controlled, per split)
#############################
def apply_neg_limiter_from_config(
    samples_df: pd.DataFrame,
    max_ratio_override: Optional[float] = None,
    stage_name: str = "final",
) -> pd.DataFrame:
    enabled = bool(getattr(config, "NEG_LIMITER_ENABLED", False))
    if not enabled:
        return samples_df

    if "split" not in samples_df.columns:
        print("#############################")
        print("WARNING: NEG_LIMITER_ENABLED=True but no split column found. Skipping neg limiter.")
        print("#############################")
        return samples_df

    max_ratio = (
        float(max_ratio_override)
        if max_ratio_override is not None
        else float(getattr(config, "NEG_POS_MAX_RATIO", 10.0))
    )
    rs = int(getattr(config, "NEG_LIMITER_RANDOM_STATE", getattr(config, "SEED", 42)))

    print("#############################")
    print(f"Applying neg limiter per split (stage={stage_name})")
    print("Rule: keep all positives, downsample negatives to Neg <= max_ratio * Pos")
    print(f"NEG_POS_MAX_RATIO: {max_ratio}")
    print(f"NEG_LIMITER_RANDOM_STATE: {rs}")
    print("#############################")

    out_blocks: List[pd.DataFrame] = []
    df = samples_df.copy()

    for sp in ["train", "val", "test"]:
        part = df[df["split"] == sp].copy()
        if part.empty:
            continue

        n_pos = int(part["label"].sum())
        if n_pos == 0:
            out_blocks.append(part)
            continue

        pos_df = part[part["label"] == 1].copy()
        neg_df = part[part["label"] == 0].copy()

        max_neg = int(np.floor(max_ratio * len(pos_df)))
        if max_neg < 0:
            max_neg = 0

        if len(neg_df) <= max_neg:
            out_blocks.append(part)
            continue

        neg_keep = neg_df.sample(n=max_neg, random_state=rs)
        new_part = (
            pd.concat([pos_df, neg_keep], axis=0)
            .sample(frac=1.0, random_state=rs)
            .reset_index(drop=True)
        )
        out_blocks.append(new_part)

        print("#############################")
        print(f"Neg limiter applied to split={sp}")
        print(f"Pos: {len(pos_df)} | Neg before: {len(neg_df)} | Neg after: {len(neg_keep)}")
        print("#############################")

    other = df[~df["split"].isin(["train", "val", "test"])].copy()
    if not other.empty:
        out_blocks.append(other)

    return pd.concat(out_blocks, axis=0).reset_index(drop=True)


def relabel_fixed_cohort(
    samples_df: pd.DataFrame,
    horizon_mins: int,
    lead_time_mins: int,
) -> pd.DataFrame:
    """Relabel retained cohort windows without changing their keys or splits."""
    out = samples_df.copy()
    t_event = pd.to_numeric(out["t_event"], errors="coerce")
    t_end = pd.to_numeric(out["t_end"], errors="coerce")
    lead = t_event - t_end
    out["label"] = (
        t_event.notna()
        & (lead > int(lead_time_mins))
        & (lead <= int(horizon_mins))
    ).astype("int64")
    out["lead_time_mins"] = lead
    return out


#############################
# Hospital + node assignment (top-K anchors from config)
#############################
def add_hospital_and_node_id(
    samples_df: pd.DataFrame,
    patients_df: pd.DataFrame,
    out_run_dir: Path,
    seed: int = 42,
) -> Tuple[pd.DataFrame, Dict[int, int], List[int]]:
    """
    Adds hospitalid and node_id.

    Rule:
      - Pick top-K hospitals by number of CA-positive patients as anchor nodes (K from config).
      - Every other hospital is randomly (but deterministically) assigned to one of the anchors.
      - Node IDs are 0..(K-1) corresponding to the anchor hospitals list order.
      - Writes mapping to out_run_dir / "hospital_to_node.json"

    Prints:
      - CA-positive patient counts for every hospital (descending)
      - The Kth (cutoff) anchor hospital count
      - The (K+1)th hospital count (just missed) when available
    """
    if "hospitalid" not in patients_df.columns:
        raise ValueError("patients_df must contain hospitalid")

    k = int(getattr(config, "NUM_ANCHOR_HOSPITALS", 5))
    if k <= 0:
        raise ValueError("config.NUM_ANCHOR_HOSPITALS must be >= 1")

    assign_seed = int(getattr(config, "NODE_ASSIGNMENT_SEED", seed))

    pmap = patients_df[["patientunitstayid", "hospitalid"]].copy()
    pmap["patientunitstayid"] = pd.to_numeric(pmap["patientunitstayid"], errors="coerce").astype(int)
    pmap["hospitalid"] = pd.to_numeric(pmap["hospitalid"], errors="coerce").astype(int)

    df = samples_df.merge(pmap, on="patientunitstayid", how="left")
    if df["hospitalid"].isna().any():
        n_missing = int(df["hospitalid"].isna().sum())
        raise RuntimeError(f"Missing hospitalid for {n_missing} sample rows after merge.")

    per_patient = (
        df.groupby(["patientunitstayid", "hospitalid"])["label"]
        .max()
        .astype(int)
        .reset_index()
        .rename(columns={"label": "patient_has_positive"})
    )

    pos_counts = (
        per_patient[per_patient["patient_has_positive"] == 1]
        .groupby("hospitalid")["patientunitstayid"]
        .nunique()
        .sort_values(ascending=False)
    )

    if len(pos_counts) == 0:
        raise RuntimeError("No positives at all. Cannot assign anchor hospitals.")

    # Print all hospitals' CA-positive patient counts (descending)
    print("#############################")
    print("CA-positive patient counts per hospital (descending)")
    print(f"Total hospitals with >=1 CA-positive patient: {len(pos_counts)}")
    for hosp_id, cnt in pos_counts.items():
        print(f"Hospital {int(hosp_id)} | CA-positive patients: {int(cnt)}")
    print("#############################")

    k_used = int(min(k, len(pos_counts)))
    topk = pos_counts.head(k_used)

    # Determine cutoff (Kth) and "just missed" (K+1th) if exists
    kth_hospital_id = int(pos_counts.index.astype(int).tolist()[k_used - 1])
    kth_count = int(pos_counts.iloc[k_used - 1])

    missed_exists = (len(pos_counts) >= (k_used + 1))
    missed_hospital_id = int(pos_counts.index.astype(int).tolist()[k_used]) if missed_exists else None
    missed_count = int(pos_counts.iloc[k_used]) if missed_exists else None

    anchor_hospitals = sorted(topk.index.astype(int).tolist())
    mapping: Dict[int, int] = {int(h): int(i) for i, h in enumerate(anchor_hospitals)}

    all_hospitals = sorted(df["hospitalid"].astype(int).unique().tolist())
    non_anchor = [int(h) for h in all_hospitals if int(h) not in mapping]

    rng = np.random.default_rng(int(assign_seed))
    for h in non_anchor:
        node = int(rng.integers(0, len(anchor_hospitals)))
        mapping[int(h)] = node

    df["node_id"] = df["hospitalid"].astype(int).map(mapping).astype(int)

    out_run_dir.mkdir(parents=True, exist_ok=True)
    out_map = out_run_dir / "hospital_to_node.json"
    with open(out_map, "w", encoding="utf-8") as f:
        json.dump({str(k): int(v) for k, v in mapping.items()}, f, indent=2)

    print("#############################")
    print("Hospital -> node mapping created")
    print(f"NUM_ANCHOR_HOSPITALS requested: {k}")
    print(f"NUM_ANCHOR_HOSPITALS used:      {k_used}")
    print(f"Cutoff (Kth) hospital:          {kth_hospital_id} | CA-positive patients: {kth_count}")
    if missed_exists and missed_hospital_id is not None and missed_count is not None:
        print(f"Just missed (K+1) hospital:     {missed_hospital_id} | CA-positive patients: {missed_count}")
    else:
        print("Just missed (K+1) hospital:     N/A (not enough hospitals with positives)")
    print(f"Anchor hospitals:               {anchor_hospitals}")
    print(f"Assignment seed:                {assign_seed}")
    print(f"Mapping saved:                  {out_map}")
    print("#############################")

    return df, mapping, anchor_hospitals


#############################
# Main
#############################
def main() -> None:
    seed = int(getattr(config, "SEED", 42))
    set_seeds(seed)

    data_dir = config.EICU_DATA_DIR
    out_samples = config.samples_path(config.DISEASE)
    run_dir = config.run_dir(config.DISEASE)
    out_meta = run_dir / f"meta__{config.disease_tag(config.DISEASE)}.json"

    print("#############################")
    print("Preparing samples (horizon labeling, vitals-only coverage)")
    print("#############################")
    print(f"Data dir: {data_dir}")
    print(f"Output samples: {out_samples}")

    print("#############################")
    print("Loading diagnosis and patient tables")
    print("#############################")
    diagnoses = read_diagnoses(data_dir)
    patients = read_patients(data_dir)
    all_patients = patients["patientunitstayid"].astype(int).unique().tolist()
    print(f"Total ICU stays in patient.csv.gz: {len(all_patients)}")

    print("#############################")
    print("Computing vitals-only coverage per patient (vp + va)")
    print("#############################")
    max_offsets_vitals = compute_max_offsets_vitals_only(data_dir)
    print(f"Patients with any vitals coverage: {len(max_offsets_vitals)}")

    print("#############################")
    print("Finding disease onset times")
    print("#############################")
    disease_rows = filter_disease_rows(diagnoses, config.DISEASE)
    onset_times = compute_onset_times(disease_rows)
    print(f"Patients with at least one matching diagnosis: {len(onset_times)}")

    print("#############################")
    print("Generating windows")
    print("#############################")
    min_history_mins = int(getattr(config, "MIN_HISTORY_MINS", 60))
    final_horizon_mins = int(getattr(config, "HORIZON_MINS", 240))
    fixed_cohort_enabled = bool(getattr(config, "FIXED_COHORT_ENABLED", False))
    cohort_horizon_mins = (
        int(getattr(config, "COHORT_HORIZON_MINS", final_horizon_mins))
        if fixed_cohort_enabled
        else final_horizon_mins
    )
    stride_mins = int(getattr(config, "STRIDE_MINS", 60))
    lead_time_mins = int(getattr(config, "LEAD_TIME_MINS", 30))

    require_full_horizon_for_negatives = bool(getattr(config, "REQUIRE_FULL_HORIZON", True))
    stop_after_event = True

    all_rows: List[Tuple[int, int, int, float, float]] = []

    eligible_patients = [pid for pid in all_patients if pid in max_offsets_vitals.index]
    print(f"Patients present in vitals coverage index: {len(eligible_patients)}")

    for pid in eligible_patients:
        max_off = int(max_offsets_vitals.loc[pid])
        onset = int(onset_times.loc[pid]) if pid in onset_times.index else None

        rows = generate_patient_windows(
            patient_id=pid,
            max_offset_vitals=max_off,
            onset_time=onset,
            min_history_mins=min_history_mins,
            horizon_mins=cohort_horizon_mins,
            stride_mins=stride_mins,
            lead_time_mins=lead_time_mins,
            require_full_horizon_for_negatives=require_full_horizon_for_negatives,
            stop_after_event=stop_after_event,
        )
        if rows:
            all_rows.extend(rows)

    samples_df = pd.DataFrame(all_rows, columns=["patientunitstayid", "t_end", "label", "t_event", "lead_time_mins"])
    if samples_df.empty:
        raise RuntimeError("No samples were generated. Check disease filter and vitals coverage rules.")

    print_counts(samples_df, f"Generated {cohort_horizon_mins // 60}h cohort (pre-split)")

    #############################
    # Patient-level split
    #############################
    print("#############################")
    print("Assigning patient-level train/val/test split")
    print("#############################")
    samples_df = add_patient_level_splits(samples_df)

    #############################
    # Reproduce the conference protocol: retain the 12h master cohort at 10:1,
    # then relabel the same rows to 4h and apply the final 5:1 limiter.
    #############################
    if fixed_cohort_enabled:
        cohort_ratio = float(getattr(config, "COHORT_NEG_POS_MAX_RATIO", 10.0))
        samples_df = apply_neg_limiter_from_config(
            samples_df,
            max_ratio_override=cohort_ratio,
            stage_name="fixed_cohort",
        )
        samples_df = relabel_fixed_cohort(
            samples_df,
            horizon_mins=final_horizon_mins,
            lead_time_mins=lead_time_mins,
        )
        print_counts(
            samples_df,
            f"Fixed cohort relabelled to {final_horizon_mins // 60}h (before final limiter)",
        )
        samples_df = apply_neg_limiter_from_config(samples_df, stage_name="final")
    else:
        samples_df = apply_neg_limiter_from_config(samples_df, stage_name="final")

    #############################
    # Add hospitalid + node_id (shared by centralised + federated)
    #############################
    samples_df, node_mapping, anchor_hospitals = add_hospital_and_node_id(
        samples_df=samples_df,
        patients_df=patients,
        out_run_dir=run_dir,
        seed=seed,
    )

    #############################
    # Final per-split counts
    #############################
    for sp in ["train", "val", "test"]:
        sub = samples_df[samples_df["split"] == sp]
        print_counts(sub, f"Split={sp} FINAL")

    #############################
    # Save
    #############################
    print("#############################")
    print("Saving samples and meta")
    print("#############################")
    out_samples.parent.mkdir(parents=True, exist_ok=True)
    samples_df.to_csv(out_samples, index=False)

    meta = {
        "disease": {"major": config.DISEASE.major, "subcategory": config.DISEASE.subcategory},
        "labeling": {
            "type": "horizon",
            "horizon_mins": int(final_horizon_mins),
            "lead_time_mins": int(lead_time_mins),
            "rule": "label=1 iff t_end + lead < onset <= t_end + horizon",
        },
        "fixed_cohort": {
            "enabled": bool(fixed_cohort_enabled),
            "cohort_horizon_mins": int(cohort_horizon_mins),
            "cohort_neg_pos_max_ratio": float(
                getattr(config, "COHORT_NEG_POS_MAX_RATIO", 10.0)
            ),
            "procedure": "split 12h cohort, limit to 10:1, relabel to 4h, limit to 5:1",
        },
        "windows": {
            "min_history_mins": int(min_history_mins),
            "stride_mins": int(stride_mins),
            "stop_after_event": bool(stop_after_event),
            "require_full_horizon_for_negatives": bool(require_full_horizon_for_negatives),
        },
        "coverage": {
            "tables": ["vitalPeriodic.csv.gz", "vitalAperiodic.csv.gz"],
            "note": "vitals-only coverage; positives not dropped for sparse vitals (no full-horizon requirement for positives).",
        },
        "neg_limiter": {
            "enabled": bool(getattr(config, "NEG_LIMITER_ENABLED", False)),
            "max_ratio": float(getattr(config, "NEG_POS_MAX_RATIO", 10.0)),
            "random_state": int(getattr(config, "NEG_LIMITER_RANDOM_STATE", getattr(config, "SEED", 42))),
        },
        "node_assignment": {
            "num_anchor_hospitals_requested": int(getattr(config, "NUM_ANCHOR_HOSPITALS", 5)),
            "node_assignment_seed": int(getattr(config, "NODE_ASSIGNMENT_SEED", seed)),
            "num_anchor_hospitals_used": int(len(anchor_hospitals)),
            "anchor_hospitals": [int(h) for h in anchor_hospitals],
            "mapping_file": str((run_dir / "hospital_to_node.json").name),
        },
        "split": {
            "type": "patient_level_train_val_test",
            "test_size": float(getattr(config, "TEST_SIZE", 0.2)),
            "val_size": float(getattr(config, "VAL_SIZE", 0.15)),
            "random_state": int(getattr(config, "SPLIT_RANDOM_STATE", 42)),
        },
        "seed": int(seed),
        "counts": {
            "final_total": int(len(samples_df)),
            "final_pos": int(samples_df["label"].sum()),
            "final_neg": int((samples_df["label"] == 0).sum()),
            "train_patients": int(samples_df.loc[samples_df["split"] == "train", "patientunitstayid"].nunique()),
            "val_patients": int(samples_df.loc[samples_df["split"] == "val", "patientunitstayid"].nunique()),
            "test_patients": int(samples_df.loc[samples_df["split"] == "test", "patientunitstayid"].nunique()),
        },
    }

    with open(out_meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("Done.")
    print(f"Saved samples: {out_samples}")
    print(f"Saved meta: {out_meta}")


if __name__ == "__main__":
    main()
