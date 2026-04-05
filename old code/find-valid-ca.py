# audit_ca_history.py
# Count cardiac arrest (CA) stays from diagnosis.csv(.gz) and estimate how many have
# at least H hours of vitals history before onset, using vitalsPeriodic + vitalAperiodic.
#
# Output:
# - total unique CA stays (diagnosis-defined)
# - CA stays with any vitals at all
# - CA stays with any vitals BEFORE onset
# - CA stays with >= H hours vitals history before onset for H=1..12
#
# Definition:
# - CA stay: any diagnosis row with major="cardiovascular" and sub="cardiac arrest"
# - Onset time: min(diagnosisoffset) for that stay
# - "Has H hours history": min_vitals_offset <= onset AND (onset - min_vitals_offset) >= H*60
#
# Usage:
#   python audit_ca_history.py --data_dir "../eICU(v2.0)"

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="../eICU(v2.0)", help="Path to eICU(v2.0) directory")
    p.add_argument("--chunksize", type=int, default=800_000, help="CSV chunksize for vitals files")
    return p.parse_args()


def load_ca_onsets(diag_path: Path) -> pd.Series:
    usecols = ["patientunitstayid", "diagnosisoffset", "diagnosisstring"]
    df = pd.read_csv(diag_path, compression="infer", low_memory=False, usecols=usecols)

    parts = df["diagnosisstring"].astype(str).str.split("|", n=2, expand=True)
    major = parts[0].astype(str).str.strip().str.lower()
    sub = parts[1].astype(str).str.strip().str.lower() if parts.shape[1] > 1 else ""

    mask = (major == "cardiovascular") & (sub == "cardiac arrest")
    ca = df.loc[mask, ["patientunitstayid", "diagnosisoffset"]].copy()

    ca["patientunitstayid"] = pd.to_numeric(ca["patientunitstayid"], errors="coerce")
    ca["diagnosisoffset"] = pd.to_numeric(ca["diagnosisoffset"], errors="coerce")
    ca = ca.dropna(subset=["patientunitstayid", "diagnosisoffset"])
    ca["patientunitstayid"] = ca["patientunitstayid"].astype(int)
    ca["diagnosisoffset"] = ca["diagnosisoffset"].astype(int)

    onset = ca.groupby("patientunitstayid")["diagnosisoffset"].min()
    return onset.astype(int)


def scan_vitals_min_offset(
    vitals_path: Path,
    offset_col: str,
    pids_of_interest: set[int],
    chunksize: int,
    min_offset: Dict[int, int],
) -> None:
    if not vitals_path.exists():
        print(f"WARNING: missing {vitals_path.name}, skipping")
        return

    reader = pd.read_csv(
        vitals_path,
        compression="infer",
        low_memory=False,
        usecols=["patientunitstayid", offset_col],
        chunksize=chunksize,
    )

    for chunk in reader:
        chunk["patientunitstayid"] = pd.to_numeric(chunk["patientunitstayid"], errors="coerce")
        chunk[offset_col] = pd.to_numeric(chunk[offset_col], errors="coerce")
        chunk = chunk.dropna(subset=["patientunitstayid", offset_col])
        if chunk.empty:
            continue

        pid = chunk["patientunitstayid"].astype(np.int64)
        mask = pid.isin(pids_of_interest)
        if not mask.any():
            continue

        sub = chunk.loc[mask, ["patientunitstayid", offset_col]].copy()
        sub["patientunitstayid"] = sub["patientunitstayid"].astype(int)
        sub[offset_col] = sub[offset_col].astype(int)

        grp = sub.groupby("patientunitstayid")[offset_col].min()
        for pid_i, v in grp.items():
            v_i = int(v)
            if pid_i not in min_offset or v_i < min_offset[pid_i]:
                min_offset[pid_i] = v_i


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)

    diag_path = data_dir / "diagnosis.csv.gz"
    if not diag_path.exists():
        # your files might be .csv not .csv.gz (some installs differ)
        alt = data_dir / "diagnosis.csv"
        if alt.exists():
            diag_path = alt
        else:
            raise FileNotFoundError(f"Missing: {diag_path} (and diagnosis.csv)")

    vp_path = data_dir / "vitalPeriodic.csv.gz"
    va_path = data_dir / "vitalAperiodic.csv.gz"

    print("Loading CA onsets from diagnosis...")
    onset = load_ca_onsets(diag_path)
    ca_pids = set(onset.index.astype(int).tolist())

    print("#############################")
    print(f"Data dir: {data_dir.resolve()}")
    print(f"Unique CA stays (diagnosis-defined): {len(ca_pids)}")
    print("#############################")

    print("Scanning vitals for earliest offset per CA stay (vitals-only)...")
    min_offset: Dict[int, int] = {}

    scan_vitals_min_offset(vp_path, "observationoffset", ca_pids, args.chunksize, min_offset)
    scan_vitals_min_offset(va_path, "observationoffset", ca_pids, args.chunksize, min_offset)

    ca_with_any_vitals = set(min_offset.keys())
    n_any_vitals = len(ca_with_any_vitals)

    # vitals before onset + history thresholds
    n_vitals_before_onset = 0

    # count for each hour threshold 1..12
    hour_thresholds = list(range(1, 13))
    counts_by_hour = {h: 0 for h in hour_thresholds}

    for pid in ca_pids:
        if pid not in min_offset:
            continue
        t0 = int(min_offset[pid])
        te = int(onset.loc[pid])

        if t0 <= te:
            n_vitals_before_onset += 1
            dt = te - t0
            for h in hour_thresholds:
                if dt >= h * 60:
                    counts_by_hour[h] += 1

    print("#############################")
    print("VITALS-ONLY HISTORY AUDIT (CA stays)")
    print(f"CA stays with any vitals at all:        {n_any_vitals}")
    print(f"CA stays with vitals before onset:      {n_vitals_before_onset}")
    print("")
    print("CA stays with >= H hours vitals history before onset:")
    for h in hour_thresholds:
        c = counts_by_hour[h]
        pct = (c / len(ca_pids)) if len(ca_pids) > 0 else 0.0
        print(f"  >= {h:2d}h: {c:5d}  ({pct:.2%} of all CA)")
    print("#############################")


if __name__ == "__main__":
    main()