# add_hospital_to_samples.py
# Finds latest run folder under ../Outputs/* that contains a samples*.csv
# Merges hospitalid from ../eICU(v2.0)/patient.csv.gz
# Writes: samples_with_hospital.csv in the same run folder

from __future__ import annotations

from pathlib import Path
import pandas as pd


def find_latest_samples_csv(project_root: Path) -> Path:
    # Match your actual naming: samples_*.csv (and also allow plain samples.csv)
    candidates = list(project_root.glob("Outputs/*/samples*.csv"))

    if not candidates:
        # Extra fallback: sometimes people put it directly under Outputs/
        candidates = list(project_root.glob("Outputs/samples*.csv"))

    if not candidates:
        raise FileNotFoundError(
            "No samples*.csv found under Outputs/*/ or Outputs/.\n"
            "Expected something like Outputs/<run_name>/samples_....csv"
        )

    # Pick most recently modified file
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)

    if len(candidates) > 1:
        print("Found multiple samples files. Using most recently modified:")
        for c in candidates[:10]:
            print("  ", c)

    return candidates[0]


def main() -> None:
    code_dir = Path(__file__).resolve().parent
    project_root = code_dir.parent

    samples_csv = find_latest_samples_csv(project_root)
    run_dir = samples_csv.parent

    patient_csv = project_root / "eICU(v2.0)" / "patient.csv.gz"
    if not patient_csv.exists():
        raise FileNotFoundError(f"Missing patient.csv.gz at: {patient_csv}")

    print("Using samples:", samples_csv)
    print("Run dir:", run_dir)
    print("Using patient file:", patient_csv)

    samples = pd.read_csv(samples_csv)

    required = {"patientunitstayid", "label", "split"}
    missing = required - set(samples.columns)
    if missing:
        raise ValueError(f"samples file missing columns: {sorted(missing)}")

    patient = pd.read_csv(
        patient_csv,
        compression="gzip",
        usecols=["patientunitstayid", "hospitalid"],
    ).drop_duplicates(subset=["patientunitstayid"], keep="first")

    merged = samples.merge(patient, on="patientunitstayid", how="left")

    n_missing = int(merged["hospitalid"].isna().sum())
    if n_missing:
        bad = merged.loc[merged["hospitalid"].isna(), "patientunitstayid"].head(10).tolist()
        raise RuntimeError(
            f"{n_missing} rows had no hospitalid after merge. "
            f"Example patientunitstayid: {bad}"
        )

    merged["hospitalid"] = merged["hospitalid"].astype(int)

    out_path = run_dir / "samples_with_hospital.csv"
    merged.to_csv(out_path, index=False)

    print("Wrote:", out_path)
    print("Unique hospitals:", int(merged["hospitalid"].nunique()))


if __name__ == "__main__":
    main()