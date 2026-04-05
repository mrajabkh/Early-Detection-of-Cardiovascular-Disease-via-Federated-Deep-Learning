"""
Analyse cardiac arrest distribution per hospital (eICU v2.0)

Folder layout (your setup):
    project_root/
        Code/
            analyse_cardiac_arrest_distribution.py   <-- this file
        eICU(v2.0)/
            patient.csv.gz
            diagnosis.csv.gz
            ...
        Outputs/
            (files will be written here)

Outputs:
    Outputs/cardiac_arrest_per_hospital.txt
    Outputs/cardiac_arrest_distribution_per_hospital.csv

Sorted descending by number of cardiac arrest stays.
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd


def main() -> None:
    # Resolve paths relative to this script file
    script_path = Path(__file__).resolve()
    project_root = script_path.parent.parent  # Code/.. -> project_root

    data_dir = project_root / "eICU(v2.0)"
    outputs_dir = project_root / "Outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    patient_file = data_dir / "patient.csv.gz"
    diagnosis_file = data_dir / "diagnosis.csv.gz"

    output_txt = outputs_dir / "cardiac_arrest_per_hospital.txt"
    output_csv = outputs_dir / "cardiac_arrest_distribution_per_hospital.csv"

    # Basic checks so it fails loudly and clearly
    if not patient_file.exists():
        raise FileNotFoundError(f"Missing file: {patient_file}")
    if not diagnosis_file.exists():
        raise FileNotFoundError(f"Missing file: {diagnosis_file}")

    print(f"Reading: {patient_file.name}")
    patient = pd.read_csv(
        patient_file,
        usecols=["patientunitstayid", "hospitalid"],
        compression="gzip",
        low_memory=False,
    )

    print(f"Reading: {diagnosis_file.name}")
    diagnosis = pd.read_csv(
        diagnosis_file,
        usecols=["patientunitstayid", "diagnosisstring"],
        compression="gzip",
        low_memory=False,
    )

    # Identify cardiac arrest stays via diagnosisstring keyword
    print("Filtering diagnosis rows for 'cardiac arrest' (case-insensitive)...")
    cardiac_rows = diagnosis[
        diagnosis["diagnosisstring"].astype(str).str.contains("cardiac arrest", case=False, na=False)
    ]

    cardiac_stay_ids = cardiac_rows["patientunitstayid"].dropna().unique()
    print(f"Unique cardiac arrest ICU stays found: {len(cardiac_stay_ids)}")

    # Map stays -> hospitals
    cardiac_patient_df = patient[patient["patientunitstayid"].isin(cardiac_stay_ids)]

    total_per_hospital = patient.groupby("hospitalid").size()
    cardiac_per_hospital = cardiac_patient_df.groupby("hospitalid").size()

    summary = pd.DataFrame(
        {
            "total_icu_stays": total_per_hospital,
            "cardiac_arrest_stays": cardiac_per_hospital,
        }
    ).fillna(0)

    summary["cardiac_arrest_stays"] = summary["cardiac_arrest_stays"].astype(int)
    summary["prevalence"] = summary["cardiac_arrest_stays"] / summary["total_icu_stays"]

    # Sort descending by cardiac arrest count
    summary = summary.sort_values(by="cardiac_arrest_stays", ascending=False)

    # Save CSV
    summary.to_csv(output_csv, index=True)

    # Save readable TXT
    with open(output_txt, "w", encoding="utf-8") as f:
        f.write("Cardiac Arrest Distribution per Hospital (eICU v2.0)\n")
        f.write("Sorted by cardiac_arrest_stays (descending)\n\n")
        for hospital_id, row in summary.iterrows():
            f.write(
                "Hospital {hid} | Total ICU stays: {total} | Cardiac arrest stays: {ca} | Prevalence: {prev:.6f}\n".format(
                    hid=int(hospital_id),
                    total=int(row["total_icu_stays"]),
                    ca=int(row["cardiac_arrest_stays"]),
                    prev=float(row["prevalence"]),
                )
            )

    print(f"Saved: {output_txt}")
    print(f"Saved: {output_csv}")
    print("Done.")


if __name__ == "__main__":
    main()
