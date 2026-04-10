from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Dict, List

import pandas as pd


def _require_columns(df: pd.DataFrame, cols: List[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect federated sample distribution by node, including positive/negative patient "
            "counts and time-window statistics for t_end."
        )
    )
    parser.add_argument(
        "samples_csv",
        nargs="?",
        default=None,
        help=(
            "Path to samples.csv or samples_with_node.csv. If omitted, the script will try to use "
            "config.samples_path(config.DISEASE) if config.py is available."
        ),
    )
    parser.add_argument(
        "--save-csv",
        type=Path,
        default=None,
        help="Optional output path to save per-node summary as a CSV file."
    )
    return parser.parse_args()


def summarize_time_windows(df: pd.DataFrame, node_id: int) -> str:
    if "t_end" not in df.columns:
        return "t_end column not found"
    if df["t_end"].dropna().empty:
        return "no valid t_end values"

    values = pd.to_numeric(df["t_end"], errors="coerce").dropna().astype(float)
    if values.empty:
        return "no valid t_end values"

    quantiles = values.quantile([0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])
    total = len(values)
    bins = pd.cut(values, bins=5)
    counts = bins.value_counts(sort=False)
    bucket_summary = "; ".join(
        f"{interval.left:.0f}-{interval.right:.0f}:{count}" for interval, count in counts.items()
    )

    return (
        f"count={total}, min={quantiles.iloc[0]:.0f}, p10={quantiles.iloc[1]:.0f}, "
        f"p25={quantiles.iloc[2]:.0f}, median={quantiles.iloc[3]:.0f}, p75={quantiles.iloc[4]:.0f}, "
        f"p90={quantiles.iloc[5]:.0f}, max={quantiles.iloc[6]:.0f}, bins=[{bucket_summary}]"
    )


def main() -> None:
    args = _parse_args()

    samples_path = args.samples_csv
    if samples_path is None:
        try:
            import config

            samples_path = Path(config.samples_path(config.DISEASE))
        except Exception as exc:
            raise SystemExit(
                "samples_csv path is required when config.py is unavailable or fails to load: "
                f"{exc}"
            )
    else:
        samples_path = Path(samples_path)

    if not samples_path.exists():
        raise FileNotFoundError(f"Could not find samples CSV at: {samples_path.resolve()}")

    df = pd.read_csv(samples_path)
    _require_columns(df, ["node_id", "label", "split", "patientunitstayid"])

    df = df.copy()
    df["node_id"] = pd.to_numeric(df["node_id"], errors="coerce")
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df["patientunitstayid"] = pd.to_numeric(df["patientunitstayid"], errors="coerce")
    df = df.dropna(subset=["node_id", "label", "patientunitstayid"])
    df["node_id"] = df["node_id"].astype(int)
    df["label"] = df["label"].astype(int)
    df["patientunitstayid"] = df["patientunitstayid"].astype(int)
    df["split"] = df["split"].astype(str)

    node_ids = sorted(df["node_id"].unique().tolist())
    total_windows = len(df)
    total_patients = df["patientunitstayid"].nunique()
    total_pos = int(df["label"].sum())
    total_neg = int((df["label"] == 0).sum())
    total_prevalence = total_pos / total_windows if total_windows else 0.0

    print("#############################")
    print(f"Samples file: {samples_path.resolve()}")
    print(f"Total nodes:  {len(node_ids)}")
    print(f"Node ids:     {node_ids}")
    print(f"Total windows:  {total_windows}")
    print(f"Total patients: {total_patients}")
    print(f"Total positives: {total_pos}")
    print(f"Total negatives: {total_neg}")
    print(f"Overall prevalence: {total_prevalence:.4f}")
    print("#############################\n")

    rows: List[Dict[str, object]] = []

    for node_id in node_ids:
        node_df = df[df["node_id"] == node_id]
        node_total = len(node_df)
        node_pos = int(node_df["label"].sum())
        node_neg = int((node_df["label"] == 0).sum())
        node_patients = node_df["patientunitstayid"].nunique()
        node_pos_patients = node_df[node_df["label"] == 1]["patientunitstayid"].nunique()
        node_neg_patients = node_df[node_df["label"] == 0]["patientunitstayid"].nunique()
        node_prevalence = node_pos / node_total if node_total else 0.0
        pos_patient_ratio = node_neg_patients / node_pos_patients if node_pos_patients else float("inf")
        patient_prevalence = node_pos_patients / node_patients if node_patients else 0.0

        print(f"Node {node_id}")
        print(f"  windows: total={node_total}, pos={node_pos}, neg={node_neg}, prevalence={node_prevalence:.4f}")
        print(
            f"  patients: total={node_patients}, pos={node_pos_patients}, neg={node_neg_patients}, "
            f"neg:pos_patients={pos_patient_ratio:.2f}:1, prevalence={patient_prevalence:.4f}"
        )

        if "t_end" in node_df.columns:
            print(f"  t_end summary: {_summarize_t_end(node_df)}")
        else:
            print("  t_end summary: column missing")

        print("  split breakdown:")
        split_summary = (
            node_df.groupby("split")["label"]
            .agg(total="size", pos="sum")
            .reset_index()
        )
        split_summary["neg"] = split_summary["total"] - split_summary["pos"]
        for _, r in split_summary.sort_values("split").iterrows():
            split_name = str(r["split"])
            total = int(r["total"])
            pos = int(r["pos"])
            neg = int(r["neg"])
            split_prevalence = pos / total if total else 0.0
            print(
                f"    {split_name:<5} total={total:>6}, pos={pos:>6}, neg={neg:>6}, prevalence={split_prevalence:.4f}"
            )
        print("")

        rows.append(
            {
                "node_id": node_id,
                "windows_total": node_total,
                "windows_pos": node_pos,
                "windows_neg": node_neg,
                "patients_total": node_patients,
                "patients_pos": node_pos_patients,
                "patients_neg": node_neg_patients,
                "prevalence_windows": node_prevalence,
                "prevalence_patients": patient_prevalence,
            }
        )

    if args.save_csv:
        out_df = pd.DataFrame(rows)
        args.save_csv.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(args.save_csv, index=False)
        print(f"Saved per-node summary to: {args.save_csv.resolve()}")


def _summarize_t_end(node_df: pd.DataFrame) -> str:
    if "t_end" not in node_df.columns:
        return "missing"
    t_end_values = pd.to_numeric(node_df["t_end"], errors="coerce").dropna().astype(float)
    if t_end_values.empty:
        return "no valid t_end values"

    quantiles = t_end_values.quantile([0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])
    bins = pd.cut(t_end_values, bins=5)
    counts = bins.value_counts(sort=False)
    bucket_summary = ", ".join(
        f"{int(interval.left)}-{int(interval.right)}:{count}" for interval, count in counts.items()
    )
    return (
        f"count={len(t_end_values)}, min={quantiles.iloc[0]:.0f}, "
        f"p10={quantiles.iloc[1]:.0f}, median={quantiles.iloc[3]:.0f}, "
        f"p90={quantiles.iloc[5]:.0f}, max={quantiles.iloc[6]:.0f}, bins=[{bucket_summary}]"
    )


if __name__ == "__main__":
    main()
