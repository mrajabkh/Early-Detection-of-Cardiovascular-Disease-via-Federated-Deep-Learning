# make_nodes_and_node_samples.py
# Uses samples_with_hospital.csv in the latest Outputs/<run_name>/ folder
# Writes:
#   - hospital_to_node.json
#   - samples_with_node.csv
# into the same run folder

from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd


SEED = 42
MIN_POS = 150


def find_latest_samples_with_hospital(project_root: Path) -> Path:
    candidates = list(project_root.glob("Outputs/*/samples_with_hospital.csv"))
    if not candidates:
        candidates = list(project_root.glob("Outputs/samples_with_hospital.csv"))

    if not candidates:
        raise FileNotFoundError(
            "No samples_with_hospital.csv found under Outputs/*/ or Outputs/.\n"
            "Run add_hospital_to_samples.py first."
        )

    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    if len(candidates) > 1:
        print("Found multiple samples_with_hospital.csv files. Using most recent:")
        for c in candidates[:10]:
            print("  ", c)
    return candidates[0]


def main() -> None:
    code_dir = Path(__file__).resolve().parent
    project_root = code_dir.parent

    in_csv = find_latest_samples_with_hospital(project_root)
    run_dir = in_csv.parent

    print("Using input:", in_csv)
    print("Run dir:", run_dir)

    df = pd.read_csv(in_csv)

    required = {"patientunitstayid", "hospitalid", "label", "split"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in samples_with_hospital.csv: {sorted(missing)}")

    df["hospitalid"] = df["hospitalid"].astype(int)

    train_df = df[df["split"] == "train"]
    if len(train_df) == 0:
        raise RuntimeError("No TRAIN rows found (split == 'train').")

    # Anchor hospitals based on TRAIN positives only (recommended)
    pos_counts = (
        train_df.groupby("hospitalid")["label"]
        .sum()
        .astype(int)
        .sort_values(ascending=False)
    )

    anchor_hospitals = pos_counts[pos_counts >= MIN_POS].index.astype(int).tolist()

    if len(anchor_hospitals) == 0:
        raise RuntimeError(f"No hospitals found with >= {MIN_POS} positives in TRAIN.")

    if len(anchor_hospitals) != 5:
        print(f"WARNING: Expected ~5 anchors, found {len(anchor_hospitals)} with >= {MIN_POS} positives.")

    # Map anchors to node_id 0..K-1
    hospital_to_node: dict[int, int] = {int(h): int(i) for i, h in enumerate(anchor_hospitals)}

    # Randomly assign the rest to one of the anchor nodes
    all_hospitals = sorted(df["hospitalid"].unique().astype(int).tolist())
    non_anchors = [h for h in all_hospitals if h not in hospital_to_node]

    rng = np.random.default_rng(SEED)
    rand_nodes = rng.integers(low=0, high=len(anchor_hospitals), size=len(non_anchors))

    for h, node_id in zip(non_anchors, rand_nodes):
        hospital_to_node[int(h)] = int(node_id)

    df["node_id"] = df["hospitalid"].map(hospital_to_node).astype(int)

    out_samples = run_dir / "samples_with_node.csv"
    df.to_csv(out_samples, index=False)

    out_json = run_dir / "hospital_to_node.json"
    payload = {
        "seed": SEED,
        "min_pos": MIN_POS,
        "anchor_hospitals": [int(x) for x in anchor_hospitals],
        "hospital_to_node": {str(int(k)): int(v) for k, v in hospital_to_node.items()},
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)

    print("Wrote:", out_samples)
    print("Wrote:", out_json)

    print("\nAnchors (node_id -> hospitalid -> train positives):")
    for i, h in enumerate(anchor_hospitals):
        print(f"  node {i}: hospital {h} (pos={int(pos_counts.loc[h])})")

    print("\nTrain summary per node (count, pos, pos_rate):")
    print(df[df["split"] == "train"].groupby("node_id")["label"].agg(["count", "sum", "mean"]))


if __name__ == "__main__":
    main()