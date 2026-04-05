# inspect_nodes.py
# Print node statistics from samples.csv:
# - number of nodes
# - per-node pos/neg totals
# - per-node split breakdown (train/val/test) with pos/neg
#
# By default uses:
#   config.samples_path(config.DISEASE)
#
# Optional override:
#   python inspect_nodes.py path/to/samples.csv

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import config


def _req_cols(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in samples.csv: {missing}")


def main() -> None:
    # If a manual path is provided, use it.
    if len(sys.argv) > 1:
        samples_path = Path(sys.argv[1])
    else:
        # Default: use same logic as prepare_data.py
        samples_path = config.samples_path(config.DISEASE)

    if not samples_path.exists():
        raise FileNotFoundError(f"Could not find samples.csv at: {samples_path.resolve()}")

    df = pd.read_csv(samples_path)

    _req_cols(df, ["node_id", "split", "label"])

    # Clean types
    df["node_id"] = pd.to_numeric(df["node_id"], errors="coerce")
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df["split"] = df["split"].astype(str)

    df = df.dropna(subset=["node_id", "label"])
    df["node_id"] = df["node_id"].astype(int)
    df["label"] = df["label"].astype(int)

    n_nodes = df["node_id"].nunique()
    nodes_sorted = sorted(df["node_id"].unique().tolist())

    print("#############################")
    print(f"samples.csv: {samples_path.resolve()}")
    print(f"Total rows: {len(df)}")
    print(f"Num nodes:  {n_nodes}")
    print(f"Node ids:   {nodes_sorted}")
    print("#############################")

    # Overall per-node totals
    totals = (
        df.groupby("node_id")["label"]
        .agg(total="size", pos="sum")
        .reset_index()
    )
    totals["neg"] = totals["total"] - totals["pos"]
    totals["neg_pos_ratio"] = totals.apply(
        lambda r: (r["neg"] / r["pos"]) if r["pos"] > 0 else float("inf"),
        axis=1,
    )

    print("\n=== Per-node totals ===")
    for _, r in totals.sort_values("node_id").iterrows():
        node = int(r["node_id"])
        total = int(r["total"])
        pos = int(r["pos"])
        neg = int(r["neg"])
        ratio = r["neg_pos_ratio"]
        ratio_str = f"{ratio:.2f}:1" if ratio != float("inf") else "inf"
        prev = (pos / total) if total > 0 else 0.0
        print(
            f"Node {node:>3} | total={total:>8} | pos={pos:>8} | "
            f"neg={neg:>8} | neg:pos={ratio_str:>7} | prev={prev:.4f}"
        )

    # Per-node split breakdown
    print("\n=== Per-node split breakdown (pos/neg) ===")
    split_counts = (
        df.groupby(["node_id", "split"])["label"]
        .agg(total="size", pos="sum")
        .reset_index()
    )
    split_counts["neg"] = split_counts["total"] - split_counts["pos"]

    splits_order = ["train", "val", "test"]

    for node in nodes_sorted:
        print(f"\nNode {node}")
        sub = split_counts[split_counts["node_id"] == node].copy()
        sub["split_order"] = sub["split"].apply(
            lambda s: splits_order.index(s) if s in splits_order else 999
        )
        sub = sub.sort_values(["split_order", "split"])

        for _, r in sub.iterrows():
            sp = str(r["split"])
            total = int(r["total"])
            pos = int(r["pos"])
            neg = int(r["neg"])
            ratio = (neg / pos) if pos > 0 else float("inf")
            ratio_str = f"{ratio:.2f}:1" if ratio != float("inf") else "inf"
            prev = (pos / total) if total > 0 else 0.0
            print(
                f"  {sp:<5} | total={total:>8} | pos={pos:>8} | "
                f"neg={neg:>8} | neg:pos={ratio_str:>7} | prev={prev:.4f}"
            )


if __name__ == "__main__":
    main()