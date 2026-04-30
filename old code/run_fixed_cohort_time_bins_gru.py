from __future__ import annotations

import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

import config
from train_eval_gru import TrainConfig, train_and_eval


LEAD_TIME_MINS = int(getattr(config, "LEAD_TIME_MINS", 30))
NEG_TO_POS_RATIO = 5.0
TIME_BINS: List[Tuple[str, int, int]] = [
    ("30m_2h", 30, 120),
    ("2h_4h", 120, 240),
    ("4h_6h", 240, 360),
    ("6h_12h", 360, 720),
]


def _feature_mode() -> str:
    mode = str(getattr(config, "FEATURE_MODE", "all")).strip().lower()
    valid = {"all", "vitals", "vitals+demo"}
    if mode not in valid:
        raise ValueError(f"Unsupported config.FEATURE_MODE={mode!r}. Expected one of {sorted(valid)}")
    return mode


def _feature_mode_tag() -> str:
    return _feature_mode().replace("+", "_")


def _master_run_dir() -> Path:
    return config.run_dir(config.DISEASE)


def _comparison_dir() -> Path:
    ratio_tag = str(NEG_TO_POS_RATIO).replace(".", "p")
    out = _master_run_dir() / f"fixed_cohort_time_bins__feat{_feature_mode_tag()}__neg{ratio_tag}to1"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _master_samples_path() -> Path:
    return config.samples_path(config.DISEASE)


def _relabel_samples(master_df: pd.DataFrame, low_mins: int, high_mins: int) -> pd.DataFrame:
    out = master_df.copy()

    t_event = pd.to_numeric(out["t_event"], errors="coerce")
    t_end = pd.to_numeric(out["t_end"], errors="coerce")
    lead = t_event - t_end

    is_pos = t_event.notna() & (lead > low_mins) & (lead <= high_mins)
    out["label"] = is_pos.astype("int64")
    out["lead_time_mins"] = lead
    return out


def _apply_neg_ratio_cap(df: pd.DataFrame, ratio: float) -> pd.DataFrame:
    if ratio <= 0:
        raise ValueError("NEG_TO_POS_RATIO must be positive.")

    rs = int(getattr(config, "NEG_LIMITER_RANDOM_STATE", getattr(config, "SEED", 42)))
    out_parts: List[pd.DataFrame] = []

    for split, part in df.groupby("split", sort=False):
        pos = part[part["label"] == 1]
        neg = part[part["label"] == 0]
        max_neg = int(len(pos) * ratio)

        if len(pos) == 0 or len(neg) <= max_neg:
            keep = part
        else:
            neg_keep = neg.sample(n=max_neg, random_state=rs)
            keep = pd.concat([pos, neg_keep], axis=0)

        out_parts.append(keep)

    out = pd.concat(out_parts, axis=0, ignore_index=True)
    return out


def _save_relabelled_samples(master_df: pd.DataFrame, bin_name: str, low_mins: int, high_mins: int, out_dir: Path) -> Path:
    relabelled = _relabel_samples(master_df, low_mins, high_mins)
    relabelled = _apply_neg_ratio_cap(relabelled, NEG_TO_POS_RATIO)
    out_path = out_dir / f"samples__fixed12hcohort__bin_{bin_name}.csv"
    relabelled.to_csv(out_path, index=False)
    return out_path


def _copy_if_exists(src: str | None, dest: Path) -> str | None:
    if not src:
        return None
    src_path = Path(src)
    if not src_path.exists():
        return None
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_path, dest)
    return str(dest)


def main() -> None:
    master_samples = _master_samples_path()
    if not master_samples.exists():
        raise FileNotFoundError(f"Master 12h samples not found: {master_samples}")

    feature_mode = _feature_mode()
    if feature_mode == "all":
        top_k = int(getattr(config, "DEFAULT_TOPK", 60))
        rank_path = str(config.stability_combined_path(config.DISEASE))
        if not Path(rank_path).exists():
            raise FileNotFoundError(f"Ranking CSV not found: {rank_path}")
    else:
        top_k = None
        rank_path = None

    out_dir = _comparison_dir()
    master_df = pd.read_csv(master_samples)
    needed = {"patientunitstayid", "t_end", "t_event", "label", "split"}
    missing = needed - set(master_df.columns)
    if missing:
        raise ValueError(f"Master samples missing columns: {sorted(missing)}")

    cfg = TrainConfig()
    rows: List[Dict[str, float | int | str]] = []

    print("############################################################")
    print("Fixed-cohort centralized GRU comparison by non-overlapping time bins")
    print(f"Master cohort: {master_samples}")
    print(f"Feature parquet: {config.features_path(config.DISEASE)}")
    print(f"Feature mode: {feature_mode}")
    print(f"Negative:positive cap per split = {NEG_TO_POS_RATIO}:1")
    print(f"Ranking CSV: {rank_path}")
    print(f"Output dir: {out_dir}")
    print("############################################################")

    for bin_name, low_mins, high_mins in TIME_BINS:
        print("============================================================")
        print(f"Running fixed 12h cohort with time-to-event bin = {bin_name} ({low_mins}, {high_mins}] mins")
        print("============================================================")

        samples_path = _save_relabelled_samples(master_df, bin_name, low_mins, high_mins, out_dir)
        relabelled = pd.read_csv(samples_path, usecols=["label"])
        n_total = int(len(relabelled))
        n_pos = int((pd.to_numeric(relabelled["label"], errors="coerce") == 1).sum())
        n_neg = int(n_total - n_pos)
        print(f"Relabelled samples: {samples_path}")
        print(f"Windows total={n_total} pos={n_pos} neg={n_neg}")

        out = train_and_eval(
            disease=config.DISEASE,
            cfg=cfg,
            top_k=top_k,
            rank_path=rank_path,
            samples_path=str(samples_path),
        )

        copied_roc = _copy_if_exists(out["extra"].get("roc_path"), out_dir / f"roc_test__bin_{bin_name}.png")
        copied_pr = _copy_if_exists(out["extra"].get("pr_path"), out_dir / f"pr_test__bin_{bin_name}.png")
        copied_model = _copy_if_exists(out["extra"].get("model_path"), out_dir / f"model__bin_{bin_name}.pt")
        copied_meta = _copy_if_exists(out["extra"].get("metadata_path"), out_dir / f"metadata__bin_{bin_name}.json")
        copied_preds = _copy_if_exists(
            out["extra"].get("predictions_path"),
            out_dir / f"test_predictions__bin_{bin_name}.csv",
        )

        rows.append(
            {
                "time_bin": bin_name,
                "bin_low_mins_exclusive": int(low_mins),
                "bin_high_mins_inclusive": int(high_mins),
                "samples_path": str(samples_path),
                "n_windows": n_total,
                "n_pos_windows": n_pos,
                "n_neg_windows": n_neg,
                "neg_to_pos_ratio_cap": float(NEG_TO_POS_RATIO),
                "feature_mode": out["extra"].get("feature_mode"),
                "top_k": "" if top_k is None else int(top_k),
                "n_features": int(out["extra"]["n_features"]),
                "train_auroc": float(out["train"]["auroc"]),
                "train_auprc": float(out["train"]["auprc"]),
                "test_auroc": float(out["test"]["auroc"]),
                "test_auprc": float(out["test"]["auprc"]),
                "threshold": float(out["test"]["threshold"]),
                "test_accuracy": float(out["test"]["accuracy"]),
                "test_precision": float(out["test"]["precision"]),
                "test_recall": float(out["test"]["recall"]),
                "test_f1": float(out["test"]["f1"]),
                "test_fpr": float(out["test"]["fpr"]),
                "test_tn": float(out["test"]["tn"]),
                "test_fp": float(out["test"]["fp"]),
                "test_fn": float(out["test"]["fn"]),
                "test_tp": float(out["test"]["tp"]),
                "test_n_pos": float(out["test"]["n_pos"]),
                "test_n_neg": float(out["test"]["n_neg"]),
                "cpu_peak_mib": float(out["extra"]["cpu_peak_mib"]),
                "runtime_sec": float(out["extra"]["runtime_sec"]),
                "roc_path": copied_roc or "",
                "pr_path": copied_pr or "",
                "model_path": copied_model or "",
                "metadata_path": copied_meta or "",
                "predictions_path": copied_preds or "",
            }
        )

    results_df = pd.DataFrame(rows)
    results_path = out_dir / f"gru_results__fixed12hcohort__time_bins__feat{_feature_mode_tag()}.csv"
    results_df.to_csv(results_path, index=False)

    print("############################################################")
    print("Fixed-cohort time-bin comparison complete")
    print(results_df.to_string(index=False))
    print(f"Saved comparison CSV: {results_path}")
    print("############################################################")


if __name__ == "__main__":
    main()
