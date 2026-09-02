# aggregate_features.py
# Build rolling-window feature parquets from samples.csv using eICU tables (memory safe).
#
# Outputs:
# - full features parquet (existing behavior, now including baseline features)
# - vitals-only parquet
# - baseline-only parquet
#
# Key memory fixes:
# - Do NOT repeatedly pd.concat into a growing feats dataframe.
# - Build blocks in a list and concat once at the end.
# - Use float32 feature blocks.
# - For charting pivots, DO NOT concat many small dataframes (causes consolidation OOM).
#   Instead, preallocate a numpy array and fill it.
#
# Location: Project/Code/aggregate_features.py

from __future__ import annotations

import re
from pathlib import Path
import argparse
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import config
from missing_value_imputation import (
    classify_feature_columns,
    impute_from_training_split,
    save_medians,
)


#############################
# Baseline feature config
#############################
PATIENT_BASE_NUMERIC_COLS = [
    "age",
    "admissionheight",
    "admissionweight",
    "unitvisitnumber",
]

PATIENT_BASE_CATEGORICAL_COLS = [
    "gender",
    "ethnicity",
    "hospitaladmitsource",
    "unitadmitsource",
    "unitstaytype",
    "unittype",
]

APACHE_BASE_KEEP_COLS = [
    "aids",
    "hepaticfailure",
    "lymphoma",
    "metastaticcancer",
    "leukemia",
    "immunosuppression",
    "cirrhosis",
    "diabetes",
    "ima",
    "bedcount",
    "graftcount",
]

# APACHE admission-source codes are excluded: patient.py already provides
# interpretable admission-source fields, making these coded columns redundant.
APACHE_BASE_CATEGORICAL_COLS: List[str] = []

HX_FEATURES = [
    "hx_hypertension",
    "hx_cancer",
    "hx_copd",
    "hx_chf",
    "hx_renal_disease",
    "hx_diabetes",
    "hx_atrial_fibrillation",
    "hx_hypothyroidism",
    "hx_mi",
    "hx_stroke",
    "hx_asthma",
    "hx_cirrhosis",
    "hx_seizure_disorder",
    "hx_vte",
    "hx_valve_disease",
    "hx_peripheral_vascular_disease",
    "hx_dementia",
    "hx_angina",
    "hx_peptic_ulcer_disease",
    "hx_tia",
    "hx_respiratory_failure",
    "hx_rheumatoid_arthritis",
    "hx_neuromuscular_disease",
    "hx_restrictive_pulmonary_disease",
    "hx_hiv",
    "hx_ventricular_tachycardia",
    "hx_sle",
    "hx_svt",
    "hx_intracranial_mass",
    "hx_hyperthyroidism",
    "hx_sick_sinus_syndrome",
    "hx_neurogenic_bladder",
    "hx_sarcoidosis",
    "hx_ventricular_fibrillation",
    "hx_vasculitis",
    "hx_hypercalcemia",
    "hx_ventricular_ectopy",
    "hx_splenomegaly",
    "hx_scleroderma",
]

MIN_HX_PATIENT_PREVALENCE = 50
MIN_NONVITAL_NONMISSING_FRAC = 0.01


#############################
# Rolling window numeric aggregator
#############################
class WindowAgg:
    def __init__(self, n_rows: int, col_names: List[str]) -> None:
        self.n = int(n_rows)
        self.cols = list(col_names)

        self.count = {c: np.zeros(self.n, dtype=np.int32) for c in self.cols}
        self.sum = {c: np.zeros(self.n, dtype=np.float64) for c in self.cols}
        self.sumsq = {c: np.zeros(self.n, dtype=np.float64) for c in self.cols}
        self.minv = {c: np.full(self.n, np.inf, dtype=np.float64) for c in self.cols}
        self.maxv = {c: np.full(self.n, -np.inf, dtype=np.float64) for c in self.cols}

        self.last_time = {c: np.full(self.n, -1, dtype=np.int32) for c in self.cols}
        self.last_val = {c: np.full(self.n, np.nan, dtype=np.float64) for c in self.cols}

    def update(self, row_idx: np.ndarray, t: np.ndarray, values: Dict[str, np.ndarray]) -> None:
        for c, v in values.items():
            mask = ~np.isnan(v)
            if not np.any(mask):
                continue

            idx = row_idx[mask]
            vv = v[mask]
            tt = t[mask].astype(np.int32)

            self.count[c][idx] += 1
            self.sum[c][idx] += vv
            self.sumsq[c][idx] += vv * vv
            self.minv[c][idx] = np.minimum(self.minv[c][idx], vv)
            self.maxv[c][idx] = np.maximum(self.maxv[c][idx], vv)

            lt = self.last_time[c][idx]
            newer = tt > lt
            if np.any(newer):
                idx2 = idx[newer]
                self.last_time[c][idx2] = tt[newer]
                self.last_val[c][idx2] = vv[newer]

    def finalize(self, prefix: str) -> pd.DataFrame:
        out: Dict[str, np.ndarray] = {}
        for c in self.cols:
            cnt = self.count[c].astype(np.float64)

            mean = np.full(self.n, np.nan, dtype=np.float64)
            std = np.full(self.n, np.nan, dtype=np.float64)
            mn = np.full(self.n, np.nan, dtype=np.float64)
            mx = np.full(self.n, np.nan, dtype=np.float64)
            last = self.last_val[c].copy()

            nonzero = cnt > 0
            if np.any(nonzero):
                mean[nonzero] = self.sum[c][nonzero] / cnt[nonzero]
                var = (self.sumsq[c][nonzero] / cnt[nonzero]) - (mean[nonzero] ** 2)
                var = np.maximum(var, 0.0)
                std[nonzero] = np.sqrt(var)
                mn[nonzero] = self.minv[c][nonzero]
                mx[nonzero] = self.maxv[c][nonzero]

            out[f"{prefix}{c}_min"] = mn.astype(np.float32)
            out[f"{prefix}{c}_max"] = mx.astype(np.float32)
            out[f"{prefix}{c}_mean"] = mean.astype(np.float32)
            out[f"{prefix}{c}_std"] = std.astype(np.float32)
            out[f"{prefix}{c}_count"] = self.count[c].astype(np.float32)
            out[f"{prefix}{c}_last"] = last.astype(np.float32)

        return pd.DataFrame(out)


#############################
# Helpers
#############################
def safe_exists(path: Path) -> bool:
    return path.exists() and path.is_file()


def load_samples(samples_path: Path) -> pd.DataFrame:
    df = pd.read_csv(samples_path)
    need = {"patientunitstayid", "t_end", "label"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"samples file missing columns: {sorted(missing)}")

    df["patientunitstayid"] = pd.to_numeric(df["patientunitstayid"], errors="coerce")
    df["t_end"] = pd.to_numeric(df["t_end"], errors="coerce")
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df.dropna(subset=["patientunitstayid", "t_end", "label"])

    df["patientunitstayid"] = df["patientunitstayid"].astype(np.int64)
    df["t_end"] = df["t_end"].astype(np.int64)
    df["label"] = df["label"].astype(np.int64)
    return df


def make_sample_index_df(samples_df: pd.DataFrame) -> pd.DataFrame:
    idx_df = samples_df[["patientunitstayid", "t_end"]].copy()
    idx_df["row_idx"] = np.arange(len(idx_df), dtype=np.int32)
    return idx_df


def get_history_mins() -> int:
    if hasattr(config, "HISTORY_MINS"):
        return int(config.HISTORY_MINS)
    if hasattr(config, "MAX_HISTORY_MINS"):
        return int(config.MAX_HISTORY_MINS)
    raise AttributeError("config.py must define HISTORY_MINS or MAX_HISTORY_MINS")


def ceil_to_stride(t: np.ndarray, stride: int) -> np.ndarray:
    return ((t + stride - 1) // stride) * stride


def build_event_to_window_mapping(
    pids: np.ndarray,
    times: np.ndarray,
    sample_index_df: pd.DataFrame,
    history_mins: int,
    stride_mins: int,
) -> pd.DataFrame:
    pids = pids.astype(np.int64)
    times = times.astype(np.int64)

    t0 = ceil_to_stride(times, stride_mins)

    offs = np.arange(0, history_mins, stride_mins, dtype=np.int64)
    k = len(offs)

    pid_rep = np.repeat(pids, k)
    time_rep = np.repeat(times, k)
    event_i_rep = np.repeat(np.arange(len(times), dtype=np.int32), k)
    t_end_rep = np.repeat(t0, k) + np.tile(offs, len(times))

    in_window = (t_end_rep - history_mins < time_rep) & (time_rep <= t_end_rep)
    if not np.any(in_window):
        return pd.DataFrame(columns=["event_i", "row_idx", "time"])

    map_df = pd.DataFrame({
        "patientunitstayid": pid_rep[in_window].astype(np.int64),
        "t_end": t_end_rep[in_window].astype(np.int64),
        "event_i": event_i_rep[in_window].astype(np.int32),
        "time": time_rep[in_window].astype(np.int64),
    })

    merged = map_df.merge(
        sample_index_df,
        on=["patientunitstayid", "t_end"],
        how="inner",
    )

    return merged[["event_i", "row_idx", "time"]]


def numeric_cols_excluding_offsets(df: pd.DataFrame, pid_col: str, offset_col: str) -> List[str]:
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    out = []
    for c in num_cols:
        cl = c.lower()
        if c == pid_col:
            continue
        if c == offset_col:
            continue
        if "offset" in cl:
            continue
        out.append(c)
    return out


def safe_label_to_col(label: str) -> str:
    s = str(label).strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    if not s:
        s = "unknown"
    return s


def parse_patient_age(x: object) -> float:
    if pd.isna(x):
        return np.nan

    s = str(x).strip()
    if not s:
        return np.nan

    if s.startswith(">"):
        digits = re.findall(r"\d+", s)
        if digits:
            return float(digits[0])
        return np.nan

    try:
        return float(s)
    except Exception:
        digits = re.findall(r"\d+", s)
        if digits:
            return float(digits[0])
        return np.nan


def dedupe_by_patient(df: pd.DataFrame) -> pd.DataFrame:
    if "patientunitstayid" not in df.columns:
        return df
    return df.drop_duplicates(subset=["patientunitstayid"], keep="last").copy()


#############################
# pastHistory normalization + mapping
#############################
def clean_pasthistory_segment(text: str) -> str:
    text = str(text).lower().strip()
    text = text.split("-", 1)[0]
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_pasthistory_condition(path: str) -> str:
    if pd.isna(path):
        return ""

    raw_parts = [p.strip() for p in str(path).split("/") if str(p).strip()]
    lowered = [p.lower().strip() for p in raw_parts]

    try:
        idx = lowered.index("organ systems")
        remainder = raw_parts[idx + 2 :]
    except ValueError:
        remainder = raw_parts

    cleaned = [clean_pasthistory_segment(x) for x in remainder]
    cleaned = [x for x in cleaned if x]

    return " / ".join(cleaned)


def normalize_pasthistory_for_match(text: str) -> str:
    s = str(text).lower().strip()
    s = s.replace("tia s", "tia")
    s = s.replace("non insulin dependent", "noninsulin dependent")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def map_pasthistory_to_hx(cond: str) -> Optional[str]:
    s = normalize_pasthistory_for_match(cond)

    if "hypertension requiring treatment" in s or re.search(r"\bhypertension\b", s):
        return "hx_hypertension"

    if any(x in s for x in [
        "oncology",
        "cancer",
        "malignancy",
        "carcinoma",
        "leukemia",
        "leukaemia",
        "lymphoma",
        "metastases",
        "metastatic",
        "myeloma",
        "neoplasm",
        "tumor",
        "tumour",
    ]):
        return "hx_cancer"

    if "copd" in s:
        return "hx_copd"

    if "congestive heart failure" in s or re.search(r"\bchf\b", s):
        return "hx_chf"

    if any(x in s for x in [
        "renal failure",
        "renal insufficiency",
        "renal tubular acidosis",
    ]):
        return "hx_renal_disease"

    if (
        "insulin dependent diabetes" in s
        or "noninsulin dependent diabetes" in s
        or "medication dependent" in s
        or re.search(r"\bdiabetes\b", s)
    ):
        return "hx_diabetes"

    if "atrial fibrillation" in s:
        return "hx_atrial_fibrillation"

    if "hypothyroidism" in s:
        return "hx_hypothyroidism"

    if "myocardial infarction" in s or re.search(r"\bmi\b", s):
        return "hx_mi"

    if "stroke" in s or "strokes" in s:
        return "hx_stroke"

    if "asthma" in s:
        return "hx_asthma"

    if "cirrhosis" in s:
        return "hx_cirrhosis"

    if any(x in s for x in [
        "seizure",
        "seizures",
        "epilepsy",
        "status epilepticus",
        "convulsion",
    ]):
        return "hx_seizure_disorder"

    if any(x in s for x in [
        "venous thrombosis",
        "deep vein thrombosis",
        "pulmonary embolism",
    ]) or re.search(r"\bdvt\b", s):
        return "hx_vte"

    if any(x in s for x in [
        "valve disease",
        "valvular disease",
    ]):
        return "hx_valve_disease"

    if re.search(r"(^| / )(as|mr|ms|ar|tr)( / |$)", s):
        return "hx_valve_disease"
    if "p avr" in s or "p mvr" in s:
        return "hx_valve_disease"

    if "peripheral vascular disease" in s:
        return "hx_peripheral_vascular_disease"

    if "dementia" in s:
        return "hx_dementia"

    if "angina" in s:
        return "hx_angina"

    if "peptic ulcer disease" in s or "peptic ulcer" in s:
        return "hx_peptic_ulcer_disease"

    if re.search(r"\btia\b", s) or "transient ischemic attack" in s:
        return "hx_tia"

    if "respiratory failure" in s:
        return "hx_respiratory_failure"

    if "rheumatoid arthritis" in s:
        return "hx_rheumatoid_arthritis"

    if "neuromuscular disease" in s:
        return "hx_neuromuscular_disease"

    if "restrictive pulmonary disease" in s or "restrictive disease" in s:
        return "hx_restrictive_pulmonary_disease"

    if "hiv positive" in s or re.search(r"\bhiv\b", s) or re.search(r"\baids\b", s):
        return "hx_hiv"

    if "ventricular tachycardia" in s:
        return "hx_ventricular_tachycardia"

    if re.search(r"\bsle\b", s) or "systemic lupus" in s:
        return "hx_sle"

    if re.search(r"\bsvt\b", s) or "supraventricular tachycardia" in s:
        return "hx_svt"

    if "intracranial mass" in s:
        return "hx_intracranial_mass"

    if "hyperthyroidism" in s:
        return "hx_hyperthyroidism"

    if "sick sinus syndrome" in s:
        return "hx_sick_sinus_syndrome"

    if "neurogenic bladder" in s:
        return "hx_neurogenic_bladder"

    if "sarcoidosis" in s:
        return "hx_sarcoidosis"

    if "ventricular fibrillation" in s:
        return "hx_ventricular_fibrillation"

    if "vasculitis" in s:
        return "hx_vasculitis"

    if "hypercalcemia" in s:
        return "hx_hypercalcemia"

    if "ventricular ectopy" in s:
        return "hx_ventricular_ectopy"

    if "splenomegaly" in s:
        return "hx_splenomegaly"

    if "scleroderma" in s:
        return "hx_scleroderma"

    return None


#############################
# Missingness masks (vitals only)
#############################
def add_vitals_missingness_masks(feats: pd.DataFrame) -> pd.DataFrame:
    """
    Memory-safe missingness features for vitals using *_count columns.

    For each vp_/va_ signal that has:
        <prefix><signal>_count
    we define missing_base = (count == 0) and then create:
        <prefix><signal>_min_missing
        <prefix><signal>_max_missing
        <prefix><signal>_mean_missing
        <prefix><signal>_std_missing
        <prefix><signal>_last_missing

    Build all new columns first, then concat once, to avoid dataframe fragmentation.
    """
    stat_suffixes = ["min", "max", "mean", "std", "last"]

    count_cols = [
        c for c in feats.columns
        if (str(c).startswith("vp_") or str(c).startswith("va_")) and str(c).endswith("_count")
    ]
    if not count_cols:
        return feats

    new_cols: Dict[str, np.ndarray] = {}

    for c_count in count_cols:
        base = str(c_count)[:-len("_count")]
        cnt = feats[c_count].to_numpy(copy=False)
        missing_base = (np.nan_to_num(cnt, nan=0.0) <= 0.0).astype(np.float32)

        for suf in stat_suffixes:
            stat_col = f"{base}_{suf}"
            miss_col = f"{base}_{suf}_missing"
            if stat_col in feats.columns and miss_col not in feats.columns:
                new_cols[miss_col] = missing_base

    if not new_cols:
        return feats

    miss_df = pd.DataFrame(new_cols, index=feats.index)
    feats = pd.concat([feats, miss_df], axis=1, copy=False)
    return feats


#############################
# Baseline features
#############################
def build_patient_baseline_patient_table(data_dir: Path, pids_set: set[int]) -> pd.DataFrame:
    patient_path = data_dir / "patient.csv.gz"
    if not safe_exists(patient_path):
        return pd.DataFrame({"patientunitstayid": sorted(pids_set)})

    usecols = ["patientunitstayid"] + PATIENT_BASE_NUMERIC_COLS + PATIENT_BASE_CATEGORICAL_COLS
    patient = pd.read_csv(patient_path, compression="infer", low_memory=False, usecols=usecols)
    patient["patientunitstayid"] = pd.to_numeric(patient["patientunitstayid"], errors="coerce")
    patient = patient.dropna(subset=["patientunitstayid"])
    patient["patientunitstayid"] = patient["patientunitstayid"].astype(np.int64)
    patient = patient[patient["patientunitstayid"].isin(pids_set)].copy()
    patient = dedupe_by_patient(patient)

    if "age" in patient.columns:
        patient["age"] = patient["age"].map(parse_patient_age)

    for c in PATIENT_BASE_NUMERIC_COLS:
        if c in patient.columns:
            patient[c] = pd.to_numeric(patient[c], errors="coerce")

    for c in PATIENT_BASE_CATEGORICAL_COLS:
        if c in patient.columns:
            patient[c] = patient[c].astype(str).replace({"nan": np.nan})

    keep_cols = ["patientunitstayid"] + [c for c in PATIENT_BASE_NUMERIC_COLS + PATIENT_BASE_CATEGORICAL_COLS if c in patient.columns]
    patient = patient[keep_cols].copy()

    numeric_cols = [c for c in PATIENT_BASE_NUMERIC_COLS if c in patient.columns]
    cat_cols = [c for c in PATIENT_BASE_CATEGORICAL_COLS if c in patient.columns]

    if cat_cols:
        patient = pd.get_dummies(
            patient,
            columns=cat_cols,
            prefix=[f"pt_{c}" for c in cat_cols],
            dummy_na=True,
        )

    rename_map = {c: f"pt_{c}" for c in numeric_cols}
    patient = patient.rename(columns=rename_map)

    return patient


def build_patient_baseline_apache_table(data_dir: Path, pids_set: set[int]) -> pd.DataFrame:
    apv_path = data_dir / "apachePredVar.csv.gz"
    if not safe_exists(apv_path):
        return pd.DataFrame({"patientunitstayid": sorted(pids_set)})

    usecols = ["patientunitstayid"] + APACHE_BASE_KEEP_COLS
    apv = pd.read_csv(apv_path, compression="infer", low_memory=False, usecols=usecols)
    apv["patientunitstayid"] = pd.to_numeric(apv["patientunitstayid"], errors="coerce")
    apv = apv.dropna(subset=["patientunitstayid"])
    apv["patientunitstayid"] = apv["patientunitstayid"].astype(np.int64)
    apv = apv[apv["patientunitstayid"].isin(pids_set)].copy()
    apv = dedupe_by_patient(apv)

    cat_cols = [c for c in APACHE_BASE_CATEGORICAL_COLS if c in apv.columns]
    num_cols = [c for c in APACHE_BASE_KEEP_COLS if c in apv.columns and c not in cat_cols]

    for c in num_cols:
        apv[c] = pd.to_numeric(apv[c], errors="coerce")

    for c in cat_cols:
        apv[c] = apv[c].astype(str).replace({"nan": np.nan})

    apv = apv[["patientunitstayid"] + num_cols + cat_cols].copy()

    if cat_cols:
        apv = pd.get_dummies(
            apv,
            columns=cat_cols,
            prefix=[f"apv_{c}" for c in cat_cols],
            dummy_na=True,
        )

    rename_map = {c: f"apv_{c}" for c in num_cols}
    apv = apv.rename(columns=rename_map)

    return apv


def build_patient_baseline_pasthistory_table(
    data_dir: Path,
    pids_set: set[int],
    min_prevalence: int,
    chunksize: int = 400_000,
) -> pd.DataFrame:
    ph_path = data_dir / "pastHistory.csv.gz"
    patient_ids = pd.DataFrame({"patientunitstayid": sorted(pids_set)})

    if not safe_exists(ph_path):
        hx = patient_ids.copy()
        for c in HX_FEATURES:
            hx[c] = 0.0
        return hx

    mapped_parts: List[pd.DataFrame] = []

    reader = pd.read_csv(
        ph_path,
        compression="infer",
        low_memory=False,
        chunksize=chunksize,
        usecols=["patientunitstayid", "pasthistorypath"],
    )

    total_mapped_rows = 0
    for chunk in reader:
        chunk["patientunitstayid"] = pd.to_numeric(chunk["patientunitstayid"], errors="coerce")
        chunk = chunk.dropna(subset=["patientunitstayid", "pasthistorypath"])
        if chunk.empty:
            continue

        chunk["patientunitstayid"] = chunk["patientunitstayid"].astype(np.int64)
        chunk = chunk[chunk["patientunitstayid"].isin(pids_set)].copy()
        if chunk.empty:
            continue

        chunk["__cond__"] = chunk["pasthistorypath"].map(extract_pasthistory_condition)
        chunk["hx_feature"] = chunk["__cond__"].map(map_pasthistory_to_hx)
        chunk = chunk.dropna(subset=["hx_feature"])
        if chunk.empty:
            continue

        part = chunk[["patientunitstayid", "hx_feature"]].copy()
        part["value"] = 1
        mapped_parts.append(part)
        total_mapped_rows += len(part)

    print("#############################")
    print("Building pastHistory baseline features")
    print("#############################")
    print(f"Mapped pastHistory rows: {total_mapped_rows}")
    print("#############################")

    if mapped_parts:
        mapped = pd.concat(mapped_parts, axis=0, ignore_index=True)
        mapped = mapped.drop_duplicates(subset=["patientunitstayid", "hx_feature"])

        hx = (
            mapped.pivot_table(
                index="patientunitstayid",
                columns="hx_feature",
                values="value",
                aggfunc="max",
                fill_value=0,
            )
            .reset_index()
        )
    else:
        hx = pd.DataFrame({"patientunitstayid": sorted(pids_set)})

    for c in HX_FEATURES:
        if c not in hx.columns:
            hx[c] = 0

    hx = hx[["patientunitstayid"] + HX_FEATURES].copy()

    hx_counts = hx[HX_FEATURES].sum(axis=0)
    keep_hx_cols = [c for c in HX_FEATURES if int(hx_counts.get(c, 0)) >= int(min_prevalence)]
    drop_hx_cols = [c for c in HX_FEATURES if c not in keep_hx_cols]

    print("#############################")
    print("pastHistory hx prevalence in modeling cohort")
    print("#############################")
    print(hx_counts.sort_values(ascending=False).to_string())
    print("#############################")
    print(f"Keeping hx columns (>= {min_prevalence} patients): {len(keep_hx_cols)}")
    if drop_hx_cols:
        print(f"Dropping rare hx columns (< {min_prevalence} patients): {drop_hx_cols}")
    print("#############################")

    hx = hx[["patientunitstayid"] + keep_hx_cols].copy()

    for c in keep_hx_cols:
        hx[c] = pd.to_numeric(hx[c], errors="coerce").fillna(0).astype(np.float32)

    return hx


def build_baseline_features(
    data_dir: Path,
    samples_df: pd.DataFrame,
    min_hx_prevalence: int = MIN_HX_PATIENT_PREVALENCE,
) -> pd.DataFrame:
    pids_set = set(samples_df["patientunitstayid"].unique().tolist())
    patient_ids = pd.DataFrame({"patientunitstayid": sorted(pids_set)})

    pt = build_patient_baseline_patient_table(data_dir, pids_set)
    apv = build_patient_baseline_apache_table(data_dir, pids_set)
    hx = build_patient_baseline_pasthistory_table(data_dir, pids_set, min_hx_prevalence)

    baseline_patient = patient_ids.merge(pt, on="patientunitstayid", how="left")
    baseline_patient = baseline_patient.merge(apv, on="patientunitstayid", how="left")
    baseline_patient = baseline_patient.merge(hx, on="patientunitstayid", how="left")

    baseline = samples_df[["patientunitstayid", "t_end"]].merge(
        baseline_patient,
        on="patientunitstayid",
        how="left",
    )

    for c in baseline.columns:
        if c in ["patientunitstayid", "t_end"]:
            continue
        if pd.api.types.is_numeric_dtype(baseline[c]):
            baseline[c] = baseline[c].astype(np.float32, copy=False)

    return baseline


#############################
# Numeric time series aggregation
#############################
def process_numeric_timeseries_table(
    table_path: Path,
    pid_col: str,
    offset_col: str,
    usecols: List[str],
    drop_cols: Optional[List[str]],
    prefix: str,
    sample_index_df: pd.DataFrame,
    n_samples: int,
    history_mins: int,
    stride_mins: int,
    chunksize: int = 1_000_000,
) -> pd.DataFrame:
    print("#############################")
    print(f"Processing table: {table_path.name}")
    print(f"Offset col: {offset_col}")
    print(f"Prefix: {prefix}")
    print("#############################")

    if not safe_exists(table_path):
        print(f"WARNING: missing file, skipping: {table_path}")
        return pd.DataFrame(index=np.arange(n_samples))

    if drop_cols is None:
        drop_cols = []

    cols = list(dict.fromkeys([pid_col, offset_col] + usecols))
    reader = pd.read_csv(
        table_path,
        compression="infer",
        low_memory=False,
        chunksize=chunksize,
        usecols=cols,
    )

    agg: Optional[WindowAgg] = None
    numeric_cols: Optional[List[str]] = None
    mapped_total = 0

    for chunk in reader:
        if pid_col not in chunk.columns or offset_col not in chunk.columns:
            continue

        chunk = chunk.drop(columns=drop_cols, errors="ignore")

        chunk[pid_col] = pd.to_numeric(chunk[pid_col], errors="coerce")
        chunk[offset_col] = pd.to_numeric(chunk[offset_col], errors="coerce")
        chunk = chunk.dropna(subset=[pid_col, offset_col])
        if chunk.empty:
            continue

        chunk[pid_col] = chunk[pid_col].astype(np.int64)
        chunk[offset_col] = chunk[offset_col].astype(np.int64)

        for c in chunk.columns:
            if c in [pid_col, offset_col]:
                continue
            chunk[c] = pd.to_numeric(chunk[c], errors="coerce")

        if numeric_cols is None:
            numeric_cols = numeric_cols_excluding_offsets(chunk, pid_col, offset_col)
            if not numeric_cols:
                print("No numeric signal columns found (after offset filtering). Skipping.")
                return pd.DataFrame(index=np.arange(n_samples))
            agg = WindowAgg(n_rows=n_samples, col_names=numeric_cols)

        assert agg is not None
        assert numeric_cols is not None

        pids = chunk[pid_col].to_numpy(dtype=np.int64)
        times = chunk[offset_col].to_numpy(dtype=np.int64)

        map_df = build_event_to_window_mapping(
            pids=pids,
            times=times,
            sample_index_df=sample_index_df,
            history_mins=history_mins,
            stride_mins=stride_mins,
        )
        if map_df.empty:
            continue

        mapped_total += len(map_df)

        event_i = map_df["event_i"].to_numpy(dtype=np.int32)
        row_idx = map_df["row_idx"].to_numpy(dtype=np.int32)
        t = map_df["time"].to_numpy(dtype=np.int64)

        values: Dict[str, np.ndarray] = {}
        for c in numeric_cols:
            arr = chunk[c].to_numpy(dtype=np.float64)
            values[c] = arr[event_i]

        agg.update(row_idx=row_idx, t=t, values=values)

    if agg is None:
        return pd.DataFrame(index=np.arange(n_samples))

    print(f"Mapped event->window pairs: {mapped_total}")
    return agg.finalize(prefix=prefix)


#############################
# Charting pivots (top-N numeric labels)
#############################
def find_top_numeric_labels(
    table_path: Path,
    label_col: str,
    value_col: str,
    top_n: int,
    chunksize: int = 800_000,
) -> List[str]:
    if not safe_exists(table_path):
        return []

    counts: Dict[str, int] = {}

    reader = pd.read_csv(
        table_path,
        compression="infer",
        low_memory=False,
        chunksize=chunksize,
        usecols=[label_col, value_col],
    )

    for chunk in reader:
        vals = pd.to_numeric(chunk[value_col], errors="coerce")
        ok = vals.notna()
        if not ok.any():
            continue
        labels = chunk.loc[ok, label_col].astype(str)
        vc = labels.value_counts()
        for k, v in vc.items():
            counts[k] = counts.get(k, 0) + int(v)

    if not counts:
        return []

    sorted_labels = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    return [lbl for lbl, _ in sorted_labels[:top_n]]


def _pivot_feature_names(prefix: str, slug: str) -> List[str]:
    return [
        f"{prefix}{slug}_min",
        f"{prefix}{slug}_max",
        f"{prefix}{slug}_mean",
        f"{prefix}{slug}_std",
        f"{prefix}{slug}_count",
        f"{prefix}{slug}_last",
    ]


def process_charting_pivot(
    table_path: Path,
    pid_col: str,
    offset_col: str,
    label_col: str,
    value_col: str,
    prefix: str,
    sample_index_df: pd.DataFrame,
    n_samples: int,
    history_mins: int,
    stride_mins: int,
    top_labels: List[str],
    chunksize: int = 800_000,
) -> pd.DataFrame:
    print("#############################")
    print(f"Processing {table_path.name} (pivot by label, memory safe)")
    print(f"Top labels: {len(top_labels)}")
    print("#############################")

    if not top_labels or not safe_exists(table_path):
        return pd.DataFrame(index=np.arange(n_samples))

    label_to_agg: Dict[str, WindowAgg] = {}
    label_to_slug: Dict[str, str] = {}
    used_slugs: Dict[str, int] = {}

    for lbl in top_labels:
        base = safe_label_to_col(lbl)
        if base not in used_slugs:
            used_slugs[base] = 1
            slug = base
        else:
            used_slugs[base] += 1
            slug = f"{base}_{used_slugs[base]}"

        label_to_slug[lbl] = slug
        label_to_agg[lbl] = WindowAgg(n_rows=n_samples, col_names=["value"])

    reader = pd.read_csv(
        table_path,
        compression="infer",
        low_memory=False,
        chunksize=chunksize,
        usecols=[pid_col, offset_col, label_col, value_col],
    )

    top_set = set(top_labels)
    mapped_total = 0

    for chunk in reader:
        chunk[label_col] = chunk[label_col].astype(str)
        chunk = chunk[chunk[label_col].isin(top_set)]
        if chunk.empty:
            continue

        chunk[pid_col] = pd.to_numeric(chunk[pid_col], errors="coerce")
        chunk[offset_col] = pd.to_numeric(chunk[offset_col], errors="coerce")
        chunk["__value_num__"] = pd.to_numeric(chunk[value_col], errors="coerce")

        chunk = chunk.dropna(subset=[pid_col, offset_col, "__value_num__"])
        if chunk.empty:
            continue

        chunk[pid_col] = chunk[pid_col].astype(np.int64)
        chunk[offset_col] = chunk[offset_col].astype(np.int64)
        chunk["__value_num__"] = chunk["__value_num__"].astype(np.float64)

        for lbl, sub in chunk.groupby(label_col, sort=False):
            pids = sub[pid_col].to_numpy(dtype=np.int64)
            times = sub[offset_col].to_numpy(dtype=np.int64)

            map_df = build_event_to_window_mapping(
                pids=pids,
                times=times,
                sample_index_df=sample_index_df,
                history_mins=history_mins,
                stride_mins=stride_mins,
            )
            if map_df.empty:
                continue

            mapped_total += len(map_df)

            event_i = map_df["event_i"].to_numpy(dtype=np.int32)
            row_idx = map_df["row_idx"].to_numpy(dtype=np.int32)
            t = map_df["time"].to_numpy(dtype=np.int64)

            vals = sub["__value_num__"].to_numpy(dtype=np.float64)
            values = {"value": vals[event_i]}

            label_to_agg[lbl].update(row_idx=row_idx, t=t, values=values)

    print(f"{table_path.name} mapped event->window pairs: {mapped_total}")

    slugs = [label_to_slug[lbl] for lbl in top_labels]
    col_names: List[str] = []
    for slug in slugs:
        col_names.extend(_pivot_feature_names(prefix=prefix, slug=slug))

    out_mat = np.zeros((n_samples, len(col_names)), dtype=np.float32)

    col_ptr = 0
    for lbl in top_labels:
        slug = label_to_slug[lbl]
        df_lbl = label_to_agg[lbl].finalize(prefix=f"{prefix}{slug}_")
        df_lbl = df_lbl.rename(columns=lambda c: c.replace(f"{prefix}{slug}_value_", f"{prefix}{slug}_"))

        want_cols = _pivot_feature_names(prefix=prefix, slug=slug)
        for c in want_cols:
            if c in df_lbl.columns:
                out_mat[:, col_ptr] = df_lbl[c].to_numpy(dtype=np.float32, copy=False)
            else:
                out_mat[:, col_ptr] = 0.0
            col_ptr += 1

    out = pd.DataFrame(out_mat, columns=col_names)
    return out


#############################
# Count/flag features
#############################
def process_count_table(
    table_path: Path,
    pid_col: str,
    offset_col: str,
    sample_index_df: pd.DataFrame,
    n_samples: int,
    history_mins: int,
    stride_mins: int,
    chunksize: int,
    out_col: str,
) -> pd.DataFrame:
    if not safe_exists(table_path):
        print(f"WARNING: missing file, skipping: {table_path}")
        return pd.DataFrame({out_col: np.zeros(n_samples, dtype=np.float32)})

    cnt = np.zeros(n_samples, dtype=np.int32)

    reader = pd.read_csv(
        table_path,
        compression="infer",
        low_memory=False,
        chunksize=chunksize,
        usecols=[pid_col, offset_col],
    )

    mapped_total = 0
    for chunk in reader:
        chunk[pid_col] = pd.to_numeric(chunk[pid_col], errors="coerce")
        chunk[offset_col] = pd.to_numeric(chunk[offset_col], errors="coerce")
        chunk = chunk.dropna(subset=[pid_col, offset_col])
        if chunk.empty:
            continue

        pids = chunk[pid_col].astype(np.int64).to_numpy()
        times = chunk[offset_col].astype(np.int64).to_numpy()

        map_df = build_event_to_window_mapping(
            pids=pids,
            times=times,
            sample_index_df=sample_index_df,
            history_mins=history_mins,
            stride_mins=stride_mins,
        )
        if map_df.empty:
            continue

        mapped_total += len(map_df)
        row_idx = map_df["row_idx"].to_numpy(dtype=np.int32)
        np.add.at(cnt, row_idx, 1)

    print(f"{out_col}: mapped pairs {mapped_total}")
    return pd.DataFrame({out_col: cnt.astype(np.float32)})


def process_treatment_dialysis(
    table_path: Path,
    sample_index_df: pd.DataFrame,
    n_samples: int,
    history_mins: int,
    stride_mins: int,
    chunksize: int,
) -> pd.DataFrame:
    if not safe_exists(table_path):
        print(f"WARNING: missing file, skipping: {table_path}")
        return pd.DataFrame({
            "treatment_count_in_window": np.zeros(n_samples, dtype=np.float32),
            "treatment_dialysis_any": np.zeros(n_samples, dtype=np.float32),
        })

    tr_cnt = np.zeros(n_samples, dtype=np.int32)
    dial_any = np.zeros(n_samples, dtype=np.int32)

    reader = pd.read_csv(
        table_path,
        compression="infer",
        low_memory=False,
        chunksize=chunksize,
        usecols=["patientunitstayid", "treatmentoffset", "treatmentstring"],
    )

    mapped_total = 0
    for chunk in reader:
        chunk["patientunitstayid"] = pd.to_numeric(chunk["patientunitstayid"], errors="coerce")
        chunk["treatmentoffset"] = pd.to_numeric(chunk["treatmentoffset"], errors="coerce")
        chunk = chunk.dropna(subset=["patientunitstayid", "treatmentoffset"])
        if chunk.empty:
            continue

        pids = chunk["patientunitstayid"].astype(np.int64).to_numpy()
        times = chunk["treatmentoffset"].astype(np.int64).to_numpy()

        map_df = build_event_to_window_mapping(
            pids=pids,
            times=times,
            sample_index_df=sample_index_df,
            history_mins=history_mins,
            stride_mins=stride_mins,
        )
        if map_df.empty:
            continue

        mapped_total += len(map_df)
        event_i = map_df["event_i"].to_numpy(dtype=np.int32)
        row_idx = map_df["row_idx"].to_numpy(dtype=np.int32)

        np.add.at(tr_cnt, row_idx, 1)

        dialysis_mask = chunk["treatmentstring"].astype(str).str.contains("dialysis", case=False, na=False).to_numpy()
        is_dial = dialysis_mask[event_i]
        if np.any(is_dial):
            dial_any[row_idx[is_dial]] = 1

    print(f"treatment mapped pairs {mapped_total}")
    return pd.DataFrame({
        "treatment_count_in_window": tr_cnt.astype(np.float32),
        "treatment_dialysis_any": dial_any.astype(np.float32),
    })


def process_infusion_vasopressor_any(
    table_path: Path,
    sample_index_df: pd.DataFrame,
    n_samples: int,
    history_mins: int,
    stride_mins: int,
    chunksize: int,
) -> pd.DataFrame:
    if not safe_exists(table_path):
        print(f"WARNING: missing file, skipping: {table_path}")
        return pd.DataFrame({"drug_vasopressor_any": np.zeros(n_samples, dtype=np.float32)})

    vaso_any = np.zeros(n_samples, dtype=np.int32)
    vaso_pattern = "norepi|norad|dopamine|epinephrine|adrenaline|phenylephrine|vasopressin|levophed"

    reader = pd.read_csv(
        table_path,
        compression="infer",
        low_memory=False,
        chunksize=chunksize,
        usecols=["patientunitstayid", "infusionoffset", "drugname"],
    )

    mapped_total = 0
    for chunk in reader:
        chunk["patientunitstayid"] = pd.to_numeric(chunk["patientunitstayid"], errors="coerce")
        chunk["infusionoffset"] = pd.to_numeric(chunk["infusionoffset"], errors="coerce")
        chunk = chunk.dropna(subset=["patientunitstayid", "infusionoffset"])
        if chunk.empty:
            continue

        is_vaso = chunk["drugname"].astype(str).str.contains(vaso_pattern, case=False, na=False).to_numpy()
        if not np.any(is_vaso):
            continue

        chunk = chunk.loc[is_vaso].reset_index(drop=True)

        pids = chunk["patientunitstayid"].astype(np.int64).to_numpy()
        times = chunk["infusionoffset"].astype(np.int64).to_numpy()

        map_df = build_event_to_window_mapping(
            pids=pids,
            times=times,
            sample_index_df=sample_index_df,
            history_mins=history_mins,
            stride_mins=stride_mins,
        )
        if map_df.empty:
            continue

        mapped_total += len(map_df)
        row_idx = map_df["row_idx"].to_numpy(dtype=np.int32)
        vaso_any[row_idx] = 1

    print(f"vasopressor mapped pairs {mapped_total}")
    return pd.DataFrame({"drug_vasopressor_any": vaso_any.astype(np.float32)})


#############################
# Save helpers
#############################
def cast_numeric_feature_cols(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.columns:
        if c in ["patientunitstayid", "t_end"]:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            df[c] = df[c].astype(np.float32, copy=False)
    return df


def ensure_no_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    if not df.columns.duplicated().any():
        return df

    new_cols = []
    seen: Dict[str, int] = {}
    for c in df.columns:
        if c not in seen:
            seen[c] = 1
            new_cols.append(c)
        else:
            seen[c] += 1
            new_cols.append(f"{c}__dup{seen[c]}")
    df.columns = new_cols
    return df


def drop_sparse_nonvital_features(
    feats: pd.DataFrame,
    min_non_missing_frac: float = MIN_NONVITAL_NONMISSING_FRAC,
) -> pd.DataFrame:
    """
    Drop sparse non-vital feature columns.

    Applies only to wide non-vital dynamic blocks:
      rch_, nch_, rc_, inf_, io_, lab_

    Rule:
      - for *_count columns: keep if at least min_non_missing_frac of rows are > 0
      - otherwise: keep if at least min_non_missing_frac of rows are non-missing

    Vitals and baseline features are untouched.
    """
    target_prefixes = ("rch_", "nch_", "rc_", "inf_", "io_", "lab_")

    n = len(feats)
    if n == 0:
        return feats

    drop_cols: List[str] = []

    for c in feats.columns:
        if c in ["patientunitstayid", "t_end"]:
            continue
        if not c.startswith(target_prefixes):
            continue

        arr = feats[c].to_numpy(copy=False)

        if c.endswith("_count"):
            support_frac = (np.nan_to_num(arr, nan=0.0) > 0).mean()
        else:
            support_frac = pd.Series(arr).notna().mean()

        if support_frac < min_non_missing_frac:
            drop_cols.append(c)

    print("#############################")
    print("Sparse non-vital feature pruning")
    print("#############################")
    print(f"Threshold: {min_non_missing_frac:.1%}")
    print(f"Columns dropped: {len(drop_cols)}")
    if drop_cols:
        print(f"First 50 dropped: {drop_cols[:50]}")
    print("#############################")

    if drop_cols:
        feats = feats.drop(columns=drop_cols)

    return feats


#############################
# Main
#############################
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    data_dir = config.EICU_DATA_DIR
    samples_csv = args.samples_path or config.samples_path(config.DISEASE)

    output_dir = args.output_dir or config.run_dir(config.DISEASE)
    output_dir.mkdir(parents=True, exist_ok=True)
    disease_tag = config.disease_tag(config.DISEASE)
    out_features_full = output_dir / f"features__{disease_tag}.parquet"
    out_features_vitals = output_dir / f"features_vitals__{disease_tag}.parquet"
    out_features_baseline = output_dir / f"features_baseline__{disease_tag}.parquet"

    print("#############################")
    print("Aggregating rolling-window features (memory safe)")
    print("#############################")
    print(f"Data dir: {data_dir}")
    print(f"Samples: {samples_csv}")
    print(f"Output full features: {out_features_full}")
    print(f"Output vitals features: {out_features_vitals}")
    print(f"Output baseline features: {out_features_baseline}")
    print("#############################")

    samples_df = load_samples(samples_csv)
    sample_index_df = make_sample_index_df(samples_df)

    n_samples = len(samples_df)
    history_mins = get_history_mins()
    stride_mins = int(config.STRIDE_MINS)

    blocks: List[pd.DataFrame] = []

    base = pd.DataFrame({
        "patientunitstayid": samples_df["patientunitstayid"].values.astype(np.int64),
        "t_end": samples_df["t_end"].values.astype(np.int64),
    })
    blocks.append(base)

    baseline_feats = build_baseline_features(
        data_dir=data_dir,
        samples_df=samples_df,
        min_hx_prevalence=MIN_HX_PATIENT_PREVALENCE,
    )
    blocks.append(baseline_feats.drop(columns=["patientunitstayid", "t_end"], errors="ignore"))

    vp_block = process_numeric_timeseries_table(
        table_path=data_dir / "vitalPeriodic.csv.gz",
        pid_col="patientunitstayid",
        offset_col="observationoffset",
        usecols=[
            "temperature", "sao2", "heartrate", "respiration",
            "cvp", "etco2",
            "systemicsystolic", "systemicdiastolic", "systemicmean",
            "pasystolic", "padiastolic", "pamean",
            "st1", "st2", "st3", "icp",
        ],
        drop_cols=["vitalperiodicid"],
        prefix="vp_",
        sample_index_df=sample_index_df,
        n_samples=n_samples,
        history_mins=history_mins,
        stride_mins=stride_mins,
        chunksize=1_000_000,
    )
    blocks.append(vp_block)

    va_block = process_numeric_timeseries_table(
        table_path=data_dir / "vitalAperiodic.csv.gz",
        pid_col="patientunitstayid",
        offset_col="observationoffset",
        usecols=[
            "noninvasivesystolic", "noninvasivediastolic", "noninvasivemean",
            "paop", "cardiacoutput", "cardiacinput", "svr", "svri", "pvr", "pvri",
        ],
        drop_cols=["vitalaperiodicid"],
        prefix="va_",
        sample_index_df=sample_index_df,
        n_samples=n_samples,
        history_mins=history_mins,
        stride_mins=stride_mins,
        chunksize=1_000_000,
    )
    blocks.append(va_block)

    blocks.append(process_numeric_timeseries_table(
        table_path=data_dir / "lab.csv.gz",
        pid_col="patientunitstayid",
        offset_col="labresultoffset",
        usecols=["labresult"],
        drop_cols=["labid"],
        prefix="lab_",
        sample_index_df=sample_index_df,
        n_samples=n_samples,
        history_mins=history_mins,
        stride_mins=stride_mins,
        chunksize=800_000,
    ))

    blocks.append(process_numeric_timeseries_table(
        table_path=data_dir / "intakeOutput.csv.gz",
        pid_col="patientunitstayid",
        offset_col="intakeoutputoffset",
        usecols=["intaketotal", "outputtotal", "dialysistotal", "nettotal", "cellvaluenumeric"],
        drop_cols=["intakeoutputid", "cellpath", "celllabel", "cellvaluetext", "intakeoutputentryoffset"],
        prefix="io_",
        sample_index_df=sample_index_df,
        n_samples=n_samples,
        history_mins=history_mins,
        stride_mins=stride_mins,
        chunksize=800_000,
    ))

    blocks.append(process_numeric_timeseries_table(
        table_path=data_dir / "respiratoryCare.csv.gz",
        pid_col="patientunitstayid",
        offset_col="respcarestatusoffset",
        usecols=[
            "airwaytype", "airwaysize", "airwayposition", "cuffpressure",
            "apneaparms", "lowexhmvlimit", "hiexhmvlimit", "lowexhtvlimit",
            "hipeakpreslimit", "lowpeakpreslimit", "hirespratelimit", "lowrespratelimit",
            "sighpreslimit", "lowironoxlimit", "highironoxlimit", "meanairwaypreslimit",
            "peeplimit", "cpaplimit", "setapneainterval", "setapneatv",
            "setapneaippeephigh", "setapnearr", "setapneapeakflow",
            "setapneainsptime", "setapneaie", "setapneafio2",
        ],
        drop_cols=["respcareid", "currenthistoryseqnum"],
        prefix="rc_",
        sample_index_df=sample_index_df,
        n_samples=n_samples,
        history_mins=history_mins,
        stride_mins=stride_mins,
        chunksize=800_000,
    ))

    blocks.append(process_numeric_timeseries_table(
        table_path=data_dir / "infusionDrug.csv.gz",
        pid_col="patientunitstayid",
        offset_col="infusionoffset",
        usecols=["drugrate", "infusionrate", "drugamount", "volumeoffluid", "patientweight"],
        drop_cols=["infusiondrugid"],
        prefix="inf_",
        sample_index_df=sample_index_df,
        n_samples=n_samples,
        history_mins=history_mins,
        stride_mins=stride_mins,
        chunksize=800_000,
    ))

    rch_path = data_dir / "respiratoryCharting.csv.gz"
    rch_labels = find_top_numeric_labels(
        table_path=rch_path,
        label_col="respchartvaluelabel",
        value_col="respchartvalue",
        top_n=int(config.RESPCHART_TOP_LABELS),
        chunksize=800_000,
    )
    print("#############################")
    print(f"RespChart top numeric labels (first 20): {rch_labels[:20]}")
    print("#############################")
    blocks.append(process_charting_pivot(
        table_path=rch_path,
        pid_col="patientunitstayid",
        offset_col="respchartoffset",
        label_col="respchartvaluelabel",
        value_col="respchartvalue",
        prefix="rch_",
        sample_index_df=sample_index_df,
        n_samples=n_samples,
        history_mins=history_mins,
        stride_mins=stride_mins,
        top_labels=rch_labels,
        chunksize=800_000,
    ))

    nch_path = data_dir / "nurseCharting.csv.gz"
    nch_labels = find_top_numeric_labels(
        table_path=nch_path,
        label_col="nursingchartcelltypevallabel",
        value_col="nursingchartvalue",
        top_n=int(config.NURSECHART_TOP_LABELS),
        chunksize=800_000,
    )
    print("#############################")
    print(f"NurseChart top numeric labels (first 20): {nch_labels[:20]}")
    print("#############################")
    blocks.append(process_charting_pivot(
        table_path=nch_path,
        pid_col="patientunitstayid",
        offset_col="nursingchartoffset",
        label_col="nursingchartcelltypevallabel",
        value_col="nursingchartvalue",
        prefix="nch_",
        sample_index_df=sample_index_df,
        n_samples=n_samples,
        history_mins=history_mins,
        stride_mins=stride_mins,
        top_labels=nch_labels,
        chunksize=800_000,
    ))

    blocks.append(process_count_table(
        table_path=data_dir / "medication.csv.gz",
        pid_col="patientunitstayid",
        offset_col="drugstartoffset",
        sample_index_df=sample_index_df,
        n_samples=n_samples,
        history_mins=history_mins,
        stride_mins=stride_mins,
        chunksize=800_000,
        out_col="med_count_in_window",
    ))

    blocks.append(process_treatment_dialysis(
        table_path=data_dir / "treatment.csv.gz",
        sample_index_df=sample_index_df,
        n_samples=n_samples,
        history_mins=history_mins,
        stride_mins=stride_mins,
        chunksize=800_000,
    ))

    blocks.append(process_count_table(
        table_path=data_dir / "nurseAssessment.csv.gz",
        pid_col="patientunitstayid",
        offset_col="nurseassessoffset",
        sample_index_df=sample_index_df,
        n_samples=n_samples,
        history_mins=history_mins,
        stride_mins=stride_mins,
        chunksize=800_000,
        out_col="nurseassess_count_in_window",
    ))

    blocks.append(process_count_table(
        table_path=data_dir / "nurseCare.csv.gz",
        pid_col="patientunitstayid",
        offset_col="nursecareoffset",
        sample_index_df=sample_index_df,
        n_samples=n_samples,
        history_mins=history_mins,
        stride_mins=stride_mins,
        chunksize=800_000,
        out_col="nursecare_count_in_window",
    ))

    blocks.append(process_infusion_vasopressor_any(
        table_path=data_dir / "infusionDrug.csv.gz",
        sample_index_df=sample_index_df,
        n_samples=n_samples,
        history_mins=history_mins,
        stride_mins=stride_mins,
        chunksize=800_000,
    ))

    print("#############################")
    print("Final concatenation of feature blocks")
    print("#############################")

    feats = pd.concat(blocks, axis=1, copy=False)
    feats = ensure_no_duplicate_columns(feats)

    feats = add_vitals_missingness_masks(feats)
    feats = drop_sparse_nonvital_features(
        feats,
        min_non_missing_frac=MIN_NONVITAL_NONMISSING_FRAC,
    )
    feats = cast_numeric_feature_cols(feats)

    vitals_cols = [
        c for c in feats.columns
        if c in ["patientunitstayid", "t_end"] or c.startswith("vp_") or c.startswith("va_")
    ]
    vitals_feats = feats[vitals_cols].copy()

    baseline_cols = [c for c in baseline_feats.columns]
    baseline_feats = baseline_feats[baseline_cols].copy()
    baseline_feats = cast_numeric_feature_cols(baseline_feats)

    print("#############################")
    print("Applying permanent causal missing-value preprocessing")
    print("Method: <=120-minute within-stay forward fill, then training-split median")
    print("#############################")

    key_cols = ["patientunitstayid", "t_end"]
    model_features = vitals_feats.merge(
        baseline_feats,
        on=key_cols,
        how="inner",
        suffixes=("", "__dup"),
    )
    duplicate_cols = [c for c in model_features.columns if c.endswith("__dup")]
    if duplicate_cols:
        model_features = model_features.drop(columns=duplicate_cols)

    imputation_input = samples_df[key_cols + ["split"]].merge(
        model_features,
        on=key_cols,
        how="inner",
    )
    if len(imputation_input) != len(samples_df):
        raise RuntimeError(
            "Cannot permanently impute features because sample/feature keys do not align: "
            f"samples={len(samples_df)} merged={len(imputation_input)}"
        )

    model_feature_cols = [c for c in model_features.columns if c not in key_cols]
    imputed_model_features, training_medians, imputation_stats = impute_from_training_split(
        imputation_input,
        model_feature_cols,
        max_age_mins=120,
    )

    vitals_feature_cols = [c for c in vitals_feats.columns if c not in key_cols]
    baseline_feature_cols = [c for c in baseline_feats.columns if c not in key_cols]
    vitals_feats = imputed_model_features[key_cols + vitals_feature_cols].copy()
    baseline_feats = imputed_model_features[key_cols + baseline_feature_cols].copy()

    preprocessing_dir = output_dir / "Preprocessing"
    medians_path = preprocessing_dir / "imputation_medians__causal_ffill_120m_trainmedian.json"
    save_medians(
        medians_path,
        training_medians,
        max_age_mins=120,
        categories=classify_feature_columns(model_feature_cols),
    )
    handled = imputation_stats["forward_filled"] + imputation_stats["median_filled"]
    ff_share = 0.0 if handled == 0 else imputation_stats["forward_filled"] / handled
    print(
        f"Forward-filled: {imputation_stats['forward_filled']} | "
        f"Median fallback: {imputation_stats['median_filled']} | "
        f"Single-observation std set to zero: "
        f"{imputation_stats['single_observation_std_zeroed']} | "
        f"Forward-fill share: {ff_share:.2%}"
    )
    print(f"Saved fitted training medians: {medians_path}")

    print("#############################")
    print("Saving feature parquets")
    print("#############################")

    out_features_full.parent.mkdir(parents=True, exist_ok=True)
    out_features_vitals.parent.mkdir(parents=True, exist_ok=True)
    out_features_baseline.parent.mkdir(parents=True, exist_ok=True)

    feats.to_parquet(out_features_full, index=False)
    vitals_feats.to_parquet(out_features_vitals, index=False)
    baseline_feats.to_parquet(out_features_baseline, index=False)

    print(f"Saved full: {out_features_full}")
    print(f"Saved vitals: {out_features_vitals}")
    print(f"Saved baseline: {out_features_baseline}")
    print(f"Full shape: {feats.shape}")
    print(f"Vitals shape: {vitals_feats.shape}")
    print(f"Baseline shape: {baseline_feats.shape}")
    print("#############################")


if __name__ == "__main__":
    main()
