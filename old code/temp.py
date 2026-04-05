# inspect_baseline_columns.py

from pathlib import Path
import pandas as pd

DATA_DIR = Path("../eICU(v2.0)")

FILES = [
    # "patient.csv.gz",
    # "pastHistory.csv.gz",
    # "apacheApsVar.csv.gz",
    # "apachePredVar.csv.gz",
    "vitalPeriodic.csv.gz",
    "vitalAperiodic.csv.gz",
]

def print_columns(file_path):
    print("########################################")
    print(f"FILE: {file_path.name}")
    print("########################################")

    df = pd.read_csv(file_path, nrows=0)
    for col in df.columns:
        print(col)

    print(f"\nTotal columns: {len(df.columns)}")
    print()

def main():
    for fname in FILES:
        path = DATA_DIR / fname
        if not path.exists():
            print(f"Missing file: {fname}")
            continue
        print_columns(path)

if __name__ == "__main__":
    main()

# temp_inspect_past_history.py
# temp_inspect_past_history.py

# temp_inspect_past_history_v2.py

# temp_inspect_past_history_v3.py
# temp_inspect_past_history_v4.py

# inspect_apache_pred_var.py
# from pathlib import Path
# import pandas as pd

# full = pd.read_parquet("../Outputs\cardiovascular__cardiac_arrest__hor4h__lead30m__stride60m__minhist1h__maxhist6h__featvitals/features__cardiovascular__cardiac_arrest.parquet")
# vitals = pd.read_parquet("../Outputs\cardiovascular__cardiac_arrest__hor4h__lead30m__stride60m__minhist1h__maxhist6h__featvitals/features_vitals__cardiovascular__cardiac_arrest.parquet")
# baseline = pd.read_parquet("../Outputs\cardiovascular__cardiac_arrest__hor4h__lead30m__stride60m__minhist1h__maxhist6h__featvitals/features_baseline__cardiovascular__cardiac_arrest.parquet")

# print(set(vitals.columns).issubset(set(full.columns)))
# print(set(baseline.columns).issubset(set(full.columns)))