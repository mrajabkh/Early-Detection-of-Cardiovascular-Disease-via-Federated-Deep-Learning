# run_multi_horizon.py
# Run the full pipeline across multiple horizons by temporarily editing config.py.
#
# What it runs per horizon:
#  1) prepare_data.py
#  2) aggregate_features.py
#  3) feature_selection_stability.py
#  4) run_gru_sweep.py
#
# Notes:
# - This script edits config.py in-place to set HORIZON_HRS, then restores it at the end.
# - Your Outputs folders stay separated automatically because run_name() includes horizon.
# - Standard ASCII only.

from __future__ import annotations

import re
import sys
import subprocess
from pathlib import Path
from typing import List


#############################
# User settings
#############################
HORIZONS_HRS: List[int] = [2, 6, 12, 24]

RUN_PREPARE_DATA = True
RUN_AGG_FEATURES = True
RUN_FEATURE_SELECTION = True
RUN_GRU_SWEEP = True

# If you want to skip feature selection for speed, set RUN_FEATURE_SELECTION = False.
# If you want to skip prepare/features when only changing model settings, set RUN_PREPARE_DATA/RUN_AGG_FEATURES = False.

#############################
# Paths (assumes this script lives in Project/Code/)
#############################
CODE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = CODE_DIR / "config.py"

PREPARE_DATA_PATH = CODE_DIR / "prepare_data.py"
AGG_FEATURES_PATH = CODE_DIR / "aggregate_features.py"
FEATSEL_PATH = CODE_DIR / "feature_selection_stability.py"
GRU_SWEEP_PATH = CODE_DIR / "run_gru_sweep.py"


#############################
# Helpers
#############################
def _run(cmd: List[str]) -> None:
    print("#############################")
    print("Running:")
    print(" ".join(cmd))
    print("#############################")
    subprocess.run(cmd, check=True)


def _set_horizon_in_config(config_text: str, horizon_hrs: int) -> str:
    # Replace only the HORIZON_HRS line. Derived values (HORIZON_MINS etc) are computed in config.py.
    pat = re.compile(r"^HORIZON_HRS\s*=\s*\d+\s*$", flags=re.MULTILINE)
    if not pat.search(config_text):
        raise RuntimeError("Could not find 'HORIZON_HRS = <int>' line in config.py")
    new_text = pat.sub(f"HORIZON_HRS = {int(horizon_hrs)}", config_text)
    return new_text


def _verify_files_exist() -> None:
    need = [CONFIG_PATH, PREPARE_DATA_PATH, AGG_FEATURES_PATH, FEATSEL_PATH, GRU_SWEEP_PATH]
    missing = [p for p in need if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required files: {[str(p) for p in missing]}")


#############################
# Main
#############################
def main() -> None:
    _verify_files_exist()

    original = CONFIG_PATH.read_text(encoding="utf-8")

    try:
        for h in HORIZONS_HRS:
            print("========================================")
            print(f"Multi-horizon run: HORIZON_HRS={h}")
            print("========================================")

            updated = _set_horizon_in_config(original, h)
            CONFIG_PATH.write_text(updated, encoding="utf-8")

            # Run pipeline steps
            py = sys.executable

            if RUN_PREPARE_DATA:
                _run([py, str(PREPARE_DATA_PATH)])

            if RUN_AGG_FEATURES:
                _run([py, str(AGG_FEATURES_PATH)])

            if RUN_FEATURE_SELECTION:
                _run([py, str(FEATSEL_PATH)])

            if RUN_GRU_SWEEP:
                _run([py, str(GRU_SWEEP_PATH)])

            print("========================================")
            print(f"Done horizon {h}h")
            print("========================================")

    finally:
        # Always restore config.py
        CONFIG_PATH.write_text(original, encoding="utf-8")
        print("#############################")
        print("Restored original config.py")
        print("#############################")


if __name__ == "__main__":
    main()
