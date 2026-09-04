"""
scripts/prep_opportunity.py

Build data/raw/opportunity.csv from the OPPORTUNITY distribution
(subject S1, ADL1 run only).

Source layout (expected under data/raw/OpportunityUCIDataset/):
    dataset/
        S1-ADL1.dat        ← used here
        S1-ADL{2..5}.dat
        S1-Drill.dat
        column_names.txt
        label_legend.txt

Each .dat file is space-separated, 250 columns. We extract a 9-channel
subset of body-worn accelerometers + the Locomotion label.

Selected channels (cols are 1-indexed per column_names.txt):
    col 11-13: RUA_ accelerometer xyz       (right upper arm)
    col 17-19: BACK accelerometer xyz       (torso)
    col 20-22: RKN_ accelerometer xyz       (right knee)
    col 244:   Locomotion label             (0=Null, 1=Stand, 2=Walk,
                                             4=Sit, 5=Lie — label 3 unused)

Forward-fill NaNs along the time axis (standard OPPORTUNITY preprocessing
for the challenge benchmark — the missing samples are isolated dropouts).

Locomotion = 0 (Null) rows are KEPT — Null is a legitimate class in
OPPORTUNITY's benchmark protocol.

Output (data/raw/opportunity.csv):
    timestamp,
    acc_rua_x,  acc_rua_y,  acc_rua_z,
    acc_back_x, acc_back_y, acc_back_z,
    acc_rkn_x,  acc_rkn_y,  acc_rkn_z,
    locomotion

Run once after extracting OPPORTUNITY into data/raw/OpportunityUCIDataset/:
    python3 scripts/prep_opportunity.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "data" / "raw"
SRC_FILE = RAW_DIR / "OpportunityUCIDataset" / "dataset" / "S1-ADL1.dat"
OUT_PATH = RAW_DIR / "opportunity.csv"

FS_HZ = 30
SAMPLE_PERIOD = pd.Timedelta(seconds=1 / FS_HZ)

# 0-indexed column positions (= 1-indexed in column_names.txt minus 1).
COL_ACC_RUA = [10, 11, 12]   # cols 11-13
COL_ACC_BACK = [16, 17, 18]  # cols 17-19
COL_ACC_RKN = [19, 20, 21]   # cols 20-22
COL_LOCOMOTION = 243         # col 244

CHANNELS = [
    "acc_rua_x",  "acc_rua_y",  "acc_rua_z",
    "acc_back_x", "acc_back_y", "acc_back_z",
    "acc_rkn_x",  "acc_rkn_y",  "acc_rkn_z",
]


def main() -> None:
    if not SRC_FILE.exists():
        raise FileNotFoundError(
            f"OPPORTUNITY ADL file not found at {SRC_FILE}. "
            f"Extract the OPPORTUNITY archive and drop the "
            f"'OpportunityUCIDataset' directory under data/raw/."
        )

    raw_names = [f"c{i}" for i in range(250)]
    df = pd.read_csv(SRC_FILE, sep=r"\s+", header=None,
                     names=raw_names, na_values="NaN")
    print(f"[opportunity] read {len(df)} raw rows from {SRC_FILE.name}")

    cols_idx = COL_ACC_RUA + COL_ACC_BACK + COL_ACC_RKN
    out = pd.DataFrame(df.iloc[:, cols_idx].values, columns=CHANNELS)
    # Locomotion may itself be NaN on some rows; treat as Null (=0).
    out["locomotion"] = df.iloc[:, COL_LOCOMOTION].fillna(0).astype(int)

    # OPPORTUNITY has isolated NaN samples (sensor dropouts). Standard
    # benchmark preprocessing is forward-fill within each recording, with
    # back-fill at the head if the file starts with NaN.
    n_nan_before = out[CHANNELS].isna().sum().sum()
    out[CHANNELS] = out[CHANNELS].ffill().bfill()
    n_nan_after = out[CHANNELS].isna().sum().sum()
    print(f"[opportunity] forward-filled {n_nan_before} NaN cells "
          f"(remaining: {n_nan_after})")
    if n_nan_after > 0:
        raise RuntimeError(
            f"{n_nan_after} NaN cells remain after ffill+bfill — "
            f"likely an entire column is NaN. Check SRC_FILE."
        )

    # Synthesise a uniform 30 Hz timestamp. The raw timestamp column is
    # in milliseconds and monotonic; the offline runner only needs strict
    # ordering and a uniform sample period.
    start = pd.Timestamp("2000-01-01")
    out.insert(0, "timestamp",
               start + np.arange(len(out), dtype=np.int64) * SAMPLE_PERIOD)
    out = out[["timestamp", *CHANNELS, "locomotion"]]

    print(f"[opportunity] n_total={len(out)}  "
          f"classes={sorted(out['locomotion'].unique().tolist())}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, index=False)
    print(f"[opportunity] wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
