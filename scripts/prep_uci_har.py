"""
scripts/prep_uci_har.py

Build data/raw/uci_har.csv from the UCI HAR distribution.

Source layout (expected under data/raw/UCI HAR Dataset/):
    activity_labels.txt
    train/
        y_train.txt
        subject_train.txt
        Inertial Signals/
            body_acc_{x,y,z}_train.txt
            body_gyro_{x,y,z}_train.txt
            total_acc_{x,y,z}_train.txt
    test/
        y_test.txt
        subject_test.txt
        Inertial Signals/
            body_acc_{x,y,z}_test.txt
            body_gyro_{x,y,z}_test.txt
            total_acc_{x,y,z}_test.txt

Window stitching (the load-bearing transform)
---------------------------------------------
Each `*_{train,test}.txt` signal file has shape (n_windows, 128). UCI HAR
applied a 50 % overlap sliding window when building the file — every row
is a 2.56 s window of 128 samples, and consecutive rows share their first
64 samples with the previous row's last 64.

To recover a continuous 50 Hz stream we keep row 0 in full (128 samples)
and only the last 64 samples of every subsequent row. The result is:

    n_continuous = 128 + (n_windows - 1) * 64

samples per split. Labels and subjects are carried through per 64-sample
chunk: the first 128 samples take row 0's label/subject, and each
subsequent block of 64 samples takes the next row's.

Train and test are then concatenated (train first). The single
train→test boundary is a tiny discontinuity (3 rows of 50 Hz = 60 ms of
context for the W=4 embedding) — negligible compared to the ~659k-row
total.

Output layout (data/raw/uci_har.csv):
    timestamp,
    body_acc_x, body_acc_y, body_acc_z,
    body_gyro_x, body_gyro_y, body_gyro_z,
    total_acc_x, total_acc_y, total_acc_z,
    activity, subject

`activity` is the integer label 1–6 (mapping in activity_labels.txt).
The classifier reads it natively; the runner already preserves the
label column through the loader.

Run once after extracting the UCI HAR archive into data/raw/:
    python3 scripts/prep_uci_har.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "data" / "raw"
SRC_DIR = RAW_DIR / "UCI HAR Dataset"
OUT_PATH = RAW_DIR / "uci_har.csv"

FS_HZ = 50
SAMPLE_PERIOD = pd.Timedelta(seconds=1 / FS_HZ)
WINDOW_LEN = 128         # samples per row in the inertial signal files
STRIDE = 64              # 50 % overlap → 64 fresh samples per subsequent row

# Order of the nine inertial-signal channels in the output CSV. Matches the
# INPUT_COLUMNS list in src/datasets/uci_har.py exactly so the runner reads
# them in the same order it expects.
CHANNELS = [
    "body_acc_x",  "body_acc_y",  "body_acc_z",
    "body_gyro_x", "body_gyro_y", "body_gyro_z",
    "total_acc_x", "total_acc_y", "total_acc_z",
]


def _signal_path(split: str, channel: str) -> Path:
    return (
        SRC_DIR
        / split
        / "Inertial Signals"
        / f"{channel}_{split}.txt"
    )


def load_split(split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load all 9 inertial channels + labels + subjects for one split.

    Returns
    -------
    signals : ndarray of shape (n_windows, 128, 9)
        Windowed signal stack. Axes: (window, sample-within-window, channel).
    labels : ndarray of shape (n_windows,)
        Integer activity label per window.
    subjects : ndarray of shape (n_windows,)
        Integer subject ID per window.
    """
    chans = []
    for ch in CHANNELS:
        path = _signal_path(split, ch)
        if not path.exists():
            raise FileNotFoundError(f"missing signal file: {path}")
        arr = np.loadtxt(path)            # shape: (n_windows, 128)
        chans.append(arr)
    signals = np.stack(chans, axis=-1)    # (n_windows, 128, 9)
    labels = np.loadtxt(SRC_DIR / split / f"y_{split}.txt", dtype=int)
    subjects = np.loadtxt(SRC_DIR / split / f"subject_{split}.txt", dtype=int)
    if len(labels) != signals.shape[0]:
        raise ValueError(
            f"{split}: signals have {signals.shape[0]} windows "
            f"but y has {len(labels)} labels"
        )
    return signals, labels, subjects


def stitch_split(
    signals: np.ndarray, labels: np.ndarray, subjects: np.ndarray
) -> pd.DataFrame:
    """
    Undo the 50 % overlap to recover a continuous-time stream.

    Strategy: take row 0 entirely (128 samples), then concatenate only the
    last 64 samples of each subsequent row. Labels and subjects ride along
    per source-row: row 0's label tags the first 128 samples, then each
    row k>0's label tags the next 64.
    """
    n_windows, _, n_channels = signals.shape
    # Pre-allocate the output buffer for speed (np.concatenate per row would
    # be O(n²) in memory copies).
    n_total = WINDOW_LEN + (n_windows - 1) * STRIDE
    out = np.empty((n_total, n_channels), dtype=np.float64)
    out_labels = np.empty(n_total, dtype=labels.dtype)
    out_subjects = np.empty(n_total, dtype=subjects.dtype)

    # Row 0 in full.
    out[:WINDOW_LEN] = signals[0]
    out_labels[:WINDOW_LEN] = labels[0]
    out_subjects[:WINDOW_LEN] = subjects[0]

    # Subsequent rows: their last STRIDE samples are the only fresh ones.
    for k in range(1, n_windows):
        lo = WINDOW_LEN + (k - 1) * STRIDE
        hi = lo + STRIDE
        out[lo:hi] = signals[k, -STRIDE:]
        out_labels[lo:hi] = labels[k]
        out_subjects[lo:hi] = subjects[k]

    df = pd.DataFrame(out, columns=CHANNELS)
    df["activity"] = out_labels
    df["subject"] = out_subjects
    return df


def main() -> None:
    if not SRC_DIR.exists():
        raise FileNotFoundError(
            f"UCI HAR Dataset folder not found at {SRC_DIR}. "
            f"Extract the official zip and drop the 'UCI HAR Dataset' "
            f"directory under data/raw/."
        )

    pieces: list[pd.DataFrame] = []
    for split in ("train", "test"):
        sigs, ys, subs = load_split(split)
        piece = stitch_split(sigs, ys, subs)
        n_classes = int(piece["activity"].nunique())
        n_subjects = int(piece["subject"].nunique())
        print(f"[uci_har] {split}: n_windows={sigs.shape[0]:>5}  "
              f"→ n_samples={len(piece):>7}  "
              f"classes={n_classes}  subjects={n_subjects}")
        pieces.append(piece)

    df = pd.concat(pieces, ignore_index=True)

    # Synthesise a monotonic 50 Hz timestamp. UCI HAR has no absolute clock
    # in the raw file — the runner only needs strict ordering.
    start = pd.Timestamp("2000-01-01")
    df.insert(0, "timestamp",
              start + np.arange(len(df), dtype=np.int64) * SAMPLE_PERIOD)

    # Reorder columns so the CSV layout exactly matches the docstring.
    df = df[["timestamp", *CHANNELS, "activity", "subject"]]

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"\n[uci_har] wrote {OUT_PATH}  n_total={len(df)}  "
          f"classes={sorted(df['activity'].unique().tolist())}  "
          f"subjects={sorted(df['subject'].unique().tolist())}")


if __name__ == "__main__":
    main()
