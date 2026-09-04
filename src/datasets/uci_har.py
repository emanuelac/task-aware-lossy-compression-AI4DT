"""
datasets/uci_har.py

Dataset configuration for UCI HAR (Human Activity Recognition Using
Smartphones, Anguita et al. 2013) — one of the two classification-capable
datasets in the compressor portfolio.

Source
------
UCI ML Repository, dataset 240:
    https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones

Smartphone accelerometer (3 axes) + gyroscope (3 axes) + a derived
body-acceleration signal (3 axes) sampled at 50 Hz while 30 subjects
performed six activities (WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS,
SITTING, STANDING, LAYING).

The classic UCI distribution ships *pre-windowed* 561-feature vectors —
useless for streaming-style compression. We want the raw inertial-signal
streams, distributed in the archive under:

    UCI HAR Dataset/train/Inertial Signals/
    UCI HAR Dataset/test/Inertial Signals/

Each "signal" file contains 128-sample windows per row; concatenating the
windows back into a continuous stream gives one CSV per subject. The
activity label per window is given by
    UCI HAR Dataset/train/y_train.txt
expanded to per-sample labels.

Prep script (run once, after downloading the archive):
    python3 scripts/prep_uci_har.py
This produces  data/raw/uci_har.csv  with the column layout below.

Layout of data/raw/uci_har.csv
------------------------------
timestamp, body_acc_x, body_acc_y, body_acc_z,
           body_gyro_x, body_gyro_y, body_gyro_z,
           total_acc_x, total_acc_y, total_acc_z,
           activity

`activity` is the integer activity label (1-6) — kept native so KNN can
operate on the original label space without an LabelEncoder dance.
"""

# ---------------------------------------------------------------------------
# Pipeline routing (unused for the offline runner, kept for symmetry)
# ---------------------------------------------------------------------------

KAFKA_TOPIC = "uci-har-stream"
INFLUXDB_DATABASE = "uci_har"
DB_SHORT = "uci_har"

# ---------------------------------------------------------------------------
# Input — 9 inertial signal channels
# ---------------------------------------------------------------------------

INPUT_COLUMNS = [
    "body_acc_x", "body_acc_y", "body_acc_z",
    "body_gyro_x", "body_gyro_y", "body_gyro_z",
    "total_acc_x", "total_acc_y", "total_acc_z",
]

# ---------------------------------------------------------------------------
# Forecasting targets — predict the three body-accel channels. Picking a
# subset keeps the forecasting head from blowing up to nine simultaneous
# targets; body_acc is the canonical HAR signal in the literature.
# ---------------------------------------------------------------------------

TARGET_COLUMNS = ["body_acc_x", "body_acc_y", "body_acc_z"]

# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

# Native label column carrying the six-class activity ID. ClassificationTask
# reads this directly from raw_train / raw_test — no quantile synthesis.
CLASSIFICATION_LABEL_COL = "activity"

# Longer window than the forecasting embedding so the classifier sees a
# meaningful slice of activity context. 50 Hz · ~2.5 s = 128 samples is
# the standard UCI HAR window. Forecasting still uses cfg.WINDOW_SIZE.
CLASSIFICATION_WINDOW_SIZE = 128

# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

CHANNEL_NAMES = {
    "body_acc_x": "Body acceleration X (g)",
    "body_acc_y": "Body acceleration Y (g)",
    "body_acc_z": "Body acceleration Z (g)",
    "body_gyro_x": "Body angular velocity X (rad/s)",
    "body_gyro_y": "Body angular velocity Y (rad/s)",
    "body_gyro_z": "Body angular velocity Z (rad/s)",
    "total_acc_x": "Total acceleration X (g)",
    "total_acc_y": "Total acceleration Y (g)",
    "total_acc_z": "Total acceleration Z (g)",
}

SAMPLING_RESOLUTION = "20ms"   # 50 Hz
SOURCE = (
    "UCI HAR Dataset (Anguita et al. 2013) — "
    "https://archive.ics.uci.edu/dataset/240"
)

# ---------------------------------------------------------------------------
# Notebook / simulation knobs
# ---------------------------------------------------------------------------
TIMESTAMP_COL = "timestamp"
BATCH_SIZE    = 256
WARMUP_N      = 1600

# ---------------------------------------------------------------------------
# Producer metadata (offline runner ignores this)
# ---------------------------------------------------------------------------
DATA_FILE = "raw/uci_har.csv"
PRODUCER_SCRIPT = "producers/uci_har_producer.py"
