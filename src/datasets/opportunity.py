"""
datasets/opportunity.py

Dataset configuration for OPPORTUNITY (Activity Recognition) — fourth
classification-capable dataset in the compressor portfolio.

Source
------
UCI ML Repository, dataset 226:
    https://archive.ics.uci.edu/dataset/226/opportunity+activity+recognition

Body-worn-sensor activity recognition collected from 4 subjects performing
a morning-routine scenario (kitchen activities, transitions, drinks,
sandwich preparation). The full dataset has 250 channels (body-worn IMUs,
ambient sensors, object sensors); we use subject S1's ADL1 run only.

We extract a 9-channel subset of body-worn accelerometers (RUA + BACK +
RKN). The Locomotion label track (5 classes:
Null, Stand, Walk, Sit, Lie) is the cleanest, best-populated label
channel; finer tracks (Mid-Level Gesture etc.) are sparser.

Prep script (run once after extracting the archive):
    python3 scripts/prep_opportunity.py
This produces  data/raw/opportunity.csv  with the column layout below.

Layout of data/raw/opportunity.csv
----------------------------------
timestamp,
acc_rua_x,  acc_rua_y,  acc_rua_z,
acc_back_x, acc_back_y, acc_back_z,
acc_rkn_x,  acc_rkn_y,  acc_rkn_z,
locomotion

`locomotion` is the integer Locomotion-track label
(0=Null, 1=Stand, 2=Walk, 4=Sit, 5=Lie — note label 3 is unused in the
official legend). KNN handles non-contiguous integer labels natively.
"""

# ---------------------------------------------------------------------------
# Pipeline routing (unused for the offline runner, kept for symmetry)
# ---------------------------------------------------------------------------

KAFKA_TOPIC = "opportunity-stream"
INFLUXDB_DATABASE = "opportunity"
DB_SHORT = "opportunity"

# ---------------------------------------------------------------------------
# Input — 9 body-worn accelerometer channels (RUA/BACK/RKN × 3 axes):
# three sensor locations — arm, torso, leg.
# ---------------------------------------------------------------------------

INPUT_COLUMNS = [
    "acc_rua_x",  "acc_rua_y",  "acc_rua_z",
    "acc_back_x", "acc_back_y", "acc_back_z",
    "acc_rkn_x",  "acc_rkn_y",  "acc_rkn_z",
]

# ---------------------------------------------------------------------------
# Forecasting targets — BACK accelerometer (central, most-stable location).
# ---------------------------------------------------------------------------

TARGET_COLUMNS = ["acc_back_x", "acc_back_y", "acc_back_z"]

# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

CLASSIFICATION_LABEL_COL = "locomotion"

# 30 Hz · ~2 s = 60 samples. OPPORTUNITY's challenge benchmark uses
# 1–3 s sliding windows over its 30 Hz sample rate.
CLASSIFICATION_WINDOW_SIZE = 60

# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

CHANNEL_NAMES = {
    "acc_rua_x":  "Right-upper-arm accelerometer X (mg)",
    "acc_rua_y":  "Right-upper-arm accelerometer Y (mg)",
    "acc_rua_z":  "Right-upper-arm accelerometer Z (mg)",
    "acc_back_x": "Back accelerometer X (mg)",
    "acc_back_y": "Back accelerometer Y (mg)",
    "acc_back_z": "Back accelerometer Z (mg)",
    "acc_rkn_x":  "Right-knee accelerometer X (mg)",
    "acc_rkn_y":  "Right-knee accelerometer Y (mg)",
    "acc_rkn_z":  "Right-knee accelerometer Z (mg)",
}

SAMPLING_RESOLUTION = "33ms"   # 30 Hz (~33.33 ms)
SOURCE = (
    "OPPORTUNITY Activity Recognition (Roggen et al. 2010) — "
    "https://archive.ics.uci.edu/dataset/226"
)

# ---------------------------------------------------------------------------
# Notebook / simulation knobs
# ---------------------------------------------------------------------------
TIMESTAMP_COL = "timestamp"
BATCH_SIZE    = 256
WARMUP_N      = 1500

# ---------------------------------------------------------------------------
# Producer metadata (offline runner ignores this)
# ---------------------------------------------------------------------------
DATA_FILE = "raw/opportunity.csv"
PRODUCER_SCRIPT = "producers/opportunity_producer.py"
