"""OPPORTUNITY Activity Recognition (Roggen et al. 2010) — biomedical DT instance.
9 body-worn accelerometer channels (arm/torso/leg) at 30 Hz; forecast back accel,
classify locomotion (Null/Stand/Walk/Sit/Lie). Subject S1, ADL1 run."""

KAFKA_TOPIC = "opportunity-stream"
INFLUXDB_DATABASE = "opportunity"
DB_SHORT = "opportunity"

INPUT_COLUMNS = [
    "acc_rua_x",  "acc_rua_y",  "acc_rua_z",
    "acc_back_x", "acc_back_y", "acc_back_z",
    "acc_rkn_x",  "acc_rkn_y",  "acc_rkn_z",
]
TARGET_COLUMNS = ["acc_back_x", "acc_back_y", "acc_back_z"]

CLASSIFICATION_LABEL_COL = "locomotion"
CLASSIFICATION_WINDOW_SIZE = 60  # ~2 s at 30 Hz

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

SAMPLING_RESOLUTION = "33ms"
SOURCE = (
    "OPPORTUNITY Activity Recognition (Roggen et al. 2010) — "
    "https://archive.ics.uci.edu/dataset/226"
)

TIMESTAMP_COL = "timestamp"
BATCH_SIZE = 256
WARMUP_N = 1500

DATA_FILE = "raw/opportunity.csv"
PRODUCER_SCRIPT = "producers/opportunity_producer.py"
