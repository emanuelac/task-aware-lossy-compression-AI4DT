"""UCI HAR (Human Activity Recognition, Anguita et al. 2013) — biomedical DT instance.
9 inertial channels at 50 Hz; forecast body acceleration, classify activity (6 classes)."""

KAFKA_TOPIC = "uci-har-stream"
INFLUXDB_DATABASE = "uci_har"
DB_SHORT = "uci_har"

INPUT_COLUMNS = [
    "body_acc_x", "body_acc_y", "body_acc_z",
    "body_gyro_x", "body_gyro_y", "body_gyro_z",
    "total_acc_x", "total_acc_y", "total_acc_z",
]
TARGET_COLUMNS = ["body_acc_x", "body_acc_y", "body_acc_z"]

CLASSIFICATION_LABEL_COL = "activity"
CLASSIFICATION_WINDOW_SIZE = 128  # standard UCI HAR window (~2.5 s at 50 Hz)

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

SAMPLING_RESOLUTION = "20ms"
SOURCE = (
    "UCI HAR Dataset (Anguita et al. 2013) — "
    "https://archive.ics.uci.edu/dataset/240"
)

TIMESTAMP_COL = "timestamp"
BATCH_SIZE = 256
WARMUP_N = 1600

DATA_FILE = "raw/uci_har.csv"
PRODUCER_SCRIPT = "producers/uci_har_producer.py"
