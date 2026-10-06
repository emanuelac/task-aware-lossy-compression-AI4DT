"""Condition Monitoring of Hydraulic Systems (Helwig et al. 2015) — industrial DT instance.
Eight 1 Hz sensors, 132 300-row stream; forecast temperature TS2, classify valve condition."""

KAFKA_TOPIC = "hydraulic-stream"
INFLUXDB_DATABASE = "hydraulic"
DB_SHORT = "hydraulic"

INPUT_COLUMNS = ["TS1", "TS2", "TS3", "TS4", "VS1", "CE", "CP", "SE"]
TARGET_COLUMNS = ["TS2"]

CLASSIFICATION_LABEL_COL = "valve"  # classes recur in every CV fold (interleaved schedule)
CLASSIFICATION_WINDOW_SIZE = 60     # one 60 s load cycle at 1 Hz

CHANNEL_NAMES = {
    "TS1": "Temperature 1 (deg C)",
    "TS2": "Temperature 2 (deg C)",
    "TS3": "Temperature 3 (deg C)",
    "TS4": "Temperature 4 (deg C)",
    "VS1": "Vibration (mm/s)",
    "CE": "Cooling efficiency, virtual (%)",
    "CP": "Cooling power, virtual (kW)",
    "SE": "Efficiency factor (%)",
}

SAMPLING_RESOLUTION = "1s"
SOURCE = (
    "UCI Condition Monitoring of Hydraulic Systems "
    "(Helwig, Pignanelli & Schutze 2015) — "
    "https://archive.ics.uci.edu/dataset/447"
)

TIMESTAMP_COL = "timestamp"
BATCH_SIZE = 60
WARMUP_N = 600

DATA_FILE = "raw/hydraulic.csv"
PRODUCER_SCRIPT = "producers/hydraulic_producer.py"
