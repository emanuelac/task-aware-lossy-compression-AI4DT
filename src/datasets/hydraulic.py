"""
datasets/hydraulic.py

UCI "Condition Monitoring of Hydraulic Systems" (ZeMA / Helwig et al.
2015) — the industrial digital-twin instance of the compressor portfolio,
replacing IMS.

A hydraulic test rig repeats a constant 60 s load cycle 2205 times while
the condition of four components is varied. We evaluate on the eight
1 Hz sensors concatenated into a continuous 132 300-row stream (prep
script: scripts/prep_hydraulic.py). The classification label is the
*valve* condition, chosen because its four classes are varied on an
interleaved schedule and therefore appear in every block-disjoint CV
fold (verified) — unlike the monotone cooler/accumulator labels or the
run-to-failure IMS labels, so classification is non-degenerate on every
fold.

Layout of data/raw/hydraulic.csv
--------------------------------
timestamp, TS1, TS2, TS3, TS4, VS1, CE, CP, SE, valve

`valve` is the integer valve-condition class (100 optimal, 90 small lag,
80 severe lag, 73 near failure) — kept native for KNN.
"""

# ---------------------------------------------------------------------------
# Pipeline routing (unused for the offline runner, kept for symmetry)
# ---------------------------------------------------------------------------
KAFKA_TOPIC = "hydraulic-stream"
INFLUXDB_DATABASE = "hydraulic"
DB_SHORT = "hydraulic"

# ---------------------------------------------------------------------------
# Input — 8 one-Hz sensor channels (4 temperatures, vibration, and the
# three virtual efficiency/cooling channels the rig exposes).
# ---------------------------------------------------------------------------
INPUT_COLUMNS = ["TS1", "TS2", "TS3", "TS4", "VS1", "CE", "CP", "SE"]

# ---------------------------------------------------------------------------
# Forecasting target — a temperature channel (smooth, quasi-periodic over
# the repeated load cycle). TS2 is the mid-circuit temperature.
# ---------------------------------------------------------------------------
TARGET_COLUMNS = ["TS2"]

# ---------------------------------------------------------------------------
# Classification — valve condition (recurring classes; see module docstring)
# ---------------------------------------------------------------------------
CLASSIFICATION_LABEL_COL = "valve"

# One 60-sample window = one full load cycle at 1 Hz. The valve state is
# constant within a cycle, so a within-cycle window is a clean classifier
# input.
CLASSIFICATION_WINDOW_SIZE = 60

# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------
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

SAMPLING_RESOLUTION = "1s"  # 1 Hz
SOURCE = (
    "UCI Condition Monitoring of Hydraulic Systems "
    "(Helwig, Pignanelli & Schutze 2015) — "
    "https://archive.ics.uci.edu/dataset/447"
)

# ---------------------------------------------------------------------------
# Notebook / simulation knobs
# ---------------------------------------------------------------------------
TIMESTAMP_COL = "timestamp"
BATCH_SIZE = 60
WARMUP_N = 600

# ---------------------------------------------------------------------------
# Producer metadata (offline runner ignores this)
# ---------------------------------------------------------------------------
DATA_FILE = "raw/hydraulic.csv"
PRODUCER_SCRIPT = "producers/hydraulic_producer.py"
