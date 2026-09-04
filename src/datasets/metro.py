"""
datasets/metro.py

UCI Metro Interstate Traffic Volume (Hogue) — a mobility digital-twin
instance where forecasting is natural and predictable.

Hourly stream of ~48k rows (prep: scripts/prep_metro.py). The forecasting
target is traffic volume (a canonical, strongly-seasonal, deployed task).
The classification label is day-type (weekday vs weekend): the label comes
from the calendar, so it is orthogonal to every compressed channel
(non-circular), recurs in every block-disjoint fold, and is learnable from
the traffic rhythm. The dataset's ``weather_main`` label is deliberately
NOT used, because it is partly encoded in the cloud/precipitation channels
and would not discriminate compressors.

Layout of data/raw/metro.csv
----------------------------
timestamp, temp, rain_1h, snow_1h, clouds_all, traffic_volume, is_weekend
"""

# ---------------------------------------------------------------------------
# Pipeline routing (unused for the offline runner, kept for symmetry)
# ---------------------------------------------------------------------------
KAFKA_TOPIC = "metro-stream"
INFLUXDB_DATABASE = "metro"
DB_SHORT = "metro"

# ---------------------------------------------------------------------------
# Input — 5 numeric channels (weather + traffic)
# ---------------------------------------------------------------------------
INPUT_COLUMNS = ["temp", "rain_1h", "snow_1h", "clouds_all", "traffic_volume"]

# ---------------------------------------------------------------------------
# Forecasting target — traffic volume (the canonical mobility task)
# ---------------------------------------------------------------------------
TARGET_COLUMNS = ["traffic_volume"]

# ---------------------------------------------------------------------------
# Classification — day-type (weekday vs weekend); see docstring
# ---------------------------------------------------------------------------
CLASSIFICATION_LABEL_COL = "is_weekend"

# 24-sample window = one day at hourly resolution, enough to expose the
# rush-hour shape that distinguishes weekday from weekend.
CLASSIFICATION_WINDOW_SIZE = 24

# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------
CHANNEL_NAMES = {
    "temp": "Temperature (K)",
    "rain_1h": "Rain, 1h (mm)",
    "snow_1h": "Snow, 1h (mm)",
    "clouds_all": "Cloud cover (%)",
    "traffic_volume": "Traffic volume (vehicles/h)",
}

SAMPLING_RESOLUTION = "1h"  # hourly
SOURCE = (
    "UCI Metro Interstate Traffic Volume (Hogue) — "
    "https://archive.ics.uci.edu/dataset/492"
)

# ---------------------------------------------------------------------------
# Notebook / simulation knobs
# ---------------------------------------------------------------------------
TIMESTAMP_COL = "timestamp"
BATCH_SIZE = 48
WARMUP_N = 480

# ---------------------------------------------------------------------------
# Producer metadata (offline runner ignores this)
# ---------------------------------------------------------------------------
DATA_FILE = "raw/metro.csv"
PRODUCER_SCRIPT = "producers/metro_producer.py"
