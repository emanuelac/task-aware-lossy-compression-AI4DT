"""
datasets/beijing.py

UCI Beijing Multi-Site Air-Quality (Zhang et al. 2017) — an environmental
digital-twin instance where forecasting is both natural and predictable.

One monitoring station (Aotizhongxin) as a continuous hourly stream of
~35k rows (prep: scripts/prep_beijing.py). The forecasting target is PM2.5
(a canonical, autocorrelated, deployed air-quality task). The
classification label is the wind quadrant (N/E/S/W, grouped from the 16
raw sectors) — recurring in every block-disjoint fold and orthogonal to
the numeric input channels, so non-circular.

Layout of data/raw/beijing.csv
------------------------------
timestamp, PM2.5, PM10, SO2, NO2, CO, O3, TEMP, PRES, DEWP, RAIN, WSPM,
           wind_quadrant
"""

# ---------------------------------------------------------------------------
# Pipeline routing (unused for the offline runner, kept for symmetry)
# ---------------------------------------------------------------------------
KAFKA_TOPIC = "beijing-stream"
INFLUXDB_DATABASE = "beijing"
DB_SHORT = "beijing"

# ---------------------------------------------------------------------------
# Input — 11 numeric channels (6 pollutants + 5 meteorology)
# ---------------------------------------------------------------------------
INPUT_COLUMNS = ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3",
                 "TEMP", "PRES", "DEWP", "RAIN", "WSPM"]

# ---------------------------------------------------------------------------
# Forecasting target — PM2.5 (the canonical air-quality forecasting task)
# ---------------------------------------------------------------------------
TARGET_COLUMNS = ["PM2.5"]

# ---------------------------------------------------------------------------
# Classification — wind quadrant (recurring, non-circular; see docstring)
# ---------------------------------------------------------------------------
CLASSIFICATION_LABEL_COL = "wind_quadrant"

# 24-sample window = one day at 1 Hz (hourly), enough context for the
# diurnal pollution/meteorology pattern the wind sector modulates.
CLASSIFICATION_WINDOW_SIZE = 24

# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------
CHANNEL_NAMES = {
    "PM2.5": "PM2.5 concentration (ug/m^3)",
    "PM10": "PM10 concentration (ug/m^3)",
    "SO2": "SO2 concentration (ug/m^3)",
    "NO2": "NO2 concentration (ug/m^3)",
    "CO": "CO concentration (ug/m^3)",
    "O3": "O3 concentration (ug/m^3)",
    "TEMP": "Temperature (deg C)",
    "PRES": "Pressure (hPa)",
    "DEWP": "Dew point (deg C)",
    "RAIN": "Precipitation (mm)",
    "WSPM": "Wind speed (m/s)",
}

SAMPLING_RESOLUTION = "1h"  # hourly
SOURCE = (
    "UCI Beijing Multi-Site Air-Quality (Zhang et al. 2017) — "
    "https://archive.ics.uci.edu/dataset/501"
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
DATA_FILE = "raw/beijing.csv"
PRODUCER_SCRIPT = "producers/beijing_producer.py"
