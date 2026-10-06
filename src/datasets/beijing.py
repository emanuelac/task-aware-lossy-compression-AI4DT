"""Beijing Multi-Site Air-Quality (Zhang et al. 2017) — environmental DT instance.
One station, hourly ~35k rows; forecast PM2.5, classify wind quadrant (N/E/S/W)."""

KAFKA_TOPIC = "beijing-stream"
INFLUXDB_DATABASE = "beijing"
DB_SHORT = "beijing"

INPUT_COLUMNS = ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3",
                 "TEMP", "PRES", "DEWP", "RAIN", "WSPM"]
TARGET_COLUMNS = ["PM2.5"]

CLASSIFICATION_LABEL_COL = "wind_quadrant"
CLASSIFICATION_WINDOW_SIZE = 24  # one day at hourly resolution

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

SAMPLING_RESOLUTION = "1h"
SOURCE = (
    "UCI Beijing Multi-Site Air-Quality (Zhang et al. 2017) — "
    "https://archive.ics.uci.edu/dataset/501"
)

TIMESTAMP_COL = "timestamp"
BATCH_SIZE = 48
WARMUP_N = 480

DATA_FILE = "raw/beijing.csv"
PRODUCER_SCRIPT = "producers/beijing_producer.py"
