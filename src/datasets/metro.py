"""Metro Interstate Traffic Volume (Hogue) — mobility DT instance.
Hourly ~48k rows; forecast traffic volume, classify weekday vs weekend."""

KAFKA_TOPIC = "metro-stream"
INFLUXDB_DATABASE = "metro"
DB_SHORT = "metro"

INPUT_COLUMNS = ["temp", "rain_1h", "snow_1h", "clouds_all", "traffic_volume"]
TARGET_COLUMNS = ["traffic_volume"]

CLASSIFICATION_LABEL_COL = "is_weekend"
CLASSIFICATION_WINDOW_SIZE = 24  # one day at hourly resolution

CHANNEL_NAMES = {
    "temp": "Temperature (K)",
    "rain_1h": "Rain, 1h (mm)",
    "snow_1h": "Snow, 1h (mm)",
    "clouds_all": "Cloud cover (%)",
    "traffic_volume": "Traffic volume (vehicles/h)",
}

SAMPLING_RESOLUTION = "1h"
SOURCE = (
    "UCI Metro Interstate Traffic Volume (Hogue) — "
    "https://archive.ics.uci.edu/dataset/492"
)

TIMESTAMP_COL = "timestamp"
BATCH_SIZE = 48
WARMUP_N = 480

DATA_FILE = "raw/metro.csv"
PRODUCER_SCRIPT = "producers/metro_producer.py"
