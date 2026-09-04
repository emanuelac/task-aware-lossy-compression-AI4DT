"""
scripts/prep_metro.py

Prep the UCI Metro Interstate Traffic Volume dataset (Hogue) into the
single-stream CSV the offline pipeline expects.

Why this dataset
----------------
Traffic-volume forecasting is a canonical, deployed digital-twin task and
is genuinely predictable (strong daily/weekly seasonality), so it anchors a
forecasting head whose absolute utility is real. The classification head is
day-type (weekday vs weekend): the label comes from the calendar, so it is
orthogonal to every compressed channel (non-circular), it recurs in
every block-disjoint fold, and it is learnable from the traffic rhythm
(weekday rush-hour peaks vs a flatter weekend). We deliberately avoid the
``weather_main`` label because it is partly encoded in the cloud/precip
channels and so would not discriminate compressors.

Stream construction
-------------------
The record is hourly. We keep five numeric channels
(temp, rain_1h, snow_1h, clouds_all, traffic_volume) and derive the binary
``is_weekend`` label from the timestamp. Duplicate timestamps (a handful in
the raw file) are dropped, keeping the first.

Run:
    python3 scripts/prep_metro.py
Produces data/raw/metro.csv with the column layout used by
src/datasets/metro.py.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_CSV = REPO_ROOT / "data" / "raw" / "Metro_Interstate_Traffic_Volume.csv"
OUT_CSV = REPO_ROOT / "data" / "raw" / "metro.csv"

NUMERIC = ["temp", "rain_1h", "snow_1h", "clouds_all", "traffic_volume"]


def main() -> None:
    if not SRC_CSV.exists():
        raise SystemExit(f"[prep] source csv not found: {SRC_CSV}")

    df = pd.read_csv(SRC_CSV)
    df["timestamp"] = pd.to_datetime(df["date_time"])
    # A few duplicate hourly timestamps exist in the raw file; keep the first
    # and sort so the stream is monotone in time.
    df = (df.sort_values("timestamp")
            .drop_duplicates("timestamp", keep="first")
            .reset_index(drop=True))

    out = df[["timestamp"] + NUMERIC].copy()
    # Day-type label from the calendar (Mon=0..Sun=6); weekend = Sat/Sun.
    out["is_weekend"] = (df["timestamp"].dt.dayofweek >= 5).astype(int)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    print(f"[prep] wrote {OUT_CSV}  ({len(out)} rows, {len(NUMERIC)} channels)")
    print(f"[prep] is_weekend classes: "
          f"{dict(out['is_weekend'].value_counts().sort_index())}")


if __name__ == "__main__":
    main()
