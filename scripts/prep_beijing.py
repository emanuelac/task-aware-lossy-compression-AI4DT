"""
scripts/prep_beijing.py

Prep the UCI Beijing Multi-Site Air-Quality dataset (Zhang et al. 2017)
into the single-stream CSV the offline pipeline expects.

Why this dataset
----------------
Air-quality monitoring is a digital-twin instance in which forecasting is
both natural and predictable: PM2.5 is strongly autocorrelated, so
forecasting it is a genuine deployed task (unlike raw-signal forecasting on
behavioural streams). The stream also carries a recurring, non-circular
classification target -- the wind direction -- so it supplies a forecasting
head whose absolute utility is real, alongside a clean classification head.

Stream construction
-------------------
We use a single monitoring station (Aotizhongxin) as one continuous 1 Hz
(hourly) stream of ~35k rows. The 11 numeric channels are kept as inputs;
the categorical wind direction ``wd`` (16 sectors) is grouped into four
quadrants (N/E/S/W) as the classification label -- verified to appear in
every block-disjoint CV fold. The forecasting target is PM2.5. Sparse
missing values (<= ~5% per channel) are forward/back-filled.

Run (after downloading the archive into
data/raw/beijing+multi+site+air+quality+data/):
    python3 scripts/prep_beijing.py
Produces data/raw/beijing.csv with the column layout used by
src/datasets/beijing.py.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ZIP = (REPO_ROOT / "data" / "raw" /
           "beijing+multi+site+air+quality+data" /
           "PRSA2017_Data_20130301-20170228.zip")
STATION = "Aotizhongxin"
OUT_CSV = REPO_ROOT / "data" / "raw" / "beijing.csv"

NUMERIC = ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3",
           "TEMP", "PRES", "DEWP", "RAIN", "WSPM"]

# 16 wind sectors -> 4 quadrants. Boundary sectors are assigned to the
# quadrant they lean toward (NE->E, NW->N, SE->S, SW->W) so each quadrant
# spans a contiguous 90-degree arc centred on its cardinal.
QUADRANT = {
    "N": "N", "NNE": "N", "NNW": "N", "NW": "N",
    "NE": "E", "ENE": "E", "E": "E", "ESE": "E",
    "SE": "S", "SSE": "S", "S": "S", "SSW": "S",
    "SW": "W", "WSW": "W", "W": "W", "WNW": "W",
}


def main() -> None:
    if not SRC_ZIP.exists():
        raise SystemExit(f"[prep] source zip not found: {SRC_ZIP}")

    with zipfile.ZipFile(SRC_ZIP) as z:
        name = [n for n in z.namelist() if STATION in n and n.endswith(".csv")][0]
        df = pd.read_csv(z.open(name))

    # Synthetic monotonic hourly timestamp from the calendar columns.
    ts = pd.to_datetime(df[["year", "month", "day", "hour"]])

    out = df[NUMERIC].copy()
    # Sparse gaps (DST/maintenance) -> forward/back-fill; <~5% per channel.
    n_nan = int(out.isna().any(axis=1).sum())
    out = out.ffill().bfill()

    # Wind-quadrant label. Missing wd -> forward-fill the raw sector first.
    wd = df["wd"].ffill().bfill()
    out["wind_quadrant"] = wd.map(QUADRANT).fillna("N")

    out.insert(0, "timestamp", ts.values)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    print(f"[prep] wrote {OUT_CSV}  ({len(out)} rows, {len(NUMERIC)} channels, "
          f"filled {n_nan} rows with NaN)")
    print(f"[prep] wind_quadrant classes: "
          f"{dict(out['wind_quadrant'].value_counts().sort_index())}")


if __name__ == "__main__":
    main()
