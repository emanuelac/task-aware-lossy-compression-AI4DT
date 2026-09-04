"""
scripts/prep_hydraulic.py

Prep the UCI "Condition Monitoring of Hydraulic Systems" dataset (ZeMA /
Helwig et al. 2015) into the single-stream CSV the offline pipeline expects.

Why this dataset
----------------
A hydraulic test rig repeats a constant 60-second load cycle 2205 times
while the condition of four components (cooler, valve, pump, accumulator)
is varied. This is the industrial digital-twin instance the portfolio was
missing after IMS was dropped — and, crucially, it avoids IMS's
time-confounding: the *valve* condition is varied on an interleaved
schedule (145 runs across the 2205 cycles), so all four valve classes
appear in every block-disjoint CV fold. Classification is therefore a
recurring-operating-mode problem, not a monotone run-to-failure.

Stream construction
-------------------
The raw files store one matrix per sensor (rows = cycles, cols = the
within-cycle samples at that sensor's rate). We use the eight 1 Hz
sensors — 60 samples per cycle — and concatenate the cycles into one
continuous 1 Hz stream of 2205 x 60 = 132 300 rows. The per-cycle valve
label is expanded to all 60 samples of its cycle. Because every cycle is
a repeat of the same load profile, the concatenated stream is
quasi-periodic (period 60 s) and genuinely forecastable, with only mild
seams at cycle boundaries.

Sensors kept (all 1 Hz, aligned at 60 samples/cycle):
    TS1..TS4  temperatures (deg C)
    VS1       vibration (mm/s)
    CE        cooling efficiency (virtual, %)
    CP        cooling power (virtual, kW)
    SE        efficiency factor (%)

Run (after downloading the archive into
data/raw/condition+monitoring+of+hydraulic+systems/):
    python3 scripts/prep_hydraulic.py
Produces data/raw/hydraulic.csv with the column layout used by
src/datasets/hydraulic.py.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "data" / "raw" / "condition+monitoring+of+hydraulic+systems"
OUT_CSV = REPO_ROOT / "data" / "raw" / "hydraulic.csv"

# 1 Hz sensors → 60 samples per cycle, all aligned.
CHANNELS = ["TS1", "TS2", "TS3", "TS4", "VS1", "CE", "CP", "SE"]
SAMPLES_PER_CYCLE = 60
# profile.txt columns: 0 cooler, 1 valve, 2 pump, 3 accumulator, 4 stable.
VALVE_COL = 1


def main() -> None:
    if not SRC_DIR.exists():
        raise SystemExit(f"[prep] source dir not found: {SRC_DIR}")

    # Load each sensor matrix (2205 x 60) and flatten row-major into a
    # single continuous stream (cycle 0 samples 0..59, cycle 1, ...).
    columns = {}
    n_cycles = None
    for ch in CHANNELS:
        arr = np.loadtxt(SRC_DIR / f"{ch}.txt")
        if arr.shape[1] != SAMPLES_PER_CYCLE:
            raise SystemExit(
                f"[prep] {ch}: expected {SAMPLES_PER_CYCLE} cols, got {arr.shape[1]}"
            )
        if n_cycles is None:
            n_cycles = arr.shape[0]
        elif arr.shape[0] != n_cycles:
            raise SystemExit(f"[prep] {ch}: cycle count {arr.shape[0]} != {n_cycles}")
        columns[ch] = arr.reshape(-1)  # (n_cycles * 60,)

    # Per-cycle valve label → per-sample by repeating each cycle's label.
    profile = np.loadtxt(SRC_DIR / "profile.txt")
    valve = profile[:, VALVE_COL].astype(int)
    if valve.shape[0] != n_cycles:
        raise SystemExit(f"[prep] profile rows {valve.shape[0]} != cycles {n_cycles}")
    valve_per_sample = np.repeat(valve, SAMPLES_PER_CYCLE)

    n = n_cycles * SAMPLES_PER_CYCLE
    df = pd.DataFrame(columns)
    df["valve"] = valve_per_sample
    # Synthetic monotonic 1 Hz timestamp (the rig samples at 1 Hz; the
    # absolute wall-clock is not part of the release).
    df.insert(0, "timestamp", pd.date_range("2015-01-01", periods=n, freq="1s"))

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"[prep] wrote {OUT_CSV}  ({n} rows, {len(CHANNELS)} channels)")
    print(f"[prep] valve classes: {dict(pd.Series(valve).value_counts().sort_index())}")


if __name__ == "__main__":
    main()
