"""
scripts/renormalise_summary.py

Recompute the derived columns (loss_tilde, storage_tilde, multi_obj) of
existing results/<dataset>/<stamp>/summary.csv files with the
canonical per-cell normalisation of src/normalisation.py, without
re-running compression or task evaluation. The raw measurement columns
(loss, nmse, nrmse, accuracy, storage, bytes_z, ...) are untouched.

Use this to refresh existing summaries in place after any change to the
normalisation pool or the α convention, so their derived columns match
the canonical implementation in src/normalisation.py without re-running
compression or task evaluation.

Usage
-----
    python3 scripts/renormalise_summary.py                # all summaries
    python3 scripts/renormalise_summary.py uci_har metro  # subset
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import config as cfg
from normalisation import normalise_summary


def main() -> None:
    datasets = sys.argv[1:] or None
    results = REPO_ROOT / "results"
    paths = sorted(results.glob("*/*/summary.csv"))
    if datasets:
        paths = [p for p in paths if p.parent.parent.name in datasets]
    if not paths:
        print("[renorm] no summary.csv files found")
        return
    for p in paths:
        df = pd.read_csv(p)
        n_before = len(df)
        df = normalise_summary(df, alpha=cfg.ALPHA)
        assert len(df) == n_before, f"row count changed for {p}"
        df.to_csv(p, index=False)
        print(f"[renorm] {p.relative_to(REPO_ROOT)}  ({n_before} rows, "
              f"alpha={cfg.ALPHA})")


if __name__ == "__main__":
    main()
