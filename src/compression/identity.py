"""
compression/identity.py

No-op compressor. Stores the raw window verbatim, decompresses to itself.
Acts as the NMSE floor for the offline evaluation: any lossy method should sit
above identity on the error axis and below it on the storage axis. If a
lossy method beats identity on NMSE, that's a bug — likely target/feature
misalignment.
"""

from __future__ import annotations

import numpy as np

from compression.base import Compressor, CompressedRecord


class Identity(Compressor):
    name = "identity"

    def fit(self, x_train: np.ndarray) -> "Identity":
        return self

    def compress(self, x: np.ndarray) -> CompressedRecord:
        return CompressedRecord(z=np.asarray(x, dtype=float))

    def decompress(self, record: CompressedRecord) -> np.ndarray:
        return record.z

    def bytes_params(self) -> int:
        return 0
