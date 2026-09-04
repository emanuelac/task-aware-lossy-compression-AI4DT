"""
compression/base.py

Compressor protocol shared by all offline compressors.

A CompressedRecord is the per-window output of compress(): the payload z
plus whatever metadata decompress() needs to rebuild the same-shape x_hat.
We pass it explicitly (rather than tucking metadata onto the compressor)
because the compressor's parameters are *frozen* and shared across all
windows — only the record changes per window.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

import numpy as np


@dataclass
class CompressedRecord:
    """Per-window compressed payload + metadata for reconstruction."""
    z: np.ndarray                                  # the stored representation
    meta: Dict[str, Any] = field(default_factory=dict)


class Compressor:
    """
    Base class for offline compressors.

    Subclasses must implement: fit, compress, decompress, bytes_params.
    bytes_z has a sensible default (z.nbytes); override only if the on-disk
    encoding is smaller than a raw float64 array (e.g. quantised or
    variable-length encoded payloads).
    """

    name: str = "compressor"

    # ----- lifecycle -------------------------------------------------------

    def fit(self, x_train: np.ndarray) -> "Compressor":
        """Learn parameters from a (n_samples, n_channels) training array."""
        raise NotImplementedError

    def compress(self, x: np.ndarray) -> CompressedRecord:
        """Compress an (n_samples, n_channels) window. Returns z + meta."""
        raise NotImplementedError

    def decompress(self, record: CompressedRecord) -> np.ndarray:
        """Reconstruct an (n_samples, n_channels) array from a record."""
        raise NotImplementedError

    # ----- cost ------------------------------------------------------------

    def bytes_z(self, record: CompressedRecord) -> int:
        """Payload bytes for a single record. Default: dense float64.

        This is the *real* encoded size of the payload (bit-packed for the
        variable-length families). It is reported for transparency but is
        NOT the storage axis of the objective — see `retained_scalars`.
        """
        return int(record.z.nbytes)

    def retained_scalars(self, record: CompressedRecord) -> int:
        """
        Number of scalar values the family retains for this record — the
        *front-end* (structure-level) storage measure, independent of any
        byte-encoding back end (quantisation, entropy/bit-packing).

        This is the storage axis of the multi-objective loss: it charges
        every family for how much information it keeps, at a common
        precision, so no family is flattered by having a bit-packed back
        end while another is stored as raw float64. The default is the
        size of the dense float64 payload `z`; families that keep side
        information (LTTB indices), a bit-packed payload (SimPiece,
        Gorilla), or that are lossless override this.
        """
        return int(record.z.size)

    def bytes_params(self) -> int:
        """Frozen parameter bytes — amortized over all windows in the offline storage accounting."""
        raise NotImplementedError

    def cell_cr(self, n_channels: int) -> float:
        """
        Compression ratio on the *cell* axis (raw cells / kept cells).

        Independent of byte encoding and parameter overhead, so the matched-CR
        analysis can be done with one groupby on `cr_cells` regardless of
        family. Default 1.0 (no cell reduction) — Identity / Gorilla inherit it.
        """
        return 1.0

    # ----- persistence -----------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Pickle the fitted compressor to disk."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str | Path) -> "Compressor":
        with open(path, "rb") as f:
            return pickle.load(f)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
