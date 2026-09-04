"""
compression/

Offline-trained compressors with persisted parameters.

Each compressor has the same lifecycle:
    fit(x_train)             — learn parameters on a training segment
    compress(x)   -> record  — produce the stored representation z (+ metadata)
    decompress(record) -> x_hat — reconstruct the same-shape signal
    bytes_z(record) -> int   — payload size per window
    bytes_params() -> int    — frozen parameter size (amortized later)
    save(path) / load(path)  — persist params to disk

The forecaster always sees x_hat = D(C(x)), never the compressed z directly.
Compressors are therefore compared by the downstream cost they induce under a
shared forecaster (one β per horizon), not by reconstruction error alone.
"""

from compression.ae import AE
from compression.base import Compressor, CompressedRecord
from compression.dct import DCT
from compression.downsample import Downsample
from compression.dwt import DWT
from compression.gorilla import Gorilla
from compression.identity import Identity
from compression.lttb import LTTB
from compression.pca import PCA
from compression.sim_piece import SimPiece

__all__ = [
    "AE",
    "Compressor",
    "CompressedRecord",
    "DCT",
    "Downsample",
    "DWT",
    "Gorilla",
    "Identity",
    "LTTB",
    "PCA",
    "SimPiece",
]
