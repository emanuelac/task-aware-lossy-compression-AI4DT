"""
compression/dct.py

Time-axis frequency-domain compressor via per-block DCT-II.

For each non-overlapping block of `block_size` rows and each channel
independently, apply the orthonormal DCT-II and keep the first
`n_coeffs` coefficients (the low-frequency approximation). Decompress
by zero-padding the dropped coefficients and applying IDCT.

Pure time-axis method — no PCA on the channel axis; PCA is already
covered by `compression/pca.py`, and keeping the two families
orthogonal makes the Pareto picture readable.

Compression ratio on cells: n_coeffs / block_size (per channel).
bytes_params is zero — DCT has no learned parameters.
"""

from __future__ import annotations

import numpy as np
from scipy.fft import dct, idct

from compression.base import Compressor, CompressedRecord


class DCT(Compressor):
    """
    Block-wise DCT-II compressor.

    Parameters
    ----------
    n_coeffs : int
        Number of low-frequency DCT coefficients retained per (block, channel).
    block_size : int
        Length of each non-overlapping block along the row axis. Default 32.
    """

    def __init__(self, n_coeffs: int, block_size: int = 32):
        if n_coeffs < 1:
            raise ValueError(f"n_coeffs must be >= 1, got {n_coeffs}")
        if block_size < n_coeffs:
            raise ValueError(
                f"block_size ({block_size}) must be >= n_coeffs ({n_coeffs})"
            )
        self.n_coeffs = int(n_coeffs)
        self.block_size = int(block_size)
        self.name = f"dct_k{n_coeffs}"

    # ----- lifecycle -------------------------------------------------------

    def fit(self, x_train: np.ndarray) -> "DCT":
        # No parameters to learn.
        return self

    def compress(self, x: np.ndarray) -> CompressedRecord:
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            x = x[:, None]
        n, c = x.shape
        SW = self.block_size
        K = self.n_coeffs

        # Pad to a whole multiple of SW. The padding cells are reconstructed
        # but then trimmed off in decompress, so they don't affect the output.
        pad = (-n) % SW
        if pad:
            x_padded = np.vstack([x, np.zeros((pad, c), dtype=float)])
        else:
            x_padded = x

        n_blocks = x_padded.shape[0] // SW
        # Reshape to (n_blocks, SW, C) and apply DCT along axis=1.
        blocks = x_padded.reshape(n_blocks, SW, c)
        coeffs = dct(blocks, type=2, norm="ortho", axis=1)[:, :K, :]
        # coeffs shape: (n_blocks, K, C)

        return CompressedRecord(
            z=coeffs.astype(np.float64),
            meta={"orig_len": n, "n_channels": c, "n_blocks": n_blocks},
        )

    def decompress(self, record: CompressedRecord) -> np.ndarray:
        coeffs = record.z
        orig_len = int(record.meta["orig_len"])
        c = int(record.meta["n_channels"])
        n_blocks = int(record.meta["n_blocks"])
        SW = self.block_size
        K = self.n_coeffs

        padded = np.zeros((n_blocks, SW, c), dtype=float)
        padded[:, :K, :] = coeffs
        recon = idct(padded, type=2, norm="ortho", axis=1)
        # Flatten back to (n_blocks * SW, C), trim padding.
        flat = recon.reshape(n_blocks * SW, c)
        return flat[:orig_len]

    # ----- cost ------------------------------------------------------------

    def bytes_params(self) -> int:
        # Pure transform — no trained parameters.
        return 0

    def cell_cr(self, n_channels: int) -> float:
        # Keeps n_coeffs / block_size of the cells per channel.
        return self.block_size / float(self.n_coeffs)
