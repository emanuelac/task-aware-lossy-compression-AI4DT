"""
compression/dwt.py

Time-axis multi-resolution compressor via per-block discrete wavelet
transform.

For each non-overlapping block of `block_size` rows and each channel
independently, run a multi-level DWT and keep only the deepest
approximation coefficients (n_coeffs of them). Detail coefficients are
zeroed and the inverse DWT reconstructs a smoothed signal. This is the
direct multi-resolution analog of `compression/dct.py`: same matched
cell-CR = block_size / n_coeffs.

Wavelet choice is `db4` by default — a good general-purpose Daubechies
wavelet with 4 vanishing moments. The block_size / n_coeffs ratio must
be a power of two; this is checked at construction time.

bytes_params is zero — DWT has no learned parameters.
"""

from __future__ import annotations

import math

import numpy as np
import pywt

from compression.base import Compressor, CompressedRecord


class DWT(Compressor):
    """
    Block-wise discrete wavelet transform compressor.

    Parameters
    ----------
    n_coeffs : int
        Number of approximation coefficients retained per (block, channel).
        Must satisfy block_size / n_coeffs == 2**L for some integer L >= 1.
    block_size : int
        Length of each non-overlapping block along the row axis. Default 32.
    wavelet : str
        pywt wavelet name. Default "db4".
    """

    def __init__(self, n_coeffs: int, block_size: int = 32, wavelet: str = "db4"):
        if n_coeffs < 1:
            raise ValueError(f"n_coeffs must be >= 1, got {n_coeffs}")
        if block_size < n_coeffs:
            raise ValueError(
                f"block_size ({block_size}) must be >= n_coeffs ({n_coeffs})"
            )
        ratio = block_size / n_coeffs
        if not float(ratio).is_integer() or not (int(ratio) & (int(ratio) - 1) == 0):
            raise ValueError(
                f"block_size / n_coeffs must be a power of two, got {ratio}"
            )
        self.n_coeffs = int(n_coeffs)
        self.block_size = int(block_size)
        self.wavelet = wavelet
        self.level = int(math.log2(int(ratio)))
        self.name = f"dwt_k{n_coeffs}"

    # ----- lifecycle -------------------------------------------------------

    def fit(self, x_train: np.ndarray) -> "DWT":
        # No parameters to learn.
        return self

    def _wavedec_block(self, block: np.ndarray) -> list[np.ndarray]:
        """
        Wavelet decomposition per channel along axis=0 of a (SW, C) block.
        Returns the list of coefficient arrays [cA_L, cD_L, ..., cD_1],
        each of shape (len_k, C).
        """
        return pywt.wavedec(block, wavelet=self.wavelet, level=self.level, axis=0,
                            mode="periodization")

    def compress(self, x: np.ndarray) -> CompressedRecord:
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            x = x[:, None]
        n, c = x.shape
        SW = self.block_size
        K = self.n_coeffs

        # Pad to whole block multiples.
        pad = (-n) % SW
        if pad:
            x_padded = np.vstack([x, np.zeros((pad, c), dtype=float)])
        else:
            x_padded = x
        n_blocks = x_padded.shape[0] // SW

        # Process all blocks in one call: reshape to (n_blocks, SW, C) and
        # decompose along axis=1.
        blocks = x_padded.reshape(n_blocks, SW, c)
        coeffs = pywt.wavedec(
            blocks, wavelet=self.wavelet, level=self.level, axis=1,
            mode="periodization",
        )
        # coeffs[0] is the approximation: (n_blocks, K, C) under periodization.
        cA = coeffs[0]
        if cA.shape[1] != K:
            raise RuntimeError(
                f"DWT approximation length {cA.shape[1]} != expected {K}; "
                f"check (block_size, n_coeffs, wavelet) compatibility"
            )

        return CompressedRecord(
            z=cA.astype(np.float64),
            meta={"orig_len": n, "n_channels": c, "n_blocks": n_blocks},
        )

    def decompress(self, record: CompressedRecord) -> np.ndarray:
        cA = record.z                                          # (n_blocks, K, C)
        orig_len = int(record.meta["orig_len"])
        c = int(record.meta["n_channels"])
        n_blocks = int(record.meta["n_blocks"])
        SW = self.block_size
        K = self.n_coeffs

        # Rebuild detail-zeroed coefficient list: cA followed by L zero arrays
        # whose shapes match what wavedec would have returned.
        zero_coeffs: list[np.ndarray] = [cA]
        len_k = K
        for _ in range(self.level):
            zero_coeffs.append(np.zeros((n_blocks, len_k, c), dtype=float))
            len_k *= 2

        recon = pywt.waverec(zero_coeffs, wavelet=self.wavelet, axis=1,
                             mode="periodization")
        # Periodization keeps lengths exact; safety trim in case of rounding.
        recon = recon[:, :SW, :]
        flat = recon.reshape(n_blocks * SW, c)
        return flat[:orig_len]

    # ----- cost ------------------------------------------------------------

    def bytes_params(self) -> int:
        return 0

    def cell_cr(self, n_channels: int) -> float:
        # Keeps n_coeffs / block_size of the cells per channel.
        return self.block_size / float(self.n_coeffs)
