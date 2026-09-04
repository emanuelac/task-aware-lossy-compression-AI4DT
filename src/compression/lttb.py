"""
compression/lttb.py

Largest Triangle Three Buckets — adaptive row-axis decimation.

LTTB partitions the time axis into B buckets and within each bucket
keeps the single sample whose triangle area, formed with the previously
selected point and the next bucket's centroid, is largest. The first
and last samples are always kept. Compared to uniform Downsample at the
same row-axis ratio, LTTB preserves local extrema, so spiky / peaky
channels reconstruct better.

Multivariate handling
---------------------
We run a *single* index selection shared across channels: the bucket
score is the sum of per-channel triangle areas. This keeps storage in
the same shape as Downsample — (n_kept, C) values + (n_kept,) indices —
and makes the matched-cr_cells comparison fair (same number of kept
cells in both methods).

Compression ratio in cells: cr_cells ≈ ratio (kept rows = orig_len / ratio).
"""

from __future__ import annotations

import numpy as np

from compression.base import Compressor, CompressedRecord


class LTTB(Compressor):
    """
    LTTB decimation by integer factor.

    Parameters
    ----------
    ratio : int
        Target row-axis compression ratio. Number of kept rows ≈ orig_len / ratio.
    """

    def __init__(self, ratio: int):
        if ratio < 1:
            raise ValueError(f"ratio must be >= 1, got {ratio}")
        self.ratio = int(ratio)
        self.name = f"lttb_{ratio}"

    # ----- lifecycle -------------------------------------------------------

    def fit(self, x_train: np.ndarray) -> "LTTB":
        # No parameters to learn.
        return self

    # ----- core algorithm --------------------------------------------------

    @staticmethod
    def _lttb_indices(x: np.ndarray, n_out: int) -> np.ndarray:
        """
        Select n_out row indices from a (n, C) array using LTTB with the
        bucket score summed across channels.
        """
        n, c = x.shape
        if n_out >= n:
            return np.arange(n, dtype=np.int64)
        if n_out < 3:
            return np.array([0, n - 1], dtype=np.int64)

        bucket_size = (n - 2) / (n_out - 2)
        out = np.empty(n_out, dtype=np.int64)
        out[0] = 0
        a = 0  # previously selected index

        for i in range(n_out - 2):
            start = int(np.floor(i * bucket_size)) + 1
            end = int(np.floor((i + 1) * bucket_size)) + 1
            end = min(end, n - 1)

            next_start = end
            next_end = int(np.floor((i + 2) * bucket_size)) + 1
            next_end = min(next_end, n)

            # Next-bucket centroid: (avg_x, avg_y_per_channel)
            avg_x = (next_start + next_end - 1) / 2.0
            avg_y = x[next_start:next_end].mean(axis=0)        # (C,)

            # For each candidate j in [start, end), triangle area summed
            # over channels with apex (a, x[a]) and base midpoint
            # (avg_x, avg_y). Vectorised across j.
            j = np.arange(start, end)
            dx_aj = (a - j).astype(float)                       # (B,)
            dx_avg = a - avg_x                                  # scalar
            dy_avg = avg_y - x[a]                               # (C,)
            dy_aj = x[start:end] - x[a]                         # (B, C)

            # |dx_avg * dy_aj - dx_aj[:,None] * dy_avg[None,:]| summed over C
            area = np.abs(
                dx_avg * dy_aj - dx_aj[:, None] * dy_avg[None, :]
            ).sum(axis=1) * 0.5

            best = start + int(np.argmax(area))
            out[i + 1] = best
            a = best

        out[-1] = n - 1
        return out

    # ----- compressor protocol --------------------------------------------

    def compress(self, x: np.ndarray) -> CompressedRecord:
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            x = x[:, None]
        n, c = x.shape

        n_out = max(2, n // self.ratio)
        idx = self._lttb_indices(x, n_out)
        z = x[idx]                                             # (n_kept, C)

        return CompressedRecord(
            z=z,
            meta={"orig_len": n, "n_channels": c, "idx": idx},
        )

    def decompress(self, record: CompressedRecord) -> np.ndarray:
        z = record.z
        orig_len = int(record.meta["orig_len"])
        n_channels = int(record.meta["n_channels"])
        idx = np.asarray(record.meta["idx"])
        if z.ndim == 1:
            z = z[:, None]

        target_idx = np.arange(orig_len)
        x_hat = np.empty((orig_len, n_channels), dtype=float)
        for ch in range(n_channels):
            x_hat[:, ch] = np.interp(target_idx, idx, z[:, ch])
        return x_hat

    # ----- cost ------------------------------------------------------------

    def bytes_z(self, record: CompressedRecord) -> int:
        # Stored: values (float64) + selected indices (int64).
        return int(record.z.nbytes + record.meta["idx"].nbytes)

    def retained_scalars(self, record: CompressedRecord) -> int:
        # Front-end count: the kept values PLUS the indices of the kept
        # rows — LTTB must store positions because its samples are chosen
        # non-uniformly (unlike Downsample's implicit stride).
        return int(record.z.size + np.asarray(record.meta["idx"]).size)

    def bytes_params(self) -> int:
        # Just the ratio int.
        return 8

    def cell_cr(self, n_channels: int) -> float:
        # Keeps ~1/ratio of the rows, all channels.
        return float(self.ratio)
