"""
compression/downsample.py

Uniform downsampling: keep every Nth raw sample, then reconstruct by
linear interpolation back to the original cadence.

This is the simplest possible row-axis compressor. Its parameters are
just (ratio,) so bytes_params is effectively zero — useful as one end of
the storage/error tradeoff axis.

Compression ratio in cells: keeps 1/ratio of the rows, all channels.
Reconstruction error grows with ratio and with how "spiky" the signal is.
"""

from __future__ import annotations

import numpy as np

from compression.base import Compressor, CompressedRecord


class Downsample(Compressor):
    """
    Uniform downsampling by integer factor.

    compress(x): z = x[::ratio]  (plus original length stored in meta so
                 decompress knows the target shape)
    decompress(record): linear interpolation of z back to length orig_len
                 along the row axis, per channel.
    """

    def __init__(self, ratio: int):
        if ratio < 1:
            raise ValueError(f"ratio must be >= 1, got {ratio}")
        self.ratio = int(ratio)
        self.name = f"downsample_{ratio}"

    # ----- lifecycle -------------------------------------------------------

    def fit(self, x_train: np.ndarray) -> "Downsample":
        # No parameters to learn — ratio is hand-picked.
        return self

    def compress(self, x: np.ndarray) -> CompressedRecord:
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            x = x[:, None]
        z = x[:: self.ratio]
        return CompressedRecord(
            z=z,
            meta={"orig_len": x.shape[0], "n_channels": x.shape[1]},
        )

    def decompress(self, record: CompressedRecord) -> np.ndarray:
        z = record.z
        orig_len = int(record.meta["orig_len"])
        n_channels = int(record.meta["n_channels"])
        if z.ndim == 1:
            z = z[:, None]

        # Sample indices where z was taken from the original timeline.
        src_idx = np.arange(0, orig_len, self.ratio)
        # Guard: when ratio doesn't evenly divide orig_len, the final stored
        # row corresponds to src_idx[-1] (not orig_len - 1). interp clamps
        # values past src_idx[-1] to the last sample — fine for our use.
        target_idx = np.arange(orig_len)

        x_hat = np.empty((orig_len, n_channels), dtype=float)
        for c in range(n_channels):
            x_hat[:, c] = np.interp(target_idx, src_idx, z[:, c])
        return x_hat

    # ----- cost ------------------------------------------------------------

    def bytes_params(self) -> int:
        # Just the ratio int — negligible. Round up to one int64.
        return 8

    def cell_cr(self, n_channels: int) -> float:
        # Keeps 1/ratio of the rows, all channels.
        return float(self.ratio)
