"""
compression/pca.py

Column-axis compression via PCA on the channel dimension.

Given a multivariate series x of shape (n_samples, n_channels), fit a PCA
on the training segment that keeps K components (either fixed K or chosen
to explain a fraction of variance), then per window project x onto the K
components for storage and reconstruct by inverse projection.

Compression ratio in cells: keeps K / n_channels of the columns, all rows.
Distinct compression axis from Downsample, so the two families are not
directly comparable on a single "ratio" knob — both feed into the same
storage-vs-error Pareto.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.decomposition import PCA as SkPCA

from compression.base import Compressor, CompressedRecord


class PCA(Compressor):
    """
    PCA on the channel axis. Stores K components + per-channel mean as
    frozen parameters; per-window payload is the (n_samples, K) score
    matrix.
    """

    def __init__(
        self,
        n_components: Optional[int] = None,
        variance_explained: Optional[float] = None,
        target_cr: Optional[float] = None,
    ):
        n_set = sum(x is not None for x in (n_components, variance_explained, target_cr))
        if n_set != 1:
            raise ValueError(
                "Specify exactly one of n_components, variance_explained, target_cr"
            )
        self.n_components = n_components
        self.variance_explained = variance_explained
        self.target_cr = target_cr
        self._pca: Optional[SkPCA] = None
        self._K: Optional[int] = None
        if n_components is not None:
            self.name = f"pca_k{n_components}"
        elif variance_explained is not None:
            self.name = f"pca_v{int(round(variance_explained * 100))}"
        else:
            # CR ≈ C/K for large n, so K = round(C / target_cr) at fit time.
            # Name uses the target, not the realised CR, so it's stable across
            # datasets.
            self.name = f"pca_cr{int(round(target_cr))}"

    # ----- lifecycle -------------------------------------------------------

    def fit(self, x_train: np.ndarray) -> "PCA":
        x_train = np.asarray(x_train, dtype=float)
        if x_train.ndim != 2:
            raise ValueError(f"x_train must be 2D, got shape {x_train.shape}")

        C = x_train.shape[1]
        if self.n_components is not None:
            K = min(self.n_components, C)
        elif self.variance_explained is not None:
            full = SkPCA(n_components=C)
            full.fit(x_train)
            cum = np.cumsum(full.explained_variance_ratio_)
            K = int(np.searchsorted(cum, self.variance_explained) + 1)
            K = max(1, min(K, C))
        else:
            # target_cr path. CR ≈ C / K for large n, so K ≈ C / target_cr.
            K = max(1, min(C, int(round(C / self.target_cr))))
        self._pca = SkPCA(n_components=K)
        self._pca.fit(x_train)

        self._K = self._pca.n_components_
        return self

    def compress(self, x: np.ndarray) -> CompressedRecord:
        if self._pca is None:
            raise RuntimeError("PCA must be fit before compress")
        x = np.asarray(x, dtype=float)
        z = self._pca.transform(x)
        return CompressedRecord(z=z, meta={"orig_len": x.shape[0]})

    def decompress(self, record: CompressedRecord) -> np.ndarray:
        if self._pca is None:
            raise RuntimeError("PCA must be fit before decompress")
        return self._pca.inverse_transform(record.z)

    # ----- cost ------------------------------------------------------------

    def bytes_params(self) -> int:
        if self._pca is None:
            return 0
        # Frozen params: components_ (K, D) and mean_ (D,), both float64.
        return int(self._pca.components_.nbytes + self._pca.mean_.nbytes)

    def cell_cr(self, n_channels: int) -> float:
        # Keeps K / n_channels of the columns, all rows. Use the fitted K
        # when available; fall back to n_components / n_channels otherwise.
        K = self._K if self._K is not None else self.n_components
        if K is None or K <= 0:
            return 1.0
        return n_channels / float(K)
