"""
tasks/_dtw.py

Compiled banded Dynamic Time Warping 1-NN kernel for ClassificationTask.

DTW distance uses a squared-Euclidean local cost summed over channels
(dependent multivariate DTW) with a Sakoe-Chiba band, plus row-level
early abandoning: if the minimum accumulated cost in a row already exceeds
the best-so-far neighbour distance, the alignment cannot beat it (all local
costs are non-negative and the warping path must cross every row), so the
computation is abandoned. The local cost is squared Euclidean, so the 1-NN
argmin is identical to standard (square-root) DTW — the monotone sqrt does
not change the nearest neighbour.

Kept separate from classification.py so numba is imported lazily, only when
classification actually runs.
"""
from __future__ import annotations

import numpy as np
from numba import njit, prange


@njit(cache=True, fastmath=True)
def _dtw_ea(a, b, band, best):
    """Banded squared-Euclidean DTW between (L, C) series a, b, with row-level
    early abandoning against `best`. Returns 1e18 if abandoned."""
    L = a.shape[0]
    C = a.shape[1]
    INF = 1e18
    prev = np.full(L, INF)
    curr = np.full(L, INF)
    for i in range(L):
        j0 = max(0, i - band)
        j1 = min(L, i + band + 1)
        rmin = INF
        for j in range(j0, j1):
            c = 0.0
            for k in range(C):
                d = a[i, k] - b[j, k]
                c += d * d
            if i == 0 and j == 0:
                acc = c
            else:
                m = INF
                if i > 0 and prev[j] < m:
                    m = prev[j]
                if j > 0 and curr[j - 1] < m:
                    m = curr[j - 1]
                if i > 0 and j > 0 and prev[j - 1] < m:
                    m = prev[j - 1]
                acc = c + m
            curr[j] = acc
            if acc < rmin:
                rmin = acc
        if rmin >= best:
            return INF
        for j in range(L):
            prev[j] = curr[j]
            curr[j] = INF
    return prev[L - 1]


@njit(cache=True, fastmath=True, parallel=True)
def dtw_1nn_predict(Xq, Xr, yr, band):
    """1-NN-DTW labels for query set Xq (nq, L, C) against reference set
    Xr (nr, L, C) with integer labels yr (nr,). Returns (nq,) int labels."""
    nq = Xq.shape[0]
    nr = Xr.shape[0]
    out = np.empty(nq, np.int64)
    for qi in prange(nq):
        best = 1e18
        bl = -1
        for ri in range(nr):
            d = _dtw_ea(Xq[qi], Xr[ri], band, best)
            if d < best:
                best = d
                bl = yr[ri]
        out[qi] = bl
    return out
