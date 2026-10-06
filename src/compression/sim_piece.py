"""
compression/sim_piece.py

Error-bounded piecewise-linear compressor in the spirit of TerseTS /
Sim-Piece (Liakos et al. VLDB 2022). The greedy core is the classic
"swing-door" segmentation: each new segment is anchored at (i, x[i]) and
the feasible slope cone [lo_slope, hi_slope] is intersected with the slope
constraint from every new point until the cone collapses, at which point
the segment closes and a new one starts at the collapse position.

Per-sample guarantee: |x_hat[j] − x[j]| ≤ eps for every reconstructed
sample (provided eps > 0).

Parameterisation
----------------
The natural knob is the absolute error bound eps. The rest of the
portfolio uses target_cr, so fit() binary-searches eps on the training
segment so the achieved CR (raw_bytes / compressed_bytes) matches
target_cr — keeping the family comparable with dct_cr*, lttb_cr*, etc.

Multivariate handling
---------------------
Each channel is segmented independently with the same fitted eps. Per-
channel segment lists are packed length-prefixed into a single bytes blob
and stored as a uint8 ndarray on the CompressedRecord so that bytes_z is
exact without needing an override.
"""

from __future__ import annotations

import math
import struct
from typing import List, Tuple

import numpy as np

from compression.base import Compressor, CompressedRecord


Segment = Tuple[int, float, float]  # (start_idx, value, slope)


# ---------------------------------------------------------------------------
# Single-channel greedy segmentation (swing door).
# ---------------------------------------------------------------------------
def _swing_door_1d(x: np.ndarray, eps: float) -> List[Segment]:
    """Greedy eps-bounded piecewise-linear segmentation of a 1D signal."""
    n = int(x.shape[0])
    segments: List[Segment] = []
    if n == 0:
        return segments
    if n == 1:
        return [(0, float(x[0]), 0.0)]

    i = 0
    while i < n:
        lo_slope = -math.inf
        hi_slope = math.inf
        x_i = float(x[i])

        j = i + 1
        while j < n:
            dt = j - i
            x_j = float(x[j])
            new_lo = (x_j - x_i - eps) / dt
            new_hi = (x_j - x_i + eps) / dt
            cand_lo = new_lo if new_lo > lo_slope else lo_slope
            cand_hi = new_hi if new_hi < hi_slope else hi_slope
            if cand_lo > cand_hi:
                # Adding j would make the cone empty — close segment
                # before j with the cone from [i, j-1].
                break
            lo_slope = cand_lo
            hi_slope = cand_hi
            j += 1

        # Slope midpoint covers all points in [i, j).
        if math.isinf(lo_slope) and math.isinf(hi_slope):
            slope = 0.0
        elif math.isinf(lo_slope):
            slope = hi_slope
        elif math.isinf(hi_slope):
            slope = lo_slope
        else:
            slope = 0.5 * (lo_slope + hi_slope)
        segments.append((i, x_i, float(slope)))
        i = j

    return segments


def _eval_segments(segments: List[Segment], n: int) -> np.ndarray:
    """Reconstruct a 1D signal of length n from its segment list."""
    out = np.empty(n, dtype=np.float64)
    if not segments:
        out.fill(0.0)
        return out
    for k, (start, value, slope) in enumerate(segments):
        end = segments[k + 1][0] if k + 1 < len(segments) else n
        t = np.arange(end - start, dtype=np.float64)
        out[start:end] = value + slope * t
    return out


# ---------------------------------------------------------------------------
# Per-channel binary packing.
# Header:    uint32 segment count
# Each seg:  int32 start_idx | float32 value | float32 slope  (12 bytes)
# Channel:   uint32 payload length | <header + segments>
# ---------------------------------------------------------------------------
_SEG_STRUCT = struct.Struct("<iff")
_HDR_STRUCT = struct.Struct("<I")
_LEN_STRUCT = struct.Struct("<I")


def _encode_channel(segments: List[Segment]) -> bytes:
    buf = bytearray(_HDR_STRUCT.pack(len(segments)))
    for start, value, slope in segments:
        buf += _SEG_STRUCT.pack(int(start), float(value), float(slope))
    return bytes(buf)


def _decode_channel(blob: bytes, offset: int) -> Tuple[List[Segment], int]:
    (n_seg,) = _HDR_STRUCT.unpack_from(blob, offset)
    offset += _HDR_STRUCT.size
    segments: List[Segment] = []
    for _ in range(n_seg):
        start, value, slope = _SEG_STRUCT.unpack_from(blob, offset)
        segments.append((int(start), float(value), float(slope)))
        offset += _SEG_STRUCT.size
    return segments, offset


def _pack_channels(per_channel: List[List[Segment]]) -> bytes:
    parts = bytearray()
    for segs in per_channel:
        payload = _encode_channel(segs)
        parts += _LEN_STRUCT.pack(len(payload))
        parts += payload
    return bytes(parts)


def _unpack_channels(blob: bytes, n_channels: int) -> List[List[Segment]]:
    offset = 0
    per_channel: List[List[Segment]] = []
    for _ in range(n_channels):
        (length,) = _LEN_STRUCT.unpack_from(blob, offset)
        offset += _LEN_STRUCT.size
        segs, new_offset = _decode_channel(blob, offset)
        per_channel.append(segs)
        offset = new_offset
    return per_channel


# ---------------------------------------------------------------------------
# Compressor.
# ---------------------------------------------------------------------------
class SimPiece(Compressor):
    """
    Error-bounded piecewise-linear compressor (swing-door / Sim-Piece family).

    Two parameterisations are supported (mutually exclusive):

    * `target_cr` — train-fit-CR mode (default; matched-CR with the other
      families): eps is binary-searched on the training segment so
      the achieved CR on training data matches target_cr. Achieved test-CR
      may diverge because segment density is signal-adaptive.
    * `eps_relative` — native error-budget mode (TerseTS / Muñiz-Cuza /
      ModelarDB convention): eps is fixed at `eps_relative · range(train)`
      with no inner search. Achieved CR is whatever falls out per fold.

    Both variants run inside the same sweep so the two parameterisations
    can be compared head-to-head.

    Parameters
    ----------
    target_cr : float, optional
        Target compression ratio. Must be > 1 if provided. Mutually
        exclusive with eps_relative.
    eps_relative : float, optional
        Per-point error tolerance as a fraction of the training segment's
        range (max - min). Must be > 0 if provided. Mutually exclusive
        with target_cr.
    search_iters : int
        Binary-search iterations for eps (target_cr mode only).
    """

    def __init__(
        self,
        target_cr: float | None = None,
        eps_relative: float | None = None,
        search_iters: int = 12,
    ):
        if (target_cr is None) == (eps_relative is None):
            raise ValueError(
                "Specify exactly one of target_cr or eps_relative"
            )
        if target_cr is not None and target_cr <= 1.0:
            raise ValueError(f"target_cr must be > 1, got {target_cr}")
        if eps_relative is not None and eps_relative <= 0:
            raise ValueError(f"eps_relative must be > 0, got {eps_relative}")

        self.target_cr = float(target_cr) if target_cr is not None else None
        self.eps_relative = (
            float(eps_relative) if eps_relative is not None else None
        )
        self.search_iters = int(search_iters)
        if target_cr is not None:
            self.name = f"sim_piece_cr{int(target_cr)}"
        else:
            # Permille suffix keeps names underscore-clean for the family
            # extractor in results.ipynb (method.split('_')[0]).
            self.name = f"sim_piece_eps{int(round(eps_relative * 1000))}"
        self._epsilon: float | None = None

    # ----- helpers ---------------------------------------------------------

    @staticmethod
    def _to_2d(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            x = x[:, None]
        return x

    @staticmethod
    def _frontend_scalars(x: np.ndarray, eps: float) -> int:
        # Front-end retained-scalar count: 3 scalars per segment
        # (start_idx, value, slope), summed over channels. This is the
        # storage axis of the objective — matched with the scalar counts of
        # the other families (a DCT coefficient, a PCA score, etc.), so the
        # CR grid is consistent across families independent of byte packing.
        total = 0
        for ch in range(x.shape[1]):
            total += 3 * len(_swing_door_1d(x[:, ch], eps))
        return total

    def _search_epsilon(self, x_train: np.ndarray) -> float:
        rng = float(x_train.max() - x_train.min())
        if rng <= 0:
            return 1e-9
        lo = math.log(rng * 1e-6)
        hi = math.log(rng * 0.5)
        # Target the FRONT-END compression ratio (retained scalars), the
        # same axis the selection rules use, so sim_piece_cr<r> sits at
        # cell-CR r on the shared grid rather than at its bit-packed byte-CR.
        target_scalars = int(x_train.size) / self.target_cr
        for _ in range(self.search_iters):
            mid = 0.5 * (lo + hi)
            eps = math.exp(mid)
            retained = self._frontend_scalars(x_train, eps)
            if retained > target_scalars:
                lo = mid  # need more compression → larger eps
            else:
                hi = mid  # tighter than needed → smaller eps
        return math.exp(0.5 * (lo + hi))

    # ----- compressor protocol --------------------------------------------

    def fit(self, x_train: np.ndarray) -> "SimPiece":
        x_train = self._to_2d(x_train)
        if self.eps_relative is not None:
            # Native error-budget mode: fix eps = eps_relative * range(train).
            # No CR target, no inner search — the segment count (and thus the
            # achieved CR) is whatever the swing-door algorithm produces.
            rng = float(x_train.max() - x_train.min())
            self._epsilon = self.eps_relative * rng if rng > 0 else 1e-9
        else:
            self._epsilon = self._search_epsilon(x_train)
        return self

    def compress(self, x: np.ndarray) -> CompressedRecord:
        if self._epsilon is None:
            raise RuntimeError("SimPiece must be fitted before compress()")
        x = self._to_2d(x)
        n, c = x.shape
        per_channel = [_swing_door_1d(x[:, ch], self._epsilon) for ch in range(c)]
        blob = _pack_channels(per_channel)
        z = np.frombuffer(blob, dtype=np.uint8).copy()
        return CompressedRecord(
            z=z,
            meta={"orig_len": int(n), "n_channels": int(c)},
        )

    def decompress(self, record: CompressedRecord) -> np.ndarray:
        n = int(record.meta["orig_len"])
        c = int(record.meta["n_channels"])
        blob = record.z.tobytes()
        per_channel = _unpack_channels(blob, c)
        out = np.empty((n, c), dtype=np.float64)
        for ch, segs in enumerate(per_channel):
            out[:, ch] = _eval_segments(segs, n)
        return out

    # ----- cost ------------------------------------------------------------

    def bytes_z(self, record: CompressedRecord) -> int:
        # z is uint8, so record.z.nbytes already counts the true bit-packed
        # payload. Reported for transparency; the objective uses
        # retained_scalars (front-end axis) instead.
        return int(record.z.nbytes)

    def retained_scalars(self, record: CompressedRecord) -> int:
        # 3 scalars per segment (start_idx, value, slope), summed over
        # channels — the front-end storage measure (see base.Compressor).
        c = int(record.meta["n_channels"])
        per_channel = _unpack_channels(record.z.tobytes(), c)
        n_seg = sum(len(segs) for segs in per_channel)
        return 3 * n_seg

    def bytes_params(self) -> int:
        # Just the fitted eps (float64) + the two integer config knobs.
        return 8 + 4 + 4

    def cell_cr(self, n_channels: int) -> float:
        # cell_cr is the fixed-rate CR used only for grid display; SimPiece
        # is variable-rate, so its realised front-end CR (raw / retained
        # scalars) is reported per cell via the `cr` column instead.
        return 1.0
