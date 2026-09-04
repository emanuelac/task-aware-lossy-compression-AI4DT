"""
compression/gorilla.py

Lossless compression using a simplified Gorilla encoder (Pelkonen et al.,
VLDB 2015 — the value-compression half of Facebook's Beringei TSDB and
what InfluxDB / Prometheus use under the hood).

Algorithm (per channel)
-----------------------
1.  First float is written as a 64-bit literal.
2.  For each subsequent float, XOR with the previous bit pattern.
    - If XOR == 0 (identical value): write 1 control bit '0'.       (1 bit)
    - Else: write '1' + 5 bits of leading-zero count + 6 bits of
      significant-bit count + the significant bits.                 (12-70 bits)

We skip the "block reuse" optimisation from the original paper — a 1-bit
shortcut when the current block has the same leading/trailing as the
previous one. That's worth ~10% on stable signals; keeping the code
small is preferred here over chasing every byte.

The decompress contract is exact bit-equality: D(C(x)) == x.

Why Gorilla and not zstd
------------------------
Generic compressors (zstd, lzma) typically beat Gorilla on raw ratio,
but they aren't what production TSDBs deploy. The "is at least one lossy
method better than lossless?" comparison is only meaningful against
the *operational* lossless floor — i.e. Gorilla — not against an
unrealistically tight general-purpose compressor.
"""

from __future__ import annotations

import struct
from typing import List

import numpy as np

from compression.base import Compressor, CompressedRecord


# ---------------------------------------------------------------------------
# Bit-level I/O
# ---------------------------------------------------------------------------

class _BitWriter:
    """MSB-first bit writer onto a bytearray."""

    def __init__(self) -> None:
        self.buf = bytearray()
        self._cur = 0
        self._n = 0

    def write_bits(self, value: int, n: int) -> None:
        for i in range(n - 1, -1, -1):
            self._cur = (self._cur << 1) | ((value >> i) & 1)
            self._n += 1
            if self._n == 8:
                self.buf.append(self._cur)
                self._cur = 0
                self._n = 0

    def finalize(self) -> bytes:
        if self._n:
            self.buf.append(self._cur << (8 - self._n))
            self._cur = 0
            self._n = 0
        return bytes(self.buf)


class _BitReader:
    """MSB-first bit reader over a bytes object."""

    def __init__(self, data: bytes) -> None:
        self.data = data
        self.pos = 0   # bit offset

    def read_bits(self, n: int) -> int:
        v = 0
        for _ in range(n):
            byte = self.data[self.pos >> 3]
            bit = (byte >> (7 - (self.pos & 7))) & 1
            v = (v << 1) | bit
            self.pos += 1
        return v


# ---------------------------------------------------------------------------
# Float ↔ uint64 helpers
# ---------------------------------------------------------------------------

def _f2u(x: float) -> int:
    return struct.unpack(">Q", struct.pack(">d", float(x)))[0]


def _u2f(u: int) -> float:
    return struct.unpack(">d", struct.pack(">Q", u))[0]


def _lz64(u: int) -> int:
    """Leading zeros in a 64-bit int."""
    if u == 0:
        return 64
    n = 0
    for b in range(63, -1, -1):
        if (u >> b) & 1:
            return n
        n += 1
    return 64


def _tz64(u: int) -> int:
    """Trailing zeros in a 64-bit int."""
    if u == 0:
        return 64
    n = 0
    for b in range(64):
        if (u >> b) & 1:
            return n
        n += 1
    return 64


# ---------------------------------------------------------------------------
# Per-channel encode / decode
# ---------------------------------------------------------------------------

def _encode_channel(values: np.ndarray) -> bytes:
    w = _BitWriter()
    n = len(values)
    w.write_bits(n, 32)
    if n == 0:
        return w.finalize()

    prev = _f2u(values[0])
    w.write_bits(prev, 64)

    for i in range(1, n):
        cur = _f2u(values[i])
        xor = prev ^ cur
        if xor == 0:
            w.write_bits(0, 1)
        else:
            lz = min(_lz64(xor), 31)        # 5-bit field
            tz = _tz64(xor)
            sig = 64 - lz - tz
            # sig in [1, 64]; 6-bit field stores sig with 0 meaning 64.
            sig_field = 0 if sig == 64 else sig
            w.write_bits(1, 1)
            w.write_bits(lz, 5)
            w.write_bits(sig_field, 6)
            w.write_bits((xor >> tz) & ((1 << sig) - 1), sig)
        prev = cur

    return w.finalize()


def _decode_channel(data: bytes) -> np.ndarray:
    r = _BitReader(data)
    n = r.read_bits(32)
    out = np.empty(n, dtype=np.float64)
    if n == 0:
        return out

    prev = r.read_bits(64)
    out[0] = _u2f(prev)

    for i in range(1, n):
        if r.read_bits(1) == 0:
            cur = prev
        else:
            lz = r.read_bits(5)
            sig_field = r.read_bits(6)
            sig = 64 if sig_field == 0 else sig_field
            tz = 64 - lz - sig
            v = r.read_bits(sig)
            xor = v << tz
            cur = prev ^ xor
        out[i] = _u2f(cur)
        prev = cur

    return out


# ---------------------------------------------------------------------------
# Compressor interface
# ---------------------------------------------------------------------------

class Gorilla(Compressor):
    """
    Lossless compressor — Gorilla value encoding per channel.

    A window of shape (n_samples, n_channels) is encoded as a list of
    per-channel byte strings. Storage cost (bytes_z) is the actual
    encoded length, not a literal 8*n bytes — that's what makes the
    "is lossy better than lossless?" gate meaningful.
    """

    name = "gorilla"

    def fit(self, x_train: np.ndarray) -> "Gorilla":
        return self  # nothing to learn — pure encoding scheme

    def compress(self, x: np.ndarray) -> CompressedRecord:
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            x = x[:, None]
        encoded: List[bytes] = [
            _encode_channel(x[:, c]) for c in range(x.shape[1])
        ]
        return CompressedRecord(
            z=np.asarray(encoded, dtype=object),
            meta={"orig_len": x.shape[0], "n_channels": x.shape[1]},
        )

    def decompress(self, record: CompressedRecord) -> np.ndarray:
        n_channels = int(record.meta["n_channels"])
        orig_len = int(record.meta["orig_len"])
        out = np.empty((orig_len, n_channels), dtype=float)
        for c in range(n_channels):
            out[:, c] = _decode_channel(record.z[c])
        return out

    def bytes_z(self, record: CompressedRecord) -> int:
        # Sum of per-channel encoded byte strings.
        return int(sum(len(b) for b in record.z))

    def retained_scalars(self, record: CompressedRecord) -> int:
        # Lossless: every value is retained (Gorilla compresses only at
        # the byte level), so the front-end axis records no reduction.
        return int(record.meta["orig_len"]) * int(record.meta["n_channels"])

    def bytes_params(self) -> int:
        return 0
