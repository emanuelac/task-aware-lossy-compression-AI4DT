"""
compression/ae.py

Offline-trained MLP autoencoder for offline evaluation.

Differences from the streaming online AE (`streaming/src/methods/ae.py`)
-----------------------------------------------------
- Trained once on `raw_train` and frozen. No online updates.
- Wrapped in the `Compressor` protocol: compress / decompress operate on
  whole-segment arrays; internal block_size is a method hyperparameter
  (default 32) that the user never has to think about.
- bytes_params reports the actual frozen weight footprint, so the AE
  shows up correctly on the storage axis once its parameter cost is
  amortised against compressed payload.

Architecture per block (SW × C):
    flatten → (SW·C,) → ReLU(W1) → bottleneck d → ReLU(W3) → (SW·C,)

Inputs are per-channel standardised using statistics computed on
raw_train; the (mu, sigma) vectors are persisted as part of the
compressor.

Training is mini-batch Adam with a cosine LR schedule and early stopping
on a held-out validation split of the training segment. 
"""

from __future__ import annotations

import numpy as np

from compression.base import Compressor, CompressedRecord


# ---------------------------------------------------------------------------
# Minimal numpy MLP autoencoder (offline training only) — Adam + cosine LR
# ---------------------------------------------------------------------------

class _MLPAE:
    """Two-layer MLP autoencoder, fully vectorised numpy. Trained with Adam."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        bottleneck_dim: int,
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        seed: int = 42,
    ):
        rng = np.random.default_rng(seed)

        # Kaiming for ReLU-feeding layers, Xavier for linear-output layers.
        self.W1 = rng.normal(0, np.sqrt(2.0 / input_dim),     (hidden_dim, input_dim))
        self.b1 = np.zeros(hidden_dim)
        self.W2 = rng.normal(0, np.sqrt(1.0 / hidden_dim),    (bottleneck_dim, hidden_dim))
        self.b2 = np.zeros(bottleneck_dim)
        self.W3 = rng.normal(0, np.sqrt(2.0 / bottleneck_dim),(hidden_dim, bottleneck_dim))
        self.b3 = np.zeros(hidden_dim)
        self.W4 = rng.normal(0, np.sqrt(1.0 / hidden_dim),    (input_dim, hidden_dim))
        self.b4 = np.zeros(input_dim)

        # Adam hyperparameters and state.
        self.lr = float(lr)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.eps = float(eps)
        self.weight_decay = float(weight_decay)

        # First and second moment buffers, one per parameter tensor.
        self._params = [
            "W1", "b1", "W2", "b2", "W3", "b3", "W4", "b4",
        ]
        self._m = {p: np.zeros_like(getattr(self, p)) for p in self._params}
        self._v = {p: np.zeros_like(getattr(self, p)) for p in self._params}
        self._t = 0  # Adam timestep

    @staticmethod
    def _relu(x):
        return np.maximum(0.0, x)

    @staticmethod
    def _relu_grad(x):
        return (x > 0.0).astype(float)

    # ----- batched forward / backward -------------------------------------

    def encode_batch(self, X: np.ndarray) -> np.ndarray:
        h1 = self._relu(X @ self.W1.T + self.b1)
        return h1 @ self.W2.T + self.b2

    def decode_batch(self, Z: np.ndarray) -> np.ndarray:
        h3 = self._relu(Z @ self.W3.T + self.b3)
        return h3 @ self.W4.T + self.b4

    def forward_loss(self, X: np.ndarray) -> float:
        """Eval-only forward — no parameter update."""
        z = self.encode_batch(X)
        out = self.decode_batch(z)
        return float(np.mean((out - X) ** 2))

    def update_batch(self, X: np.ndarray, lr: float | None = None) -> float:
        n, d_in = X.shape

        pre1 = X @ self.W1.T + self.b1
        h1   = self._relu(pre1)
        z    = h1 @ self.W2.T + self.b2
        pre3 = z @ self.W3.T + self.b3
        h3   = self._relu(pre3)
        out  = h3 @ self.W4.T + self.b4

        loss = float(np.mean((out - X) ** 2))

        # Backward pass.
        d_out = 2.0 * (out - X) / (n * d_in)
        d_W4  = d_out.T @ h3
        d_b4  = d_out.sum(axis=0)
        d_h3  = (d_out @ self.W4) * self._relu_grad(pre3)

        d_W3  = d_h3.T @ z
        d_b3  = d_h3.sum(axis=0)
        d_z   = d_h3 @ self.W3

        d_h1  = (d_z @ self.W2) * self._relu_grad(pre1)
        d_W2  = d_z.T @ h1
        d_b2  = d_z.sum(axis=0)
        d_W1  = d_h1.T @ X
        d_b1  = d_h1.sum(axis=0)

        grads = {
            "W1": d_W1, "b1": d_b1, "W2": d_W2, "b2": d_b2,
            "W3": d_W3, "b3": d_b3, "W4": d_W4, "b4": d_b4,
        }

        # Global-norm gradient clipping — same as the previous trainer, kept
        # to defang the occasional Adam spike at the start of training.
        total_norm = np.sqrt(sum(np.sum(g ** 2) for g in grads.values()))
        if total_norm > 5.0:
            scale = 5.0 / total_norm
            for k in grads:
                grads[k] *= scale

        # Adam update with bias correction. Optional decoupled weight decay.
        self._t += 1
        b1_t = 1.0 - self.beta1 ** self._t
        b2_t = 1.0 - self.beta2 ** self._t
        step_lr = float(self.lr if lr is None else lr)

        for p, g in grads.items():
            if self.weight_decay > 0.0 and p.startswith("W"):
                # Decoupled weight decay (AdamW-style): applied directly to
                # the parameter, never picked up by the first/second moment.
                setattr(self, p, getattr(self, p) * (1.0 - step_lr * self.weight_decay))

            self._m[p] = self.beta1 * self._m[p] + (1.0 - self.beta1) * g
            self._v[p] = self.beta2 * self._v[p] + (1.0 - self.beta2) * (g * g)
            m_hat = self._m[p] / b1_t
            v_hat = self._v[p] / b2_t
            update = step_lr * m_hat / (np.sqrt(v_hat) + self.eps)
            setattr(self, p, getattr(self, p) - update)

        return loss

    def snapshot(self) -> dict:
        """Return a deep copy of the parameter tensors for best-val rollback."""
        return {p: getattr(self, p).copy() for p in self._params}

    def restore(self, snapshot: dict) -> None:
        for p, arr in snapshot.items():
            setattr(self, p, arr.copy())

    def total_param_bytes(self) -> int:
        return int(sum(getattr(self, p).nbytes for p in self._params))


# ---------------------------------------------------------------------------
# AE compressor
# ---------------------------------------------------------------------------

class AE(Compressor):
    """
    Offline-trained MLP autoencoder compressor.

    Parameters
    ----------
    bottleneck_dim : int
        Dimensionality of the bottleneck vector per block.
    block_size : int
        Number of rows per non-overlapping block, default 32.
    n_epochs : int
        Maximum training epochs over raw_train. Default 300 — early
        stopping usually terminates well before this on small datasets.
    batch_size : int
        Mini-batch size for Adam. Default 64.
    lr : float
        Adam learning rate (peak; decays under the cosine schedule).
        Default 1e-3.
    hidden_dim : int | None
        Override the auto-sized hidden layer. If None, sized as
        `max(2·bottleneck_dim, input_dim // 2)` capped at 256.
    weight_decay : float
        Decoupled weight-decay coefficient (AdamW-style). Default 0.0.
    val_fraction : float
        Fraction of training blocks held out for early stopping.
        Default 0.1.
    patience : int
        Stop training if val MSE fails to improve for this many epochs.
        Default 25.
    seed : int
        Train/val-split seed. Also used as the single training seed when
        `seeds` is None.
    seeds : list[int] | None
        Optional seed ensemble. If provided, one AE is trained per seed and
        the model with the lowest val loss is kept. Defends against unlucky
        random inits without inflating bytes_params (only the winner is
        persisted). When None, behaves as a single-seed trainer using `seed`.
    """

    def __init__(
        self,
        bottleneck_dim: int,
        block_size: int = 32,
        n_epochs: int = 300,
        batch_size: int = 64,
        lr: float = 1e-3,
        hidden_dim: int | None = None,
        weight_decay: float = 0.0,
        val_fraction: float = 0.1,
        patience: int = 25,
        seed: int = 42,
        seeds: list[int] | None = None,
    ):
        if bottleneck_dim < 1:
            raise ValueError(f"bottleneck_dim must be >= 1, got {bottleneck_dim}")
        self.bottleneck_dim = int(bottleneck_dim)
        self.block_size = int(block_size)
        self.n_epochs = int(n_epochs)
        self.batch_size = int(batch_size)
        self.lr = float(lr)
        self.hidden_dim_override = (
            int(hidden_dim) if hidden_dim is not None else None
        )
        self.weight_decay = float(weight_decay)
        self.val_fraction = float(val_fraction)
        self.patience = int(patience)
        self.seed = int(seed)
        self.seeds = [int(s) for s in seeds] if seeds is not None else None
        self.name = f"ae_d{bottleneck_dim}"

        self._ae: _MLPAE | None = None
        self._mu: np.ndarray | None = None      # (C,) per-channel mean
        self._sigma: np.ndarray | None = None   # (C,) per-channel std
        self._n_channels: int | None = None

    # ----- helpers ---------------------------------------------------------

    def _make_blocks(self, x: np.ndarray) -> tuple[np.ndarray, int, int]:
        """
        (n, C) → ((n_blocks, SW·C), orig_n, n_blocks).
        Pads with zeros so n is a multiple of block_size.
        """
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            x = x[:, None]
        n, c = x.shape
        SW = self.block_size
        pad = (-n) % SW
        if pad:
            x = np.vstack([x, np.zeros((pad, c), dtype=float)])
        n_blocks = x.shape[0] // SW
        blocks = x.reshape(n_blocks, SW * c)
        return blocks, n, n_blocks

    def _auto_hidden_dim(self, input_dim: int) -> int:
        """
        Wider than the previous hard cap of 64.

        Old rule  : max(2·d_bottleneck, min(input_dim//4, 64))
        New rule  : max(2·d_bottleneck, min(input_dim//2, 256))

        Empirically the 64-cap was undersized for the high-channel datasets
        (solar → input_dim ≈ 4384), which is why the AE looked weak at
        small bottlenecks. The 256 cap is generous without blowing up
        bytes_params: a (256, 4384) weight matrix is ~9 MB, still
        amortised across the entire test segment.
        """
        if self.hidden_dim_override is not None:
            return self.hidden_dim_override
        return max(2 * self.bottleneck_dim, min(input_dim // 2, 256))

    # ----- lifecycle -------------------------------------------------------

    def fit(self, x_train: np.ndarray) -> "AE":
        x_train = np.asarray(x_train, dtype=float)
        if x_train.ndim != 2:
            raise ValueError(f"x_train must be 2D, got shape {x_train.shape}")
        self._n_channels = x_train.shape[1]

        # Per-channel standardisation.
        self._mu = x_train.mean(axis=0)
        self._sigma = x_train.std(axis=0) + 1e-8
        x_std = (x_train - self._mu) / self._sigma

        blocks, _, _ = self._make_blocks(x_std)
        n_blocks = blocks.shape[0]
        input_dim = self.block_size * self._n_channels
        hidden_dim = self._auto_hidden_dim(input_dim)

        # Train/val split on the block axis. With ≤10 blocks total we
        # disable early stopping (val would be a single block — too noisy
        # to be useful as a stopping signal). Seeded with self.seed so the
        # split is identical across all candidate seeds in the ensemble —
        # that makes val losses directly comparable when picking the winner.
        split_rng = np.random.default_rng(self.seed)
        n_val = int(np.floor(n_blocks * self.val_fraction))
        do_es = n_val >= 2 and (n_blocks - n_val) >= self.batch_size
        if do_es:
            perm0 = split_rng.permutation(n_blocks)
            val_idx = perm0[:n_val]
            train_idx = perm0[n_val:]
            X_tr = blocks[train_idx]
            X_val = blocks[val_idx]
        else:
            X_tr = blocks
            X_val = None

        # Seed ensemble: train one AE per seed and keep the one with the
        # lowest score (val loss if early stopping is active, otherwise the
        # final epoch's train loss). With seeds=None we fall back to a single
        # run on self.seed — preserves prior behaviour for any caller that
        # hasn't opted into the ensemble yet.
        candidate_seeds = (
            list(self.seeds) if self.seeds is not None else [self.seed]
        )

        best_ae: _MLPAE | None = None
        best_score = float("inf")

        for s_seed in candidate_seeds:
            ae = _MLPAE(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                bottleneck_dim=self.bottleneck_dim,
                lr=self.lr,
                weight_decay=self.weight_decay,
                seed=s_seed,
            )
            score = self._train_one(ae, X_tr, X_val, shuffle_seed=s_seed)
            if score < best_score:
                best_score = score
                best_ae = ae

        self._ae = best_ae
        return self

    def _train_one(
        self,
        ae: "_MLPAE",
        X_tr: np.ndarray,
        X_val: np.ndarray | None,
        shuffle_seed: int,
    ) -> float:
        """
        Train one AE in place and return its score for ensemble selection.

        Score is the best val loss seen during training when early stopping
        is active, otherwise the average mini-batch loss of the final epoch.
        On exit, `ae` holds the best-val weights (rolled back from the
        snapshot) when X_val is present, or the final-epoch weights when
        it isn't.
        """
        rng = np.random.default_rng(shuffle_seed)
        n_tr = X_tr.shape[0]
        best_val = float("inf")
        best_snapshot = ae.snapshot()
        patience_left = self.patience

        lr_min = 1e-2 * self.lr
        last_loss = float("inf")

        for epoch in range(self.n_epochs):
            # Cosine annealing — smooth, no schedule hyperparameters to tune.
            t = epoch / max(1, self.n_epochs - 1)
            lr_t = lr_min + 0.5 * (self.lr - lr_min) * (1.0 + np.cos(np.pi * t))

            perm = rng.permutation(n_tr)
            epoch_loss_sum = 0.0
            nb = 0
            for s in range(0, n_tr, self.batch_size):
                batch = X_tr[perm[s : s + self.batch_size]]
                if batch.shape[0] == 0:
                    continue
                epoch_loss_sum += ae.update_batch(batch, lr=float(lr_t))
                nb += 1
            if nb > 0:
                last_loss = epoch_loss_sum / nb

            if X_val is not None:
                val_loss = ae.forward_loss(X_val)
                if val_loss + 1e-12 < best_val:
                    best_val = val_loss
                    best_snapshot = ae.snapshot()
                    patience_left = self.patience
                else:
                    patience_left -= 1
                    if patience_left <= 0:
                        break

        if X_val is not None:
            ae.restore(best_snapshot)
            return best_val
        return last_loss

    def compress(self, x: np.ndarray) -> CompressedRecord:
        if self._ae is None:
            raise RuntimeError("AE must be fit before compress")
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            x = x[:, None]
        n_orig, c = x.shape
        if c != self._n_channels:
            raise ValueError(
                f"channel count mismatch: got {c}, AE trained on {self._n_channels}"
            )

        x_std = (x - self._mu) / self._sigma
        blocks, _, n_blocks = self._make_blocks(x_std)
        z = self._ae.encode_batch(blocks).astype(np.float64)
        return CompressedRecord(
            z=z,
            meta={"orig_len": n_orig, "n_blocks": n_blocks},
        )

    def decompress(self, record: CompressedRecord) -> np.ndarray:
        if self._ae is None:
            raise RuntimeError("AE must be fit before decompress")
        orig_len = int(record.meta["orig_len"])
        n_blocks = int(record.meta["n_blocks"])
        c = self._n_channels
        SW = self.block_size

        recon = self._ae.decode_batch(record.z)             # (n_blocks, SW*C)
        recon = recon.reshape(n_blocks * SW, c)
        recon = recon * self._sigma + self._mu              # de-standardise
        return recon[:orig_len]

    # ----- cost ------------------------------------------------------------

    def cell_cr(self, n_channels: int) -> float:
        """
        Cell-axis compression ratio for the matched-CR analysis.

        AE keeps `bottleneck_dim` numbers per (block_size × n_channels) cells,
        so cr_cells = (block_size · C) / bottleneck_dim. This is independent
        of the byte encoding and parameter overhead — the matched-CR strip
        joins AE with the row/spectral families on this value.
        """
        return float(self.block_size * n_channels) / float(self.bottleneck_dim)

    def bytes_params(self) -> int:
        if self._ae is None:
            return 0
        weights = self._ae.total_param_bytes()
        # mu and sigma vectors are also part of the persisted compressor.
        stats = int(self._mu.nbytes + self._sigma.nbytes) if self._mu is not None else 0
        return weights + stats
