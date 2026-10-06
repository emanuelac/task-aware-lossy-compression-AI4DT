"""Single source of truth for the compressor portfolio. Both the offline runner
(scripts/run_eval.py) and the streaming pipeline import build_method_grid, so
both run the identical compressors under the same family_cr{target} naming."""

from typing import List

import config as cfg
from compression import (
    AE, Compressor, DCT, Downsample, DWT, Gorilla, Identity, LTTB, PCA,
    SimPiece,
)


def build_method_grid(n_channels: int, block_size: int) -> List[Compressor]:
    """Identity/Gorilla baselines plus each lossy family (Downsample, LTTB, PCA,
    DCT, DWT, AE, SimPiece) at matched target CRs {2,3,4,6,8}, named
    <family>_cr<target>."""
    methods: List[Compressor] = [Identity(), Gorilla()]
    target_crs = (2, 3, 4, 6, 8)

    for r in target_crs:  # row-axis: cr_cells = r
        ds_method = Downsample(ratio=r)
        ds_method.name = f"downsample_cr{r}"
        methods.append(ds_method)
    for r in target_crs:
        lttb_method = LTTB(ratio=r)
        lttb_method.name = f"lttb_cr{r}"
        methods.append(lttb_method)

    # PCA: K = round(C / target), collisions dropped (banker's rounding, so
    # low-channel datasets reach fewer distinct K). Named by target CR.
    seen_K: set[int] = set()
    for target in target_crs:
        K = max(1, round(n_channels / target))
        if K > n_channels or K in seen_K:
            continue
        seen_K.add(K)
        pca = PCA(n_components=K)
        pca.name = f"pca_cr{target}"
        methods.append(pca)

    # DCT: cr_cells = block_size / k; k rounded per target, duplicates dropped.
    seen_k_dct: set[int] = set()
    for target in target_crs:
        k = max(1, round(block_size / target))
        if k in seen_k_dct:
            continue
        seen_k_dct.add(k)
        dct_method = DCT(n_coeffs=k, block_size=block_size)
        dct_method.name = f"dct_cr{target}"
        methods.append(dct_method)

    # DWT: needs block_size/k a power of two, so targets 3 and 6 are skipped.
    seen_k_dwt: set[int] = set()
    for target in target_crs:
        k = block_size // target
        if k < 1 or k * target != block_size or (k & (k - 1)) != 0:
            continue
        if k in seen_k_dwt:
            continue
        seen_k_dwt.add(k)
        dwt_method = DWT(n_coeffs=k, block_size=block_size)
        dwt_method.name = f"dwt_cr{target}"
        methods.append(dwt_method)

    # AE: bottleneck d = round(block_size * C / target). Weights go to
    # bytes_params, excluded from the storage column.
    seen_d_ae: set[int] = set()
    for target in target_crs:
        d = max(1, round(block_size * n_channels / target))
        if d in seen_d_ae:
            continue
        seen_d_ae.add(d)
        ae_method = AE(
            bottleneck_dim=d,
            block_size=block_size,
            seeds=cfg.AE_SEEDS,
        )
        ae_method.name = f"ae_cr{target}"
        methods.append(ae_method)

    # SimPiece: error-bounded piecewise-linear; eps fitted per dataset to match
    # each target CR (matched-CR mode only).
    for target in target_crs:
        sp_method = SimPiece(target_cr=target)
        sp_method.name = f"sim_piece_cr{target}"
        methods.append(sp_method)

    return methods
