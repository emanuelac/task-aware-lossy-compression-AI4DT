"""
method_grid.py

Single source of truth for the compressor portfolio. Both the offline
evaluation (scripts/run_eval.py) and the streaming pipeline
(streaming/src/methods/) import build_method_grid from here, so the two
layers always run the identical set of compressors under the identical
family_cr{target} naming.
"""

from typing import List

import config as cfg
from compression import (
    AE, Compressor, DCT, Downsample, DWT, Gorilla, Identity, LTTB, PCA,
    SimPiece,
)


def build_method_grid(n_channels: int, block_size: int) -> List[Compressor]:
    """
    shared compressor portfolio:
      - Identity   : NMSE floor reference (no compression)
      - Gorilla    : operational lossless TSDB baseline
      - Downsample : row-axis lossy at ratios target_crs
      - LTTB       : row-axis adaptive at ratios target_crs
      - PCA        : channel-axis lossy at K = round(C / target_cr)
      - DCT        : time-frequency lossy at cr_cells ∈ {2, 3, 4, 6, 8}
                     (block_size = cfg.SPECTRAL_WINDOW_SIZE)
      - DWT        : time-frequency lossy at cr_cells ∈ {2, 4, 8}
                     (DWT requires block_size/k to be a power of two; the
                      reachable subset is invariant as long as block_size
                      is itself a power of two)

    Intermediate levels per family are what make Pareto plots and the
    task-aware / task-agnostic comparison meaningful.
    """
    methods: List[Compressor] = [Identity(), Gorilla()]
    # Lossy families are sized so the cell-axis CR is comparable across them:
    #   downsample_r       → cr_cells = r
    #   lttb_r             → cr_cells = r
    #   dct_k{K}, b=32     → cr_cells = 32 / K
    #   dwt_k{K}, b=32     → cr_cells = 32 / K
    #   pca_k{K}           → cr_cells = n_channels / K
    # PCA's K is picked per-dataset to land on the same target CRs as the row-
    # axis families, so the matched-CR analysis joins cleanly on `cr_cells`
    # across every family.
    # Every lossy family is named <family>_cr<target> so the comparison axis is
    # uniform. The underlying knob (ratio, K, n_coeffs) varies with how each
    # family reaches that CR, but it's an implementation detail — the matched-
    # CR analysis joins on `cr_cells` and the plots read by name.
    target_crs = (2, 3, 4, 6, 8)
    for r in target_crs:
        ds_method = Downsample(ratio=r)
        ds_method.name = f"downsample_cr{r}"
        methods.append(ds_method)
    for r in target_crs:                          # LTTB matches Downsample at cr_cells=r
        lttb_method = LTTB(ratio=r)
        lttb_method.name = f"lttb_cr{r}"
        methods.append(lttb_method)
    # PCA targets the same CRs as the row-axis families. K = round(C / target),
    # so cell_cr ≈ target on every dataset instead of drifting with n_channels.
    #
    # Why some datasets end up with fewer PCA points than others:
    #   PCA's CR is C/K with K ∈ {1, ..., C}. The number of *distinct* K values
    #   reachable for the target set {2, 3, 4, 6, 8} therefore depends on C.
    #   NB Python's round() is banker's rounding (round-half-to-even), so
    #   round(9/2) = 4, not 5. e.g. C=9 → K ∈ {4, 3, 2, 2, 1} → 4 distinct
    #   points (6 collides with 4); low-channel datasets reach fewer.
    #   Collisions are dropped — they would just re-fit the same K. This is
    #   not a grid bug: low-channel datasets genuinely have less channel-axis
    #   compression headroom, and the matched-CR analysis joins on `cr_cells`
    #   so missing points are visible rather than silently treated as ties.
    # Name PCA by *target* CR (pca_cr2, pca_cr3, ...) rather than by K. The
    # underlying K varies with n_channels and isn't comparable across datasets;
    # the target CR is. The actual cell_cr lives in the storage_rows column,
    # so the honest number is still available downstream.
    seen_K: set[int] = set()
    for target in target_crs:
        K = max(1, round(n_channels / target))
        if K > n_channels or K in seen_K:
            continue
        seen_K.add(K)
        pca = PCA(n_components=K)
        pca.name = f"pca_cr{target}"
        methods.append(pca)
    # DCT: cell_cr = block_size / k. k can be any integer in [1, block_size],
    # so we hit every target_cr. We round k to the nearest integer per target
    # and drop duplicates. At block_size=32: target→k = 2→16, 3→11, 4→8,
    # 6→5, 8→4. Sweeps of `block_size` rescale k proportionally.
    seen_k_dct: set[int] = set()
    for target in target_crs:
        k = max(1, round(block_size / target))
        if k in seen_k_dct:
            continue
        seen_k_dct.add(k)
        dct_method = DCT(n_coeffs=k, block_size=block_size)
        dct_method.name = f"dct_cr{target}"
        methods.append(dct_method)
    # DWT: block_size / n_coeffs must be a power of two, so target_crs 3 and
    # 6 are unreachable and are skipped (we don't fake-match by rounding to a
    # different CR). At block_size=32 the surviving triple is k=16→cr=2,
    # k=8→cr=4, k=4→cr=8. For other block sizes the surviving triple
    # rescales (e.g. block_size=16 → k=8/4/2; block_size=64 → k=32/16/8).
    seen_k_dwt: set[int] = set()
    for target in target_crs:
        k = block_size // target
        # k must be a positive power of two AND exactly divide block_size.
        if k < 1 or k * target != block_size or (k & (k - 1)) != 0:
            continue
        if k in seen_k_dwt:
            continue
        seen_k_dwt.add(k)
        dwt_method = DWT(n_coeffs=k, block_size=block_size)
        dwt_method.name = f"dwt_cr{target}"
        methods.append(dwt_method)
    # AE: bottleneck_dim chosen so cr_cells = (block_size · C) / d matches the
    # target CR. The frozen weight footprint is reported in `bytes_params`
    # but excluded from the storage column.
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
    # Sim-Piece: error-bounded piecewise-linear (TerseTS / swing-door family).
    # Adds a new paradigm to the portfolio — pointwise eps-bounded rather than
    # L2-optimal. eps is fitted per dataset in fit() so the achieved CR on
    # training data matches each target. Naming follows family_cr{target} so
    # §6 / §7 / §14 join cleanly.
    for target in target_crs:
        sp_method = SimPiece(target_cr=target)
        sp_method.name = f"sim_piece_cr{target}"
        methods.append(sp_method)
    # Note: SimPiece's native error-budget parameterisation (fixed-ε, with a
    # signal-adaptive CR off the matched-CR grid) is deliberately NOT
    # instantiated — compressors are compared on the shared matched-CR
    # axis only, so only the sim_piece_cr<target> wrappers above enter the
    # portfolio.
    return methods
