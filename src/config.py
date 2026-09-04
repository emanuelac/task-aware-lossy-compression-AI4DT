"""
config.py

Shared configuration for the offline pipeline and any streaming
code that needs the same constants (e.g. window sizes used by both the
streaming method registry and offline compressors).

All constants here are dataset-agnostic — no column names, no domain
specifics. Dataset-specific configuration lives in:
    src/datasets/<dataset_name>.py

streaming-only constants (Kafka, RLS, InfluxDB tables, signal-analyst warm-up)
live in streaming/src/streaming_config.py.

Import this module in every offline stage:
    from config import WINDOW_SIZE, HORIZONS, ...
"""

# ---------------------------------------------------------------------------
# Windowing
# ---------------------------------------------------------------------------

# Number of time steps in the sliding window (including current and target)
# Window of 4 gives: t-2, t-1, t (features) and t+1 (target)
WINDOW_SIZE = 4

# Sliding window size used by spectral methods (DCT, DFT, DWT, PCA+DWT).
# Larger than WINDOW_SIZE to give meaningful frequency resolution.
# 32 points at 10-min sampling ≈ 5 hours of look-back.
SPECTRAL_WINDOW_SIZE = 32

# Query-window size used by AggregateQueryTask (mean / max / min / std).
# This is the granularity at which a TSDB user would issue a windowed
# aggregate query — independent of WINDOW_SIZE (time-delay embedding) and
# SPECTRAL_WINDOW_SIZE (compressor block).
AGGREGATE_QUERY_WINDOW = 64

# ---------------------------------------------------------------------------
# forecasting horizons
# ---------------------------------------------------------------------------
# Direct h-step forecasting: one frozen β per horizon, no recursive rollout.
# Target column for horizon h is <col>_tp{h}.
#
# A subset of the Muñiz-Cuza et al. (EDBT 2024) horizon set
# {1, 6, 12, 24, 36, 48}. The block-disjoint K-fold layout used by `past`
# (train_fold > eval_fold) requires an embargo gap of
# `max(HORIZONS) + WINDOW_SIZE - 1` between adjacent train / test blocks
# (López de Prado 2018, Ch. 7), so a large maximum horizon consumes block
# area; {1, 12, 36} spans short / medium / long-range dynamics on
# high-rate sensor streams.
HORIZONS = [1, 12, 36]

# Unused by the block-disjoint K-fold layout
# (scripts/run_eval.py:make_folds), which tiles the series into
# N_FOLDS + 1 disjoint blocks and trains on one block (~1/(N_FOLDS + 1)
# of N) regardless of this value. Kept only as a CLI argument.
TRAIN_FRACTION = 0.30

# Number of block-disjoint K-fold + embargo CV folds (López de Prado
# 2018, Ch. 7). Each fold k refits compressors and β on block k and
# evaluates on the embargo-trimmed block k+1; Phase 2 cross-evaluates
# every (train_fold, eval_fold) pair of the K × K temporal-shift
# matrix. summary.csv carries `train_fold` and `eval_fold` columns;
# the notebook aggregates the diagonal as mean ± std across folds.
N_FOLDS = 5

# Multi-objective loss as a convex combination of the two normalised
# terms (α is the storage weight):
#     L = (1 - ALPHA) * loss_tilde + ALPHA * storage_tilde
# α = 0 recovers pure reconstruction, α = 1 pure storage, α = 0.5 equal
# weight. Both terms are per-cell min-max normalised (src/normalisation.py).
ALPHA = 0.5

# ---------------------------------------------------------------------------
# Headline compressor pool (normalisation + selection scope)
# ---------------------------------------------------------------------------
# Single source of truth for which compressors enter the headline comparison:
# the pool over which the per-cell min-max normalisation and the arg-min
# selection operate. Methods outside the pool are still evaluated and their
# raw loss / storage recorded (for reference), but they never anchor the
# normalisation and never enter the arg-min: their loss_tilde / storage_tilde
# / multi_obj are left empty (NaN). Change the active pool here, in one place.
#
# Excluded from the headline: the lossless baselines (identity, gorilla),
# kept only as an NMSE / CR reference; the selection is defined over the lossy
# pool. SimPiece is instantiated only in matched-CR mode (see
# src/method_grid.py), so the lossy pool is exactly the matched-CR grid.
LOSSLESS_METHODS = frozenset({"identity", "gorilla"})


def is_headline_method(method_name: str) -> bool:
    """True iff `method_name` is in the matched-CR lossy pool that the
    normalisation and the arg-min operate over. Edit the rule here to change
    the active pool everywhere at once."""
    return method_name not in LOSSLESS_METHODS


# Cap on the classification reference set. 1-NN-DTW inference is O(n_ref) per
# query, so this keeps wall time tractable. Uniform-random, fixed seed.
CLASSIFICATION_KNN_TRAIN_SUBSAMPLE = 5_000

# AE training is init-sensitive: train one AE per seed, keep the lowest
# val loss.
AE_SEEDS = [0, 1, 2]

# ---------------------------------------------------------------------------
# Runtime caps
# ---------------------------------------------------------------------------

# Row cap on dataset load. `None` disables it. A dataset can set
# `MAX_ROWS_OVERRIDE` in its config to impose its own ceiling.
MAX_ROWS = None
