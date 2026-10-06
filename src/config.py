"""Shared, dataset-agnostic config for the offline pipeline.
Dataset-specific config lives in src/datasets/<name>.py."""

WINDOW_SIZE = 4              # time-delay embedding window (t-2..t -> t+1)
SPECTRAL_WINDOW_SIZE = 32    # block size for DCT/DFT/DWT/PCA+DWT
AGGREGATE_QUERY_WINDOW = 64  # windowed mean/max/min/std granularity

HORIZONS = [1, 12, 36]       # direct h-step forecasting horizons
TRAIN_FRACTION = 0.30        # CLI back-compat only; unused by block-disjoint CV
N_FOLDS = 5                  # block-disjoint K-fold + embargo
ALPHA = 0.5                  # storage weight: L = (1-ALPHA)*loss_tilde + ALPHA*storage_tilde

# Lossless baselines: evaluated for reference, excluded from normalisation and arg-min.
LOSSLESS_METHODS = frozenset({"identity", "gorilla"})


def is_headline_method(method_name: str) -> bool:
    return method_name not in LOSSLESS_METHODS


CLASSIFICATION_KNN_TRAIN_SUBSAMPLE = 5_000  # cap on DTW-1NN reference set
AE_SEEDS = [0, 1, 2]         # train one AE per seed, keep lowest val loss
MAX_ROWS = None              # row cap on load; None disables
