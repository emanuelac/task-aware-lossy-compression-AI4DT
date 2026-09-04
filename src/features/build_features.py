"""
features/build_features.py

Generic time delay embedding for multivariate time series.

Given a list of input columns and a window size W, this module builds
a sliding window feature matrix where each row contains the last W-1 values
of each channel plus a one-step-ahead target.

The name window_size refers to the full window including the target step.

Column naming convention:
    <col>_tm<k>  — value k steps in the past  (tm = "time minus")
    <col>_t      — current value
    <col>_tp1    — one step ahead, used as target (tp = "time plus")

For W=4 and input_columns=["a", "b"], the output columns are:
    features : a_tm2, a_tm1, a_t, b_tm2, b_tm1, b_t
    targets  : a_tp1, b_tp1

This module is used by silver_transform.py inside the Spark pipeline.
It is intentionally free of any Spark dependencies so it can also be
used standalone in notebooks or tests.
"""

import pandas as pd
from typing import List, Optional, Sequence


def get_feature_column_names(input_columns: List[str], window_size: int) -> List[str]:
    """
    Generate lag feature column names for a given set of input columns
    and window size, without needing actual data.

    Parameters
    ----------
    input_columns : list of str
        Names of the input channels, e.g. ["systemimbalance", "netregulationvolume"]
    window_size : int
        Total window size including target step.
        e.g. window_size=4 produces features tm2, tm1, t (3 features)
        and target tp1 — 4 positions in total.

    Returns
    -------
    list of str
        Ordered list of feature column names, e.g.
        ["si_tm2", "si_tm1", "si_t", "nrv_tm2", "nrv_tm1", "nrv_t"]

    Examples
    --------
    >>> get_feature_column_names(["si", "nrv"], 4)
    ['si_tm2', 'si_tm1', 'si_t', 'nrv_tm2', 'nrv_tm1', 'nrv_t']
    """
    cols = []
    for col in input_columns:
        for lag in range(window_size - 2, -1, -1):
            if lag == 0:
                cols.append(f"{col}_t")
            else:
                cols.append(f"{col}_tm{lag}")
    return cols


def get_target_column_names(
    target_columns: List[str],
    horizons: Sequence[int] = (1,),
) -> List[str]:
    """
    Generate target column names for one or more forecast horizons.

    Parameters
    ----------
    target_columns : list of str
        Names of the channels to forecast.
    horizons : sequence of int, default (1,)
        Forecast horizons h. Target column for channel `col` at horizon h
        is `<col>_tp{h}`. Backwards-compatible with the single-step default.

    Returns
    -------
    list of str
        Target column names, ordered (col_outer, horizon_inner). For
        target_columns=["si"] and horizons=[6, 12] the output is
        ["si_tp6", "si_tp12"].

    Examples
    --------
    >>> get_target_column_names(["si", "nrv"])
    ['si_tp1', 'nrv_tp1']
    >>> get_target_column_names(["si"], horizons=[6, 12])
    ['si_tp6', 'si_tp12']
    """
    return [f"{col}_tp{h}" for col in target_columns for h in horizons]


def build_embedding(
    df: pd.DataFrame,
    input_columns: List[str],
    target_columns: List[str],
    window_size: int,
    horizons: Sequence[int] = (1,),
    target_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Apply time delay embedding to a pandas DataFrame.

    Constructs a feature matrix by stacking lag values of each input channel
    jointly, preserving temporal alignment across channels. Targets for each
    row are the values of each target channel at every requested horizon.

    Features are built from `df`. Targets are built from `target_df` if
    provided, otherwise from `df`. The two-DataFrame mode is what the offline
    pipeline uses: features come from the reconstructed signal x_hat (so
    the forecaster only sees what compression preserved), while targets
    come from the raw series (so we measure information loss honestly
    rather than fitting the decoder's bias).

    Parameters
    ----------
    df : pd.DataFrame
        Source for *features*. Must contain all `input_columns`. When
        `target_df` is None, must also contain `target_columns`.
    input_columns : list of str
        Channels to use as features.
    target_columns : list of str
        Channels to use as forecast targets.
    window_size : int
        Total window size including the one-step-ahead target.
        Features span tm{window_size-2} to t (window_size-1 steps).
    horizons : sequence of int, default (1,)
        Forecast horizons. For each h, target column for channel `col`
        is `<col>_tp{h} = col.shift(-h)`. Default is one-step-ahead.
    target_df : pd.DataFrame, optional
        Source for *targets*. Must be aligned with `df` (same length and
        index). If None, targets are read from `df`.

    Returns
    -------
    pd.DataFrame
        Embedded DataFrame with feature columns then target columns.
        Rows with NaN in any column (lag-induced at the head, shift-induced
        at the tail) are dropped.
    """
    if target_df is None:
        target_df = df
    elif len(target_df) != len(df):
        raise ValueError(
            f"target_df length ({len(target_df)}) must match df length ({len(df)})"
        )

    # Build all columns into a dict first, then construct the DataFrame in
    # one shot. Repeated `result[col] = ...` assignment fragments pandas'
    # block manager (one BlockManager block per dtype-aligned column) and
    # triggers a PerformanceWarning per insertion once the column count
    # grows. With wide channel sets (e.g. UCI HAR's 9 channels × 128 lags,
    # solar's 50 channels) the warning spam is real and the fragmentation
    # also slows the downstream `.to_numpy()` in the tasks.
    cols: dict[str, "pd.Series"] = {}

    # Features come from df (potentially the reconstructed signal x_hat)
    for col in input_columns:
        for lag in range(window_size - 2, -1, -1):
            col_name = f"{col}_t" if lag == 0 else f"{col}_tm{lag}"
            cols[col_name] = df[col].shift(lag).values

    # Targets come from target_df (raw, in the offline pipeline)
    for col in target_columns:
        for h in horizons:
            cols[f"{col}_tp{h}"] = target_df[col].shift(-h).values

    result = pd.DataFrame(cols, index=df.index)
    return result.dropna()


def build_embedding_from_buffer(
    buffer: List[dict],
    input_columns: List[str],
    target_columns: List[str],
    window_size: int,
) -> pd.DataFrame:
    """
    Build embedding from a list of raw row dictionaries.

    Convenience wrapper used by the Spark foreachBatch function,
    which maintains the rolling buffer as a list of dicts.

    Parameters
    ----------
    buffer : list of dict
        Each dict has keys: "timestamp" + one key per input column.
    input_columns : list of str
        Channels to use as features.
    target_columns : list of str
        Channels to forecast.
    window_size : int
        Total window size including target step.

    Returns
    -------
    pd.DataFrame
        Embedded feature matrix ready for model training/evaluation.
    """
    df = pd.DataFrame(buffer)

    if "timestamp" in df.columns:
        df = df.set_index("timestamp")

    return build_embedding(df, input_columns, target_columns, window_size)