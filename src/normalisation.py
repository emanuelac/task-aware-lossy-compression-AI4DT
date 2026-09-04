"""
normalisation.py

Canonical per-cell min-max normalisation of the evaluation summary —
the single implementation used by
scripts/run_eval.py at sweep time, by scripts/renormalise_summary.py to
refresh existing summary.csv files after a convention change, and by
notebook/results.ipynb (which consumes the columns as-is and performs
no re-normalisation of its own).

Scopes
------
loss_tilde   — min-max across methods within each
               (task, subtask..., train_fold, eval_fold) cell.
               The normalisation is per (t, s, k) cell so
               subtask-specific difficulty does not mask compressor
               differences, and every cell attains 0 (best method) and
               1 (worst method).
storage_tilde — min-max across methods within each
               (train_fold, eval_fold) cell only. bytes_z
               depends on (method, cell), not on which task reads them,
               so storage enters once per cell, not once per subtask.
multi_obj    — (1 − α) · loss_tilde + α · storage_tilde,
               with α the STORAGE weight (α = 0 pure reconstruction,
               α = 1 pure storage, α = 0.5 the headline equal weight).

Convention: α weights storage. The argmin of multi_obj is invariant
under the monotone reparametrisation λ = α / (1 − α).

Headline pool (normalisation scope)
-----------------------------------
The min-max anchors are taken over the HEADLINE pool only — by default
the methods for which config.is_headline_method() is True (the matched-CR
lossy pool; lossless baselines excluded).
Methods outside the pool keep their raw loss / storage but receive NaN
loss_tilde / storage_tilde / multi_obj, so they can neither anchor the
normalisation nor win the arg-min, while remaining available (raw) for
reference plots. Pass `pool_filter="all"` to normalise
over every method (all methods), or a custom callable to scope it
differently. This is the single place that decides the active pool, so
run_eval.py and renormalise_summary.py stay consistent by construction.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import pandas as pd

# Subtask axes per task family — must match TaskEvaluator.subtask_columns.
TASK_SUBTASK_COLUMNS: Dict[str, List[str]] = {
    "forecasting": ["target", "horizon"],
    "aggregate_query": ["agg"],
    "classification": [],
}

FOLD_COLUMNS = ["train_fold", "eval_fold"]


def _minmax_masked(
    df: pd.DataFrame,
    keys: Sequence[str],
    col: str,
    pool_mask: pd.Series,
) -> pd.Series:
    """Per-group min-max of `col`, anchored ONLY to rows where `pool_mask`
    is True. Pool rows get their value in [0, 1] (0 if the pool range is
    degenerate); rows outside the pool get NaN — they are recorded raw but
    take no part in the headline comparison."""
    pool = df.loc[pool_mask]
    lo = pool.groupby(list(keys))[col].transform("min").reindex(df.index)
    hi = pool.groupby(list(keys))[col].transform("max").reindex(df.index)
    rng = hi - lo
    tilde = (df[col] - lo) / rng
    # degenerate pool range (all methods equal) → 0 for the pool rows
    tilde = tilde.mask(pool_mask & (rng.abs() < 1e-12), 0.0)
    # rows outside the headline pool carry no normalised value
    tilde = tilde.mask(~pool_mask, float("nan"))
    return tilde


def normalise_summary(
    summary: pd.DataFrame,
    alpha: float,
    subtask_columns: Dict[str, List[str]] | None = None,
    pool_filter=None,
) -> pd.DataFrame:
    """
    Return `summary` with loss_tilde / storage_tilde / multi_obj
    (re)computed per the canonical scopes above. Existing columns of
    the same names are overwritten.

    Parameters
    ----------
    summary : DataFrame
        Long-form summary with at least [task, method, train_fold,
        eval_fold, loss, storage] and the per-task subtask columns.
    alpha : float
        Storage weight in Eq. (3.5), in [0, 1].
    subtask_columns : dict, optional
        task name → subtask column list. Defaults to
        TASK_SUBTASK_COLUMNS; tasks not in the mapping are normalised
        with no subtask split (like classification).
    pool_filter : callable(method_name)->bool | "all" | None
        Selects the HEADLINE pool that anchors the min-max normalisation.
        Rows outside the pool keep raw loss / storage but get NaN
        loss_tilde / storage_tilde / multi_obj. `None` (default) uses
        config.is_headline_method (lossless baselines excluded);
        "all" includes every method.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    mapping = TASK_SUBTASK_COLUMNS if subtask_columns is None else subtask_columns

    if pool_filter is None:
        from config import is_headline_method as pool_filter
    elif pool_filter == "all":
        pool_filter = lambda _name: True  # noqa: E731

    parts = []
    for task in summary["task"].unique():
        sub = summary[summary["task"] == task].copy()
        grp_keys = list(mapping.get(task, [])) + FOLD_COLUMNS
        mask = sub["method"].map(pool_filter).astype(bool)
        sub["loss_tilde"] = _minmax_masked(sub, grp_keys, "loss", mask)
        parts.append(sub)
    out = pd.concat(parts, ignore_index=True)

    out_mask = out["method"].map(pool_filter).astype(bool)
    out["storage_tilde"] = _minmax_masked(out, FOLD_COLUMNS, "storage", out_mask)
    out["multi_obj"] = (
        (1.0 - alpha) * out["loss_tilde"] + alpha * out["storage_tilde"]
    )
    return out
