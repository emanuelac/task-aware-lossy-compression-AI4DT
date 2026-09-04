"""
scripts/run_eval.py

Evaluation runner — offline study, no streaming, no batch loop.

Pipeline (block-disjoint K-fold × temporal-shift matrix)
--------------------------------------------------------
1.  Load a dataset.
2.  Define K = N_FOLDS block-disjoint folds with embargo (see
    make_folds below). The series is tiled into K + 1 disjoint blocks
    of width B = N // (K + 1); fold k has
      train = block k       = df[k·B : (k+1)·B)
      test  = block k+1     = df[(k+1)·B + g : (k+2)·B)
    with embargo gap g = max(HORIZONS) + WINDOW_SIZE − 1. Train and
    test blocks are pairwise disjoint; no row is tested more than once.
3.  Phase 1 — fit. For each fold k:
      a. Fit a fresh method grid on raw_train_k. Persist parameters
         under compressors/fold_{k}/.
      b. Fit each TaskEvaluator on raw_train_k. β / KNN frozen here.
4.  Phase 2 — cross-evaluate. For each evaluated (train_fold k, eval_fold j):
      a. For each method m in fold k's grid: compress raw_test_j →
         decompress → x_hat. Score every task with (x_hat, raw_test_j).
      b. Finalize tasks → per-(method, subtask) rows tagged with
         (train_fold=k, eval_fold=j). Reset accumulators before moving
         to the next cell so β survives across cells but stats don't.
    By default only the cells any reported metric consumes are evaluated:
    the diagonal (k == j, read by the forward tasks) and the admissible
    `past` cells (k − j ≥ 2). Since normalisation is per-cell, this is
    byte-identical to the full matrix on every reported number at ~half
    the cost; pass --full-matrix for all K × K cells (§7 diagnostic).
5.  Storage is per (method, train_fold, eval_fold) — equal to
    `bytes_z` only (compressed-payload bytes). Model-parameter bytes
    (AE weights, PCA basis, …) are reported in `bytes_params` for
    transparency but excluded from `storage`, since parameters are a
    fixed one-shot deployment cost rather than a per-window cost.
6.  Min-max normalisation is done per (task, subtask) over the entire
    (method × train_fold × eval_fold) matrix so multi_obj is directly
    comparable across cells of the temporal-shift matrix.
7.  Save summary.csv (with `train_fold` and `eval_fold` columns) and
    print a compact diagonal argmin printout (the standard sliding-
    window CV view); off-diagonal cells are inspected in the notebook.

Run
---
    python3 scripts/run_eval.py --dataset ett
    python3 scripts/run_eval.py --dataset ett --n-folds 5
"""

from __future__ import annotations

import argparse
import importlib
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

# Allow running from repo root: add src/ to sys.path.
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import config as cfg
from compression import (
    AE, Compressor, DCT, Downsample, DWT, Gorilla, Identity, LTTB, PCA,
    SimPiece,
)
from normalisation import normalise_summary
from method_grid import build_method_grid
from tasks import (
    AggregateQueryTask,
    ClassificationTask,
    ForecastingTask,
    TaskEvaluator,
)


# ---------------------------------------------------------------------------
# Method grid.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Task grid. Forecasting is the primary task; AD slots in here as a second
# evaluator without touching the rest of the runner.
# ---------------------------------------------------------------------------
def build_task_grid(
    ds, aggregate_query_window: int, forecast_window_size: int = cfg.WINDOW_SIZE,
) -> List[TaskEvaluator]:
    tasks: List[TaskEvaluator] = [
        ForecastingTask(
            input_columns=ds.INPUT_COLUMNS,
            target_columns=ds.TARGET_COLUMNS,
            window_size=forecast_window_size,
            horizons=cfg.HORIZONS,
        ),
        AggregateQueryTask(
            target_columns=ds.TARGET_COLUMNS,
            window=aggregate_query_window,
        ),
    ]
    # Classification runs only on datasets that ship a native class label.
    label_col = getattr(ds, "CLASSIFICATION_LABEL_COL", None)
    if label_col:
        cls_window = getattr(ds, "CLASSIFICATION_WINDOW_SIZE", cfg.WINDOW_SIZE)
        tasks.append(
            ClassificationTask(
                input_columns=ds.INPUT_COLUMNS,
                label_column=label_col,
                window_size=cls_window,
                train_subsample=cfg.CLASSIFICATION_KNN_TRAIN_SUBSAMPLE,
            )
        )
    return tasks


# ---------------------------------------------------------------------------
# Min-max normalisation — canonical implementation lives in
# src/normalisation.py. This wrapper adapts the
# TaskEvaluator list to the task → subtask-columns mapping it expects.
# ---------------------------------------------------------------------------
def normalise_loss(
    summary: pd.DataFrame,
    tasks: List[TaskEvaluator],
    alpha: float,
) -> pd.DataFrame:
    """
    (Re)compute loss_tilde / storage_tilde / multi_obj per the canonical
    per-cell scopes — see src/normalisation.py for the full contract:

    loss_tilde    — min-max across methods within each
                    (task, subtask, train_fold, eval_fold) cell (Eq. 3.3).
    storage_tilde — min-max across methods within each
                    (train_fold, eval_fold) cell, no subtask split (Eq. 3.4).
    multi_obj     — (1 − α) · loss_tilde + α · storage_tilde (Eq. 3.5),
                    α = STORAGE weight. NOT to be averaged across task
                    families to obtain L_TA — storage would enter once
                    per subtask; the notebook composes
                    L_TA = (1 − α) · h̄ + α · S̃ instead.

    POOL SCOPE: the min-max anchors are taken over the HEADLINE pool only
    (config.is_headline_method — the matched-CR lossy pool; lossless and
    SimPiece native-ε variants excluded). Those excluded methods are still
    evaluated and their RAW loss / storage recorded, but they receive NaN
    loss_tilde / storage_tilde / multi_obj, so they neither anchor the
    normalisation nor win the arg-min. See src/normalisation.py.
    """
    mapping = {t.name: list(t.subtask_columns) for t in tasks}
    return normalise_summary(summary, alpha=alpha, subtask_columns=mapping)


# ---------------------------------------------------------------------------
# Block-disjoint K-fold with embargo (López de Prado 2018, Ch. 7).
# ---------------------------------------------------------------------------
def make_folds(
    n_total: int,
    train_fraction: float,
    n_folds: int,
    embargo: int | None = None,
) -> List[Tuple[int, int, int, int]]:
    """
    Return [(train_start, train_end, test_start, test_end), ...] —
    one tuple per fold.

    Block-disjoint K-fold with embargo. The series is tiled into
    n_folds + 1 disjoint contiguous blocks of width B = n_total //
    (n_folds + 1). Fold k ∈ {0, …, n_folds - 1}:

        train_k = block k     = [k·B,             (k+1)·B)
        test_k  = block k+1   = [(k+1)·B + g,     (k+2)·B)

    where g = max(HORIZONS) + WINDOW_SIZE - 1 is the embargo gap: the
    minimum separation needed so that train *targets* (x_{t+h} for
    t ∈ train) and test *features* (x_{t-W+1..t} for t ∈ test) cannot
    overlap. The last fold's test block absorbs the remainder
    (n_total - (n_folds + 1)·B rows) so the residual tail is not lost.

    Leakage properties
    ------------------
    - Diagonal cell (train=k, eval=k): the embargo guarantees no
      train target lands inside the test feature span, even at the
      longest horizon.
    - Past cell (train=k, eval=j) with k > j: the train block B_k
      and the test block B_{j+1} are disjoint *blocks* in time. They
      are leakage-free iff k − j ≥ 2: when k = j + 1, train_k and
      test_j sit inside the *same* B_{j+1} block (the embargo only
      trims test_j's leading edge, not its trailing one). So the
      downstream past-task aggregator must mask {(k, j) : k > j AND
      k − j ≥ 2}; for K = 5 this leaves 6 valid past cells out of
      the 10 below-diagonal candidates.

    Why this replaces the prior sliding-window layout
    -------------------------------------------------
    The previous layout (W = train_fraction · N, shifted by T per
    fold) gave overlapping train windows AND placed train_2 entirely
    around test_1 — a structural leak on the past task that no
    horizon choice could fix. The block-disjoint layout drops that
    leak at the cost of using only ~1/(n_folds + 1) of n_total for
    training instead of train_fraction. The `train_fraction`
    argument is a CLI option only and does not affect the layout.
    """
    if n_folds < 1:
        raise ValueError(f"n_folds must be >= 1, got {n_folds}")
    n_blocks = n_folds + 1
    B = n_total // n_blocks
    if B <= 0:
        raise ValueError(
            f"n_total={n_total} too small for {n_folds} folds "
            f"({n_blocks} blocks)"
        )
    if embargo is None:
        embargo = max(cfg.HORIZONS) + cfg.WINDOW_SIZE - 1
    if embargo >= B:
        raise ValueError(
            f"embargo={embargo} >= block width {B}; reduce horizons "
            f"or n_folds"
        )
    folds: List[Tuple[int, int, int, int]] = []
    for k in range(n_folds):
        train_start = k * B
        train_end = (k + 1) * B
        test_start = (k + 1) * B + embargo
        test_end = (k + 2) * B if k < n_folds - 1 else n_total
        folds.append((train_start, train_end, test_start, test_end))
    return folds


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default="uci_har",
        help="Dataset module under src/datasets/ (default: uci_har)",
    )
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=cfg.TRAIN_FRACTION,
        help="Initial training fraction for fold 0 (default: cfg.TRAIN_FRACTION).",
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        default=cfg.N_FOLDS,
        help="Number of sliding-window CV folds (default: cfg.N_FOLDS).",
    )
    parser.add_argument(
        "--spectral-window-size",
        type=int,
        default=cfg.SPECTRAL_WINDOW_SIZE,
        help=(
            "block_size for the spectral-family compressors (DCT / DWT / AE) "
            "(default: cfg.SPECTRAL_WINDOW_SIZE). Sensitivity-sweep knob."
        ),
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=cfg.WINDOW_SIZE,
        help=(
            "Time-delay embedding width for the FORECASTING task: window_size "
            "points per channel (window_size - 1 lags). Also sets the embargo "
            "gap g = max(HORIZONS) + window_size - 1. Default cfg.WINDOW_SIZE. "
            "Sensitivity-sweep knob; does NOT change the classification window."
        ),
    )
    parser.add_argument(
        "--aggregate-query-window",
        type=int,
        default=getattr(cfg, "AGGREGATE_QUERY_WINDOW", 64),
        help=(
            "Window size for AggregateQueryTask (mean / max / min / std) "
            "(default: cfg.AGGREGATE_QUERY_WINDOW). Sensitivity-sweep knob."
        ),
    )
    parser.add_argument(
        "--label",
        type=str,
        default="",
        help=(
            "Optional suffix for the output-stamp directory. When sweeping a "
            "knob across multiple runs, set this per-run (e.g. 'w16', 'w64', "
            "'tf02', 'k10') so the resulting summary.csvs land in distinct "
            "subdirectories rather than trampling each other."
        ),
    )
    parser.add_argument(
        "--full-matrix",
        action="store_true",
        help=(
            "Evaluate every (train_fold, eval_fold) cell of the K×K "
            "temporal-shift matrix. Default (off) evaluates only the cells any "
            "reported metric reads: the diagonal (train_fold == eval_fold, "
            "read by the forward tasks) and the admissible past cells "
            "(train_fold - eval_fold >= 2). Because loss_tilde / storage_tilde "
            "normalise within each cell, the reduced run is byte-identical to "
            "the full one on every reported number at ~half the Phase-2 cost. "
            "Enable only for the notebook's §7 full-matrix diagnostic."
        ),
    )
    args = parser.parse_args()

    # ---- Load dataset ------------------------------------------------------
    ds = importlib.import_module(f"datasets.{args.dataset}")
    csv_path = REPO_ROOT / "data" / ds.DATA_FILE
    print(f"[eval] dataset={args.dataset}  file={csv_path}")
    df = pd.read_csv(csv_path)
    df[ds.TIMESTAMP_COL] = pd.to_datetime(df[ds.TIMESTAMP_COL])
    df = df.sort_values(ds.TIMESTAMP_COL).set_index(ds.TIMESTAMP_COL)
    # Keep the input columns (cast to float for the compressors) plus, if the
    # dataset is classification-capable, the native label column. The label
    # column is preserved as-is — string labels and integer labels are both
    # fine for sklearn's KNN.
    label_col = getattr(ds, "CLASSIFICATION_LABEL_COL", None)
    kept_cols = list(ds.INPUT_COLUMNS)
    df_inputs = df[kept_cols].astype(float)
    if label_col and label_col in df.columns:
        df = df_inputs.assign(**{label_col: df[label_col].values})
    else:
        df = df_inputs
    # Some datasets (e.g. aus_elec_demand) have sporadic NaN cells from DST /
    # maintenance gaps. Forward-fill then back-fill preserves length and
    # continuity without inventing data — at <1% missing this is harmless.
    n_nan_before = int(df.isna().any(axis=1).sum())
    if n_nan_before:
        df = df.ffill().bfill()
        print(f"[eval] forward/back-filled {n_nan_before} rows with NaN")

    # Truncate to the configured row cap so runtime stays bounded. `MAX_ROWS =
    # None` means no cap; per-dataset `MAX_ROWS_OVERRIDE` is honoured if present.
    override = getattr(ds, "MAX_ROWS_OVERRIDE", None)
    max_rows = override if override is not None else cfg.MAX_ROWS
    n_raw = len(df)
    if max_rows is not None and n_raw > int(max_rows):
        max_rows = int(max_rows)
        df = df.iloc[:max_rows]
        print(f"[eval] truncated {n_raw} → {len(df)} rows (cap={max_rows})")

    n_total = len(df)
    print(f"[eval] n_total={n_total}  channels={ds.INPUT_COLUMNS}  "
          f"targets={ds.TARGET_COLUMNS}")
    print(
        f"[eval] horizons={cfg.HORIZONS}  window_size={args.window_size}  "
        f"spectral_window_size={args.spectral_window_size}  "
        f"aggregate_query_window={args.aggregate_query_window}  "
        f"train_fraction={args.train_fraction}  n_folds={args.n_folds}"
    )

    # ---- Fold layout -------------------------------------------------------
    folds = make_folds(
        n_total, args.train_fraction, args.n_folds,
        embargo=max(cfg.HORIZONS) + args.window_size - 1,
    )
    print(f"[eval] n_folds={args.n_folds}  layout:")
    for k, (tr_start, tr_end, te_start, te_end) in enumerate(folds):
        print(f"  fold {k}: train=[{tr_start},{tr_end})  "
              f"test=[{te_start},{te_end})  "
              f"train_rows={tr_end - tr_start}  "
              f"test_rows={te_end - te_start}")

    # ---- Output paths ------------------------------------------------------
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.label:
        stamp = f"{stamp}_{args.label}"
    out_dir = REPO_ROOT / "results" / args.dataset / stamp
    (out_dir / "compressors").mkdir(parents=True, exist_ok=True)

    # ---- Phase 1: per-fold fit (compressors + tasks) ----------------------
    # Cache each fold's fitted compressors and tasks so Phase 2 can evaluate
    # every (train_fold, eval_fold) cell of the temporal-shift matrix without
    # re-fitting. Compressor fits and β fits are unchanged from the
    # standard CV; only the inner compress+evaluate loop now spans K × K
    # cells instead of K diagonals.
    fold_methods: List[List[Any]] = []
    fold_tasks: List[List[TaskEvaluator]] = []
    raw_trains: List[pd.DataFrame] = []
    raw_tests: List[pd.DataFrame] = []
    tasks_proto: List[TaskEvaluator] = []  # captured from fold 0 for the printout

    for fold_idx, (train_start, train_end, test_start, test_end) in enumerate(folds):
        print(f"\n[eval] === fit fold {fold_idx} ===")
        raw_train_k = df.iloc[train_start:train_end]
        raw_test_k = df.iloc[test_start:test_end]
        raw_trains.append(raw_train_k)
        raw_tests.append(raw_test_k)

        # Fit compressors fresh on this fold's training segment. Slice to the
        # numeric input columns — a label column (if present) is for the
        # ClassificationTask only and must not enter the compressor matrices.
        x_train_k = raw_train_k[list(ds.INPUT_COLUMNS)].to_numpy()
        methods = build_method_grid(
            n_channels=x_train_k.shape[1],
            block_size=args.spectral_window_size,
        )
        fold_pkl_dir = out_dir / "compressors" / f"fold_{fold_idx}"
        fold_pkl_dir.mkdir(parents=True, exist_ok=True)
        for m in methods:
            m.fit(x_train_k)
            m.save(fold_pkl_dir / f"{m.name}.pkl")
        print(f"[eval] fit {len(methods)} compressors  "
              f"n_train={len(raw_train_k)}")

        # Fit tasks fresh on this fold's training segment. β / KNN frozen
        # here; accumulators reset between cells in Phase 2.
        tasks = build_task_grid(ds, args.aggregate_query_window, args.window_size)
        for t in tasks:
            t.fit(raw_train_k)

        fold_methods.append(methods)
        fold_tasks.append(tasks)
        if not tasks_proto:
            tasks_proto = tasks

    # ---- Phase 2: cross-evaluate (K × K matrix) ---------------------------
    # For each (train_fold k, eval_fold j): take compressors_k and tasks_k
    # (β_k, KNN_k), compress raw_test_j, decompress, score tasks_k. Storage
    # is also per (k, j) because bytes_z depends on the data being
    # compressed.
    all_summaries: List[pd.DataFrame] = []
    all_storage: List[pd.DataFrame] = []

    for k in range(args.n_folds):
        methods_k = fold_methods[k]
        tasks_k = fold_tasks[k]
        print(f"\n[eval] === eval train_fold={k} on all {args.n_folds} test blocks ===")
        for j in range(args.n_folds):
            # Skip cells no reported metric consumes. Forward tasks read the
            # diagonal (k == j); `past` reads the admissible below-diagonal
            # cells (k - j >= 2). The gap-1 below-diagonal cells are excluded
            # from `past` (adjacent folds share a block, so they would score a
            # compressor on its own training block) and the above-diagonal
            # cells feed no aggregation. loss_tilde / storage_tilde normalise
            # *within* each (train_fold, eval_fold) cell, so the skipped cells
            # never anchor another cell's normalisation either — the reduced
            # run is byte-identical to the full one on every reported number.
            # --full-matrix restores all K×K cells for the §7 diagnostic.
            if not args.full_matrix and not (k == j or k - j >= 2):
                continue
            raw_test_j = raw_tests[j]
            x_test_j = raw_test_j[list(ds.INPUT_COLUMNS)].to_numpy()
            raw_bytes_test_j = int(x_test_j.nbytes)

            storage_rows = []
            for m in methods_k:
                rec = m.compress(x_test_j)
                x_hat = m.decompress(rec)
                if x_hat.shape != x_test_j.shape:
                    raise RuntimeError(
                        f"{m.name}: x_hat shape {x_hat.shape} != "
                        f"raw_test shape {x_test_j.shape}"
                    )
                x_hat_df = pd.DataFrame(
                    x_hat, index=raw_test_j.index, columns=list(ds.INPUT_COLUMNS)
                )

                bz = int(m.bytes_z(rec))
                bp = int(m.bytes_params())
                # Storage axis = the FRONT-END retained-scalar count, priced
                # as float64-equivalent bytes (8 * scalars). This strips the
                # byte-encoding back end (bit-packing / entropy coding) from
                # every family, so the comparison is over how much
                # information each family retains at a common precision —
                # not over which families happen to ship a packed encoder.
                # `bytes_z` (the real packed payload) is kept as a
                # transparency column for the secondary byte-CR view.
                # Model parameters (AE weights, PCA basis) remain excluded:
                # a one-shot deployment cost, not a per-window one.
                scalars = int(m.retained_scalars(rec))
                storage = 8 * scalars
                storage_rows.append({
                    "method": m.name,
                    "train_fold": k,
                    "eval_fold": j,
                    "bytes_z": bz,
                    "bytes_params": bp,
                    "storage": storage,
                    "cr": raw_bytes_test_j / max(1, storage),
                    "cr_bytes": raw_bytes_test_j / max(1, bz),
                    "cr_cells": m.cell_cr(n_channels=x_test_j.shape[1]),
                })

                for t in tasks_k:
                    t.evaluate_window(m.name, x_hat_df, raw_test_j)

            all_storage.append(pd.DataFrame(storage_rows))

            cell_summary = pd.concat(
                [t.finalize() for t in tasks_k], ignore_index=True
            )
            cell_summary["train_fold"] = k
            cell_summary["eval_fold"] = j
            all_summaries.append(cell_summary)
            print(f"[eval]   cell (k={k}, j={j}) done  "
                  f"n_test={len(raw_test_j)}  "
                  f"rows_emitted={len(cell_summary)}")

            # Drop sufficient stats so the next eval_fold starts clean; β
            # and the KNN classifier stay fitted (no refit between cells).
            for t in tasks_k:
                t.reset_accumulators()

    # ---- Concatenate cells, merge storage, normalise ----------------------
    summary = pd.concat(all_summaries, ignore_index=True)
    storage_df = pd.concat(all_storage, ignore_index=True)
    summary = summary.merge(
        storage_df, on=["method", "train_fold", "eval_fold"], how="left"
    )
    summary = normalise_loss(summary, tasks_proto, alpha=cfg.ALPHA)
    summary.to_csv(out_dir / "summary.csv", index=False)

    # ---- CV-aware argmin printout -----------------------------------------
    # Restricted to the diagonal (train_fold == eval_fold) — the standard
    # sliding-window CV view. The off-diagonal cells are inspected
    # downstream in the temporal-shift notebook section.
    for t in tasks_proto:
        sub = summary[
            (summary["task"] == t.name)
            & (summary["train_fold"] == summary["eval_fold"])
        ]

        # Per-fold argmin per subtask.
        idx = sub.groupby(list(t.subtask_columns) + ["eval_fold"])["multi_obj"].idxmin()
        best = sub.loc[idx, list(t.subtask_columns) + ["eval_fold", "method", "multi_obj"]]

        # Modal argmin across folds + agreement count.
        # Tasks like ClassificationTask have no subtask axis (no horizon, no
        # per-target split). pandas raises "No group keys passed!" on
        # groupby([]), so we collapse the whole-task case into a single
        # one-row frame computed directly from `best`.
        if t.subtask_columns:
            agreement = (
                best.groupby(list(t.subtask_columns))["method"]
                .agg(
                    modal_method=lambda s: s.mode().iat[0],
                    n_unique=lambda s: int(s.nunique()),
                    n_folds=lambda s: int(s.size),
                )
                .reset_index()
            )
        else:
            methods_by_fold = best["method"]
            agreement = pd.DataFrame([{
                "modal_method": methods_by_fold.mode().iat[0],
                "n_unique": int(methods_by_fold.nunique()),
                "n_folds": int(methods_by_fold.size),
            }])
        agreement["agreement_rate"] = (
            (agreement["n_unique"] == 1).astype(int)
        )  # 1 if all folds picked the same method, else 0

        n_full_agree = int(agreement["agreement_rate"].sum())
        n_total_sub = len(agreement)
        print(f"\n[eval] task='{t.name}' — argmin stability across folds:")
        print(f"  {n_full_agree}/{n_total_sub} subtasks have full cross-fold agreement")
        print(f"  modal winners ({len(agreement)} rows):")
        print(agreement.to_string(index=False))

    print(f"\n[eval] artefacts written to {out_dir}")


if __name__ == "__main__":
    main()
