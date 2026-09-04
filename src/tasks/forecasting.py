"""
tasks/forecasting.py

Forecasting task: one frozen β per (target column, horizon), shared
across all compression methods. β is fit offline on the raw training
segment by closed-form least squares (with intercept), and at evaluation
time is applied to the embedding built from x_hat with targets from raw.

Aggregation
-----------
Compression is per-window (that's the TSDB ingestion model), but
forecasting evaluation is global: we accumulate sufficient statistics
across all windows and report one (NMSE, NRMSE) per (method, target,
horizon) at the end. This avoids the per-batch Var(y) collapse that
inflates per-window NMSEs.

Reported metrics
----------------
- NMSE = MSE / Var(y) over the full test segment.
  Standard "is the model better than predicting the mean?" reference.
  Enters the multi-objective loss as `loss`.
- NRMSE = sqrt(MSE) / (max(y) - min(y)) over the full test segment.
  Matches the Muñiz-Cuza 2024 evaluation convention.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from features.build_features import (
    build_embedding,
    get_feature_column_names,
)
from tasks.base import TaskEvaluator


def _add_intercept(X: np.ndarray) -> np.ndarray:
    """Prepend a column of ones so the fitted β includes a bias term."""
    return np.column_stack([np.ones(X.shape[0]), X])


class ForecastingTask(TaskEvaluator):
    """
    Multi-horizon, multi-target forecasting task.

    Each subtask is one (target, horizon) cell. The reported loss is
    NMSE over the entire test segment.
    """

    name = "forecasting"
    subtask_columns = ["target", "horizon"]

    def __init__(
        self,
        input_columns: List[str],
        target_columns: List[str],
        window_size: int,
        horizons: Sequence[int],
    ):
        self.input_columns = list(input_columns)
        self.target_columns = list(target_columns)
        self.window_size = int(window_size)
        self.horizons = list(horizons)

        # beta[target][h] -> (1 + n_features,) array — first entry is the intercept
        self.beta: Dict[str, Dict[int, np.ndarray]] = {}
        self._feature_cols = get_feature_column_names(
            self.input_columns, self.window_size
        )

        # Sufficient-statistics accumulator, keyed by (method, target, h).
        # Carries everything finalize() needs to compute global NMSE / NRMSE
        # without retaining per-sample arrays.
        self._acc: Dict[Tuple[str, str, int], Dict[str, float]] = {}

    # ----- fit -------------------------------------------------------------

    def fit(self, raw_train: pd.DataFrame) -> "ForecastingTask":
        emb = build_embedding(
            raw_train,
            input_columns=self.input_columns,
            target_columns=self.target_columns,
            window_size=self.window_size,
            horizons=self.horizons,
        )
        X = _add_intercept(emb[self._feature_cols].to_numpy())

        self.beta = {col: {} for col in self.target_columns}
        for col in self.target_columns:
            for h in self.horizons:
                y = emb[f"{col}_tp{h}"].to_numpy()
                beta_h, *_ = np.linalg.lstsq(X, y, rcond=None)
                self.beta[col][h] = beta_h
        return self

    # ----- evaluate (accumulate) -------------------------------------------

    def _slot(self, key: Tuple[str, str, int]) -> Dict[str, float]:
        if key not in self._acc:
            self._acc[key] = {
                "sse": 0.0,       # Σ (y_true - y_pred)²
                "sum_y": 0.0,     # Σ y_true
                "sum_y2": 0.0,    # Σ y_true²
                "n": 0,           # sample count
                "y_min": float("inf"),
                "y_max": float("-inf"),
            }
        return self._acc[key]

    def evaluate_window(
        self,
        method_name: str,
        x_hat: pd.DataFrame,
        raw: pd.DataFrame,
    ) -> None:
        if len(x_hat) != len(raw):
            raise ValueError(
                f"x_hat ({len(x_hat)}) and raw ({len(raw)}) must be the same length"
            )

        emb = build_embedding(
            x_hat,
            input_columns=self.input_columns,
            target_columns=self.target_columns,
            window_size=self.window_size,
            horizons=self.horizons,
            target_df=raw,
        )
        if emb.empty:
            return
        X = _add_intercept(emb[self._feature_cols].to_numpy())

        for col in self.target_columns:
            for h in self.horizons:
                beta_h = self.beta[col][h]
                y_true = emb[f"{col}_tp{h}"].to_numpy()
                y_pred = X @ beta_h

                slot = self._slot((method_name, col, h))
                slot["sse"] += float(np.sum((y_true - y_pred) ** 2))
                slot["sum_y"] += float(np.sum(y_true))
                slot["sum_y2"] += float(np.sum(y_true ** 2))
                slot["n"] += int(len(y_true))
                slot["y_min"] = min(slot["y_min"], float(y_true.min()))
                slot["y_max"] = max(slot["y_max"], float(y_true.max()))

    # ----- finalize --------------------------------------------------------

    def finalize(self) -> pd.DataFrame:
        rows: List[Dict[str, Any]] = []
        for (method, col, h), slot in self._acc.items():
            n = slot["n"]
            if n == 0:
                continue
            mse = slot["sse"] / n
            mean_y = slot["sum_y"] / n
            var_y = slot["sum_y2"] / n - mean_y ** 2
            y_range = slot["y_max"] - slot["y_min"]
            nmse = mse / var_y if var_y > 0 else float("nan")
            nrmse = float(np.sqrt(mse)) / y_range if y_range > 0 else float("nan")
            rows.append({
                "task": self.name,
                "method": method,
                "target": col,
                "horizon": h,
                "loss": nmse,
                "nmse": nmse,
                "nrmse": nrmse,
                "rmse": float(np.sqrt(mse)),
                "mse": mse,
                "n_samples": n,
            })
        return pd.DataFrame(rows)

    # ----- reset -----------------------------------------------------------

    def reset_accumulators(self) -> None:
        """Drop sufficient statistics; leaves the fitted β untouched."""
        self._acc.clear()

    # ----- introspection ---------------------------------------------------

    def n_features(self) -> int:
        # Excludes the intercept column from the feature count.
        return len(self._feature_cols)
