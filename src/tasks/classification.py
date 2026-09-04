"""
tasks/classification.py

Per-window time-series classification with 1-NN Dynamic Time Warping on the
time-delay embedding. One classifier per (compressor, fold); no subtask axis.
Loss = 1 - macro-F1 (equal weight per class, so majority-class prediction is
not rewarded). The banded-DTW kernel lives in tasks/_dtw.py.
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from features.build_features import build_embedding, get_feature_column_names
from tasks.base import TaskEvaluator


class ClassificationTask(TaskEvaluator):
    name = "classification"
    subtask_columns: List[str] = []

    def __init__(
        self,
        input_columns: List[str],
        label_column: str,
        window_size: int,
        train_subsample: int | None = None,
        subsample_seed: int = 0,
        band_frac: float = 0.1,
        query_subsample: int | None = 3000,
    ):
        self.input_columns = list(input_columns)
        self.label_column = str(label_column)
        self.window_size = int(window_size)
        self.train_subsample = (
            int(train_subsample) if train_subsample is not None else None
        )
        self.subsample_seed = int(subsample_seed)
        self.band_frac = float(band_frac)
        self.query_subsample = int(query_subsample) if query_subsample else None
        # build_embedding needs a target column to fix row alignment; the label
        # comes from raw[label_column] separately, so this choice is cosmetic.
        self._anchor_target = self.input_columns[0]
        self._feature_cols = get_feature_column_names(
            self.input_columns, self.window_size
        )
        self.classes_: np.ndarray | None = None
        self._Xref: np.ndarray | None = None   # (n_ref, L, C)
        self._yref: np.ndarray | None = None   # (n_ref,) class indices
        self._L: int | None = None
        self._C: int | None = None
        self._band: int | None = None
        self._acc: Dict[str, Dict[str, Any]] = {}

    def _labels_for_embedding(self, raw: pd.DataFrame, emb: pd.DataFrame) -> np.ndarray:
        return raw.loc[emb.index, self.label_column].to_numpy()

    def _to_series(self, X_flat: np.ndarray) -> np.ndarray:
        # get_feature_column_names is channel-major, so reshape to (n, C, L)
        # then transpose to time-major (L, C) for the DTW kernel.
        n = X_flat.shape[0]
        return np.ascontiguousarray(
            X_flat.reshape(n, self._C, self._L).transpose(0, 2, 1), dtype=np.float64
        )

    def fit(self, raw_train: pd.DataFrame) -> "ClassificationTask":
        if self.label_column not in raw_train.columns:
            raise KeyError(
                f"label column '{self.label_column}' not found in raw_train"
            )
        emb = build_embedding(
            raw_train,
            input_columns=self.input_columns,
            target_columns=[self._anchor_target],
            window_size=self.window_size,
            horizons=[1],
        )
        X = emb[self._feature_cols].to_numpy()
        y = self._labels_for_embedding(raw_train, emb)
        if self.train_subsample is not None and len(X) > self.train_subsample:
            rng = np.random.default_rng(self.subsample_seed)
            idx = rng.choice(len(X), size=self.train_subsample, replace=False)
            X, y = X[idx], y[idx]

        self._C = len(self.input_columns)
        self._L = self.window_size - 1
        self._band = max(1, int(self.band_frac * self._L))
        self.classes_ = np.unique(y)
        cls_to_idx = {c: i for i, c in enumerate(self.classes_)}
        self._yref = np.array([cls_to_idx[v] for v in y], dtype=np.int64)
        self._Xref = self._to_series(X)
        return self

    def _slot(self, method: str) -> Dict[str, Any]:
        if method not in self._acc:
            n_c = len(self.classes_) if self.classes_ is not None else 0
            self._acc[method] = {
                "n_correct": 0,
                "n_total": 0,
                "confusion": np.zeros((n_c, n_c), dtype=int),
            }
        return self._acc[method]

    def evaluate_window(self, method_name: str, x_hat: pd.DataFrame, raw: pd.DataFrame) -> None:
        if self._Xref is None:
            raise RuntimeError("ClassificationTask must be fit before evaluate")
        emb = build_embedding(
            x_hat,
            input_columns=self.input_columns,
            target_columns=[self._anchor_target],
            window_size=self.window_size,
            horizons=[1],
            target_df=raw,
        )
        if emb.empty:
            return
        X = emb[self._feature_cols].to_numpy()
        y_true = self._labels_for_embedding(raw, emb)

        # Deterministic stride subsample of the query windows (same positions
        # for every compressor) to keep the DTW cost tractable.
        m = len(X)
        if self.query_subsample is not None and m > self.query_subsample:
            sel = np.arange(0, m, int(np.ceil(m / self.query_subsample)))
            X, y_true = X[sel], y_true[sel]

        from ._dtw import dtw_1nn_predict
        Xq = self._to_series(X)
        y_pred = self.classes_[dtw_1nn_predict(Xq, self._Xref, self._yref, self._band)]

        slot = self._slot(method_name)
        slot["n_correct"] += int((y_pred == y_true).sum())
        slot["n_total"] += int(len(y_true))
        cls_to_idx = {c: i for i, c in enumerate(self.classes_)}
        mask = np.array(
            [t in cls_to_idx and p in cls_to_idx for t, p in zip(y_true, y_pred)]
        )
        if mask.any():
            ti = np.array([cls_to_idx[t] for t in y_true[mask]])
            pi = np.array([cls_to_idx[p] for p in y_pred[mask]])
            np.add.at(slot["confusion"], (ti, pi), 1)

    def finalize(self) -> pd.DataFrame:
        rows: List[Dict[str, Any]] = []
        for method, slot in self._acc.items():
            n = slot["n_total"]
            if n == 0:
                continue
            cm = slot["confusion"].astype(float)
            tp = np.diag(cm)
            fp = cm.sum(axis=0) - tp
            fn = cm.sum(axis=1) - tp
            precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
            recall = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
            f1 = np.divide(
                2 * precision * recall, precision + recall,
                out=np.zeros_like(tp), where=(precision + recall) > 0,
            )
            macro_f1 = float(f1.mean())
            rows.append({
                "task": self.name,
                "method": method,
                "loss": 1.0 - macro_f1,
                "accuracy": slot["n_correct"] / n,
                "macro_f1": macro_f1,
                "n_samples": n,
                "n_classes": int(len(self.classes_)) if self.classes_ is not None else 0,
            })
        return pd.DataFrame(rows)

    def reset_accumulators(self) -> None:
        self._acc.clear()
