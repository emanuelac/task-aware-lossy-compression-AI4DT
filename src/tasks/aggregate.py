"""
tasks/aggregate.py

Window-aggregate query task: for each fixed-size query window we compute
a scalar statistic on raw (the ground truth) and on x_hat (the
reconstruction), then score the method by how closely the reconstructed
aggregate series tracks the true one.

Why this task exists
--------------------
Real TSDB workloads aren't only point forecasts. A large share of
query traffic is windowed aggregates: mean / max / min / std over a
fixed time window. Reconstruction NMSE doesn't compose linearly into
aggregate error: a compressor with great NMSE can still blow up the
max (DCT smooths peaks), and one that preserves extrema (LTTB) can
miss the mean. So we expect different compressor families to win
different agg subtasks — exactly the kind of task heterogeneity the
task-aware-vs-task-agnostic claim needs.

Each aggregate stat in `self.aggs` becomes one subtask row per method.
Channels are an internal axis: we compute per-(method, agg, channel)
NMSE on the aggregate series and average over channels at finalize
time, so the reported `loss` is one scalar per (method, agg). NMSE
(mse / var) is the same variance-normalised family as the forecasting
task's loss, so both fidelity axes are reported on one consistent
metric; the per-channel NRMSE (sqrt of NMSE) is retained as a
reference column.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from tasks.base import TaskEvaluator


_AGG_FUNCS = {
    "mean": np.mean,
    "max": np.max,
    "min": np.min,
    "std": np.std,
}


class AggregateQueryTask(TaskEvaluator):
    """
    Per-window aggregate-query task.

    Parameters
    ----------
    target_columns : list of str
        Channels on which the aggregates are computed. Same set as the
        forecasting task's targets, so storage / NMSE / aggregate error
        all reference the same channel set.
    window : int
        Query-window size in samples. Independent of cfg.WINDOW_SIZE
        (time-delay embedding) and cfg.SPECTRAL_WINDOW_SIZE.
    aggs : tuple of str
        Aggregate stats to score. Default {mean, max, min, std} —
        cardinal SQL/Flux aggregates, with the property that each
        rewards a different geometric feature of the reconstruction.
    """

    name = "aggregate_query"
    subtask_columns: List[str] = ["agg"]

    def __init__(
        self,
        target_columns: List[str],
        window: int = 64,
        aggs: Sequence[str] = ("mean", "max", "min", "std"),
    ):
        if window < 2:
            raise ValueError(f"window must be >= 2, got {window}")
        unknown = [a for a in aggs if a not in _AGG_FUNCS]
        if unknown:
            raise ValueError(
                f"unknown aggregates {unknown}; supported: {list(_AGG_FUNCS)}"
            )
        self.target_columns = list(target_columns)
        self.window = int(window)
        self.aggs = tuple(aggs)

        # Accumulator keyed by (method, agg, channel) — channels collapse
        # to a single mean at finalize time.
        self._acc: Dict[Tuple[str, str, str], Dict[str, float]] = {}

    # ----- fit -------------------------------------------------------------

    def fit(self, raw_train: pd.DataFrame) -> "AggregateQueryTask":
        # No parameters to learn — aggregates are deterministic
        # functions of the window. We just sanity-check column presence
        # so downstream evaluate_window can assume the columns exist.
        missing = [c for c in self.target_columns if c not in raw_train.columns]
        if missing:
            raise KeyError(
                f"target columns missing from raw_train: {missing}"
            )
        return self

    # ----- helpers ---------------------------------------------------------

    def _slot(self, key: Tuple[str, str, str]) -> Dict[str, float]:
        if key not in self._acc:
            self._acc[key] = {
                "sse": 0.0,     # Σ (a_true - a_pred)²  over windows
                "sum_y": 0.0,   # Σ a_true              over windows
                "sum_y2": 0.0,  # Σ a_true²             over windows
                "n": 0,         # number of windows
                "y_min": float("inf"),
                "y_max": float("-inf"),
            }
        return self._acc[key]

    def _tile_windows(self, arr: np.ndarray) -> np.ndarray:
        """Reshape a 1D array into (n_win, window), dropping any tail."""
        n_win = arr.shape[0] // self.window
        if n_win == 0:
            return np.empty((0, self.window), dtype=arr.dtype)
        return arr[: n_win * self.window].reshape(n_win, self.window)

    # ----- evaluate (accumulate) -------------------------------------------

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
        if len(raw) < self.window:
            return  # nothing to aggregate

        for ch in self.target_columns:
            raw_arr = raw[ch].to_numpy(dtype=float)
            hat_arr = x_hat[ch].to_numpy(dtype=float)

            raw_win = self._tile_windows(raw_arr)
            hat_win = self._tile_windows(hat_arr)
            if raw_win.shape[0] == 0:
                continue

            for agg in self.aggs:
                fn = _AGG_FUNCS[agg]
                a_true = fn(raw_win, axis=1)
                a_pred = fn(hat_win, axis=1)

                slot = self._slot((method_name, agg, ch))
                slot["sse"] += float(np.sum((a_true - a_pred) ** 2))
                slot["sum_y"] += float(np.sum(a_true))
                slot["sum_y2"] += float(np.sum(a_true ** 2))
                slot["n"] += int(a_true.shape[0])
                slot["y_min"] = min(slot["y_min"], float(a_true.min()))
                slot["y_max"] = max(slot["y_max"], float(a_true.max()))

    # ----- finalize --------------------------------------------------------

    def finalize(self) -> pd.DataFrame:
        # Collapse the channel axis: per (method, agg) we average the
        # per-channel NMSE (mse / var) so channels with larger absolute
        # scale don't dominate the reported loss. NMSE is the same
        # variance-normalised family as the forecasting task's loss, so
        # both fidelity axes are reported consistently. The channel-mean
        # NRMSE (sqrt of the per-channel NMSE) is kept for reference.
        nmse_by: Dict[Tuple[str, str], List[float]] = {}
        nrmse_by: Dict[Tuple[str, str], List[float]] = {}
        sample_counts: Dict[Tuple[str, str], int] = {}
        for (method, agg, _ch), slot in self._acc.items():
            n = slot["n"]
            if n == 0:
                continue
            mse = slot["sse"] / n
            mean_y = slot["sum_y"] / n
            var_y = slot["sum_y2"] / n - mean_y ** 2
            if var_y > 0:
                nmse = mse / var_y
                nrmse = float(np.sqrt(nmse))
            else:
                # Degenerate channel: constant aggregate series. Skip the
                # contribution — there's nothing meaningful to score.
                continue
            nmse_by.setdefault((method, agg), []).append(nmse)
            nrmse_by.setdefault((method, agg), []).append(nrmse)
            sample_counts[(method, agg)] = sample_counts.get((method, agg), 0) + n

        rows: List[Dict[str, Any]] = []
        for (method, agg), nmses in nmse_by.items():
            mean_nmse = float(np.mean(nmses))
            mean_nrmse = float(np.mean(nrmse_by[(method, agg)]))
            rows.append({
                "task": self.name,
                "method": method,
                "agg": agg,
                "loss": mean_nmse,
                "nmse": mean_nmse,
                "nrmse": mean_nrmse,
                "n_channels_scored": len(nmses),
                "n_samples": sample_counts[(method, agg)],
            })
        return pd.DataFrame(rows)

    # ----- reset -----------------------------------------------------------

    def reset_accumulators(self) -> None:
        self._acc.clear()
