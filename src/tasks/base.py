"""
tasks/base.py

TaskEvaluator protocol — the contract every downstream task implements.

Compression is per-window in a TSDB (each ingest batch is compressed
independently), but the *evaluation* of a frozen task model is global:
methods and β are trained once on raw_train and never touched again, so
there's no operational reason to compute loss per-window. The protocol
mirrors that: `evaluate_window` accumulates sufficient statistics, and
`finalize` produces one row per (method, subtask) with the global loss.

Each subtask in a task family is identified by the columns listed in
`subtask_columns`. Forecasting uses ["target", "horizon"]; anomaly
detection will use something like ["detector"].
"""

from __future__ import annotations

from typing import List

import pandas as pd


class TaskEvaluator:
    """
    Base class for downstream task evaluators.

    Attributes
    ----------
    name : str
        Short identifier — appears as the `task` column in result rows.
    subtask_columns : list of str
        Names of the columns that, together with `task` and `method`,
        uniquely identify a (method, subtask) result. Used by the runner
        to group rows for normalisation across methods.
    """

    name: str = "task"
    subtask_columns: List[str] = []

    # ----- lifecycle -------------------------------------------------------

    def fit(self, raw_train: pd.DataFrame) -> "TaskEvaluator":
        """Fit frozen models from the raw training segment. Returns self."""
        raise NotImplementedError

    def evaluate_window(
        self,
        method_name: str,
        x_hat: pd.DataFrame,
        raw: pd.DataFrame,
    ) -> None:
        """
        Score one (window, method) pair and add the result to the task's
        internal accumulator. No return value — finalize() emits the summary.

        x_hat is the method's reconstruction (feature source); raw is the
        un-compressed truth (target source).
        """
        raise NotImplementedError

    def finalize(self) -> pd.DataFrame:
        """
        Emit per-(method, subtask) summary rows accumulated by
        evaluate_window. Each row contains:
          - "task" : str  (== self.name)
          - "method" : str
          - one entry for each col in self.subtask_columns
          - "loss" : float  (the scalar that enters the multi-objective L)
          - any auxiliary metrics the task wants to report
        """
        raise NotImplementedError

    def reset_accumulators(self) -> None:
        """
        Drop the sufficient statistics accumulated since the last fit/reset,
        without touching the fitted frozen models (β / classifier weights).
        Called between cells of the cross-evaluation matrix so the same
        fitted task can be reused across multiple eval blocks.
        """
        raise NotImplementedError
