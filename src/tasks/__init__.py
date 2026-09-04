"""
tasks/

Downstream task evaluators. Each TaskEvaluator scores one task family against
a compressor: the runner hands it a window of x_hat (reconstructed) and raw,
and it returns one row per subtask with a scalar loss.
"""

from tasks.aggregate import AggregateQueryTask
from tasks.base import TaskEvaluator
from tasks.classification import ClassificationTask
from tasks.forecasting import ForecastingTask

__all__ = [
    "TaskEvaluator",
    "ForecastingTask",
    "ClassificationTask",
    "AggregateQueryTask",
]
