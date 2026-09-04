"""
datasets/__init__.py

Auto-discovers all dataset config modules in this directory.
Each .py file (except __init__.py) is imported and registered in DATASETS.

To add a new dataset, create src/datasets/<name>.py exposing at least:
    INPUT_COLUMNS            : List[str]
    TARGET_COLUMNS           : List[str]
    CLASSIFICATION_LABEL_COL : str | None   (None for forecast-only datasets)

Nothing else in the codebase needs to change.
"""

import importlib
import os

DATASETS: dict = {}

_datasets_dir = os.path.dirname(__file__)
for _fname in sorted(os.listdir(_datasets_dir)):
    if _fname.endswith(".py") and not _fname.startswith("_"):
        _name = _fname[:-3]
        DATASETS[_name] = importlib.import_module(f"datasets.{_name}")
