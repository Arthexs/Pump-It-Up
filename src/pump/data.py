"""
Typed data loading, schema validation, and train/test splitting.

Single entry point for raw data — nothing else reads CSVs directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


LABEL_COL = "status_group"
LABELS = ["functional", "functional needs repair", "non functional"]
LABEL_TO_INT: dict[str, int] = {label: i for i, label in enumerate(LABELS)}
INT_TO_LABEL: dict[int, str] = {i: label for label, i in LABEL_TO_INT.items()}


@dataclass
class RawDataset:
    X: pd.DataFrame
    y: pd.Series        # integer-encoded
    y_raw: pd.Series    # string labels


@dataclass
class DataSplit:
    X_train: pd.DataFrame
    X_val: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_train_raw: pd.Series
    y_val_raw: pd.Series


def load(features_path: str | Path, labels_path: str | Path) -> RawDataset:
    """Load and join feature + label CSVs. Returns a typed RawDataset."""
    raise NotImplementedError


def load_test(features_path: str | Path) -> pd.DataFrame:
    """Load the held-out test set (no labels)."""
    raise NotImplementedError


def split(
    dataset: RawDataset,
    val_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True,
) -> DataSplit:
    """Stratified train/val split. Stratify to preserve minority class ratio."""
    raise NotImplementedError
