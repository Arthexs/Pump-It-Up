"""
Model persistence: save, load, list, and delete trained Pipeline objects.

Naming convention: {name}_{YYYYMMDD_HHMMSS}.joblib
This gives free versioning — multiple runs are never overwritten, and you
can always tell when a model was trained.

Usage:
    store.save(pipeline, name="xgb_baseline")
    # → artifacts/models/xgb_baseline_20250619_143022.joblib

    pipeline = store.load("xgb_baseline_20250619_143022")
    pipeline = store.load_latest("xgb_baseline")

    store.list_models()
"""

from __future__ import annotations

import os
from pathlib import Path

_DEFAULT_DIR = Path(os.getenv("MODEL_DIR", "./artifacts/models"))


def save(pipeline: object, name: str, model_dir: Path | None = None) -> Path:
    """Serialize pipeline to disk with a timestamp suffix."""
    raise NotImplementedError


def load(stem: str, model_dir: Path | None = None) -> object:
    """Load a model by its full stem, e.g. 'xgb_baseline_20250619_143022'."""
    raise NotImplementedError


def load_latest(name: str, model_dir: Path | None = None) -> object:
    """Load the most recently saved model whose filename starts with name."""
    raise NotImplementedError


def list_models(model_dir: Path | None = None) -> list[dict[str, str]]:
    """Return metadata for all saved models: name, stem, path, saved_at, size_mb."""
    raise NotImplementedError


def delete(stem: str, model_dir: Path | None = None) -> None:
    """Delete a saved model by its full stem. Irreversible."""
    raise NotImplementedError
