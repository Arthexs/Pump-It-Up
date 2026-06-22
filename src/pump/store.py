"""
Model persistence: save, load, list, and delete trained Pipeline objects.

Naming convention: {name}_{YYYYMMDD_HHMMSS_ffffff}.joblib
Microsecond precision prevents collisions on rapid successive saves.

Usage:
    store.save(pipeline, name="xgb_baseline")
    # → artifacts/models/xgb_baseline_20250619_143022_001234.joblib

    pipeline = store.load("xgb_baseline_20250619_143022_001234")
    pipeline = store.load_latest("xgb_baseline")

    store.list_models()
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

_DEFAULT_DIR = Path("artifacts/models")
_TS_FMT = "%Y%m%d_%H%M%S_%f"
_TS_RE = re.compile(r"^(.+)_(\d{8}_\d{6}_\d{6})$")


def _resolve_dir(models_dir: Path | str | None) -> Path:
    return Path(models_dir) if models_dir is not None else _DEFAULT_DIR


def save(
    pipeline: Pipeline,
    name: str,
    *,
    models_dir: Path | str | None = None,
) -> Path:
    """Persist pipeline to {models_dir}/{name}_{YYYYMMDD_HHMMSS_ffffff}.joblib."""
    out_dir = _resolve_dir(models_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    while True:
        ts = datetime.now().strftime(_TS_FMT)
        path = out_dir / f"{name}_{ts}.joblib"
        if not path.exists():
            break
    joblib.dump(pipeline, path)
    return path


def load(
    stem: str,
    *,
    models_dir: Path | str | None = None,
) -> Pipeline:
    """Load a pipeline by its full stem (without .joblib extension)."""
    path = _resolve_dir(models_dir) / f"{stem}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")
    return joblib.load(path)


def load_latest(
    name: str,
    *,
    models_dir: Path | str | None = None,
) -> Pipeline:
    """Load the most recently saved pipeline whose stem is exactly {name}_{timestamp}."""
    out_dir = _resolve_dir(models_dir)
    exact = re.compile(rf"^{re.escape(name)}_\d{{8}}_\d{{6}}_\d{{6}}$")
    matches = sorted(p for p in out_dir.glob(f"{name}_*.joblib") if exact.match(p.stem))
    if not matches:
        raise FileNotFoundError(f"No saved models found for name '{name}' in {out_dir}")
    return joblib.load(matches[-1])


def list_models(
    *,
    models_dir: Path | str | None = None,
) -> pd.DataFrame:
    """Return a DataFrame of saved models: stem, name, saved_at, path."""
    out_dir = _resolve_dir(models_dir)
    if not out_dir.exists():
        return pd.DataFrame(columns=["stem", "name", "saved_at", "path"])
    rows = []
    for path in sorted(out_dir.glob("*.joblib")):
        stem = path.stem
        m = _TS_RE.match(stem)
        if m:
            model_name = m.group(1)
            saved_at: datetime | None = datetime.strptime(m.group(2), _TS_FMT)
        else:
            model_name, saved_at = stem, None
        rows.append({"stem": stem, "name": model_name, "saved_at": saved_at, "path": path})
    return pd.DataFrame(rows, columns=["stem", "name", "saved_at", "path"])
