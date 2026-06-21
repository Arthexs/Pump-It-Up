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
