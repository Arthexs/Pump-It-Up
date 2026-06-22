"""
Evaluation: metrics, confusion matrix, cost analysis, and SHAP values.

All functions return structured results (DataFrames, dicts) — not prints —
so they can be consumed by visualizations.py or saved to artifacts/reports/.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from pump.configs import CostMatrixConfig, EvaluationConfig


def compute_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
    eval_cfg: EvaluationConfig | None = None,
) -> dict[str, Any]:
    """Accuracy, F1-macro, F1-weighted, and per-class precision/recall/F1."""
    cfg = eval_cfg if eval_cfg is not None else EvaluationConfig()
    report: dict[str, Any] = classification_report(
        y_true, y_pred, labels=cfg.labels, output_dict=True, zero_division=0
    )
    per_class = {
        label: {
            "precision": report[label]["precision"],
            "recall": report[label]["recall"],
            "f1": report[label]["f1-score"],
        }
        for label in cfg.labels
        if label in report
    }
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": report["macro avg"]["f1-score"],
        "f1_weighted": report["weighted avg"]["f1-score"],
        "per_class": per_class,
    }


def confusion_matrix_df(
    y_true: pd.Series,
    y_pred: np.ndarray,
    eval_cfg: EvaluationConfig | None = None,
) -> pd.DataFrame:
    """Labeled confusion matrix as a DataFrame (rows=true, cols=predicted)."""
    cfg = eval_cfg if eval_cfg is not None else EvaluationConfig()
    cm = confusion_matrix(y_true, y_pred, labels=cfg.labels)
    return pd.DataFrame(cm, index=cfg.labels, columns=cfg.labels)


def cost_score(
    y_true: pd.Series,
    y_pred: np.ndarray,
    eval_cfg: EvaluationConfig | None = None,
    cost_cfg: CostMatrixConfig | None = None,
) -> float:
    """Total misclassification cost using the asymmetric cost matrix."""
    ecfg = eval_cfg if eval_cfg is not None else EvaluationConfig()
    ccfg = cost_cfg if cost_cfg is not None else CostMatrixConfig()
    matrix = np.array(ccfg.matrix)
    label_idx = {label: i for i, label in enumerate(ecfg.labels)}
    true_idx = np.array([label_idx[t] for t in y_true])
    pred_idx = np.array([label_idx[p] for p in y_pred])
    return float(matrix[true_idx, pred_idx].sum())


def cross_val_eval(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    eval_cfg: EvaluationConfig | None = None,
    cost_cfg: CostMatrixConfig | None = None,
    *,
    n_splits: int = 5,
    random_state: int = 42,
) -> dict[str, float]:
    """StratifiedKFold CV; returns mean ± std of accuracy, F1 scores, and cost."""
    ecfg = eval_cfg if eval_cfg is not None else EvaluationConfig()
    ccfg = cost_cfg if cost_cfg is not None else CostMatrixConfig()
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    fold_metrics: list[dict[str, float]] = []
    for train_idx, val_idx in cv.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        pipe = clone(pipeline)
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_val)
        m = compute_metrics(y_val, y_pred, ecfg)
        fold_metrics.append(
            {
                "accuracy": float(m["accuracy"]),
                "f1_macro": float(m["f1_macro"]),
                "f1_weighted": float(m["f1_weighted"]),
                "cost": cost_score(y_val, y_pred, ecfg, ccfg),
            }
        )

    result: dict[str, float] = {}
    for k in ("accuracy", "f1_macro", "f1_weighted", "cost"):
        vals = [fm[k] for fm in fold_metrics]
        result[f"{k}_mean"] = float(np.mean(vals))
        result[f"{k}_std"] = float(np.std(vals))
    return result
