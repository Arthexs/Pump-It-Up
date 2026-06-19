"""
Evaluation: metrics, confusion matrix, cost analysis, and SHAP values.

All functions return structured results (DataFrames, dicts) — not prints —
so they can be consumed by visualizations.py or saved to artifacts/reports/.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pump.config.evaluation import CostMatrixConfig, EvaluationConfig, SHAPConfig


def accuracy(y_true: pd.Series, y_pred: np.ndarray) -> float:
    """Overall accuracy — the competition's scoring metric."""
    raise NotImplementedError


def classification_report_df(
    y_true: pd.Series,
    y_pred: np.ndarray,
    cfg: EvaluationConfig | None = None,
) -> pd.DataFrame:
    """
    Per-class precision, recall, F1 as a DataFrame.
    Focus on 'functional needs repair' — smallest and hardest class.
    """
    raise NotImplementedError


def confusion_matrix_df(
    y_true: pd.Series,
    y_pred: np.ndarray,
    normalize: bool = False,
) -> pd.DataFrame:
    """
    Labeled confusion matrix DataFrame.
    normalize=True gives row-wise recall rates — useful for imbalanced classes.
    """
    raise NotImplementedError


def expected_cost(
    y_true: pd.Series,
    y_pred: np.ndarray,
    cfg: CostMatrixConfig | None = None,
) -> dict[str, float]:
    """
    Asymmetric misclassification cost given a cost matrix.
    Returns total cost, avg cost per sample, and breakdown by true class.
    Key slide for the business case.
    """
    raise NotImplementedError


def compute_shap(
    pipeline: object,
    X_transformed: pd.DataFrame,
    cfg: SHAPConfig | None = None,
) -> tuple[Any, pd.DataFrame]:
    """
    Compute SHAP values for the pipeline's fitted estimator.
    Returns (shap_values, importance_df) where importance_df has mean |SHAP| per feature.
    X_transformed should be the output of pipeline.transform(X_val).
    """
    raise NotImplementedError


def full_report(
    y_true: pd.Series,
    y_pred: np.ndarray,
    cfg: EvaluationConfig | None = None,
    cost_cfg: CostMatrixConfig | None = None,
) -> dict[str, object]:
    """Run all evaluation functions. Returns named dict of results."""
    raise NotImplementedError
