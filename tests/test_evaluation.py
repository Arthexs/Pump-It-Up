"""Tests for pump.evaluation."""

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import Pipeline

from pump.configs import CostMatrixConfig, EvaluationConfig
from pump.evaluation import (
    compute_metrics,
    confusion_matrix_df,
    cost_score,
    cross_val_eval,
)

LABELS = ["functional", "functional needs repair", "non functional"]
EVAL_CFG = EvaluationConfig(labels=LABELS)
COST_CFG = CostMatrixConfig()


@pytest.fixture
def perfect_pred() -> tuple[pd.Series, np.ndarray]:
    y = pd.Series(LABELS * 30, name="status_group")
    return y, np.array(y)


@pytest.fixture
def toy_xy() -> tuple[pd.DataFrame, pd.Series]:
    n = 90
    X = pd.DataFrame({"a": np.zeros(n)})
    y = pd.Series(LABELS * 30, name="status_group")
    return X, y


@pytest.fixture
def dummy_pipe() -> Pipeline:
    return Pipeline([("clf", DummyClassifier(strategy="most_frequent", random_state=0))])


# ── compute_metrics ────────────────────────────────────────────────────────────


class TestComputeMetrics:
    def test_perfect_scores(self, perfect_pred):
        y_true, y_pred = perfect_pred
        m = compute_metrics(y_true, y_pred, EVAL_CFG)
        assert m["accuracy"] == pytest.approx(1.0)
        assert m["f1_macro"] == pytest.approx(1.0)
        assert m["f1_weighted"] == pytest.approx(1.0)

    def test_per_class_labels(self, perfect_pred):
        y_true, y_pred = perfect_pred
        m = compute_metrics(y_true, y_pred, EVAL_CFG)
        assert set(m["per_class"].keys()) == set(LABELS)

    def test_per_class_fields(self, perfect_pred):
        y_true, y_pred = perfect_pred
        m = compute_metrics(y_true, y_pred, EVAL_CFG)
        for label in LABELS:
            assert set(m["per_class"][label].keys()) == {"precision", "recall", "f1"}

    def test_defaults_when_no_cfg(self, perfect_pred):
        y_true, y_pred = perfect_pred
        m = compute_metrics(y_true, y_pred)
        assert "accuracy" in m and "f1_macro" in m


# ── confusion_matrix_df ────────────────────────────────────────────────────────


class TestConfusionMatrixDf:
    def test_shape(self, perfect_pred):
        y_true, y_pred = perfect_pred
        cm = confusion_matrix_df(y_true, y_pred, EVAL_CFG)
        assert cm.shape == (3, 3)

    def test_index_and_columns(self, perfect_pred):
        y_true, y_pred = perfect_pred
        cm = confusion_matrix_df(y_true, y_pred, EVAL_CFG)
        assert list(cm.index) == LABELS
        assert list(cm.columns) == LABELS

    def test_perfect_is_diagonal(self, perfect_pred):
        y_true, y_pred = perfect_pred
        cm = confusion_matrix_df(y_true, y_pred, EVAL_CFG)
        assert (np.diag(cm.values) == 30).all()
        assert cm.values.sum() - np.diag(cm.values).sum() == 0


# ── cost_score ─────────────────────────────────────────────────────────────────


class TestCostScore:
    def test_perfect_is_zero(self, perfect_pred):
        y_true, y_pred = perfect_pred
        assert cost_score(y_true, y_pred, EVAL_CFG, COST_CFG) == pytest.approx(0.0)

    def test_single_worst_case(self):
        # true 'non functional' predicted as 'functional' — cost matrix row 2, col 0 → 5.0
        y_true = pd.Series(["non functional"])
        y_pred = np.array(["functional"])
        assert cost_score(y_true, y_pred, EVAL_CFG, COST_CFG) == pytest.approx(5.0)

    def test_all_predicted_functional(self, perfect_pred):
        y_true, _ = perfect_pred
        y_pred = np.array(["functional"] * len(y_true))
        # 30 × needs_repair→functional (2.0) + 30 × non_functional→functional (5.0)
        assert cost_score(y_true, y_pred, EVAL_CFG, COST_CFG) == pytest.approx(30 * 2.0 + 30 * 5.0)

    def test_returns_float(self, perfect_pred):
        y_true, y_pred = perfect_pred
        assert isinstance(cost_score(y_true, y_pred, EVAL_CFG, COST_CFG), float)


# ── cross_val_eval ─────────────────────────────────────────────────────────────


class TestCrossValEval:
    def test_returns_expected_keys(self, toy_xy, dummy_pipe):
        X, y = toy_xy
        result = cross_val_eval(dummy_pipe, X, y, EVAL_CFG, COST_CFG, n_splits=3)
        expected = {
            "accuracy_mean",
            "accuracy_std",
            "f1_macro_mean",
            "f1_macro_std",
            "f1_weighted_mean",
            "f1_weighted_std",
            "cost_mean",
            "cost_std",
        }
        assert set(result.keys()) == expected

    def test_all_values_are_floats(self, toy_xy, dummy_pipe):
        X, y = toy_xy
        result = cross_val_eval(dummy_pipe, X, y, EVAL_CFG, COST_CFG, n_splits=3)
        assert all(isinstance(v, float) for v in result.values())

    def test_std_nonnegative(self, toy_xy, dummy_pipe):
        X, y = toy_xy
        result = cross_val_eval(dummy_pipe, X, y, EVAL_CFG, COST_CFG, n_splits=3)
        assert all(result[k] >= 0.0 for k in result if k.endswith("_std"))

    def test_does_not_fit_original_pipeline(self, toy_xy, dummy_pipe):
        X, y = toy_xy
        cross_val_eval(dummy_pipe, X, y, EVAL_CFG, COST_CFG, n_splits=2)
        with pytest.raises(NotFittedError):
            dummy_pipe.predict(X)
