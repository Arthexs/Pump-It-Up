"""Tests for pump.models."""

import numpy as np
import pandas as pd
import pytest

from pump.configs import LGBMConfig, LogisticRegressionConfig, RandomForestConfig, XGBConfig
from pump.models import (
    LGBMEstimator,
    LogisticRegressionEstimator,
    RandomForestEstimator,
    XGBEstimator,
)
from pump.registry import ESTIMATORS

CLASSES = ["functional", "functional needs repair", "non functional"]


@pytest.fixture
def toy_data() -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(0)
    n = 90  # 30 per class, balanced
    X = pd.DataFrame(
        {
            "a": rng.standard_normal(n),
            "b": rng.standard_normal(n),
            "c": rng.standard_normal(n),
            "d": rng.standard_normal(n),
        }
    )
    y = pd.Series(CLASSES * 30, name="status_group")
    return X, y


# ── Registry ───────────────────────────────────────────────────────────────────


class TestRegistry:
    def test_all_four_models_registered(self):
        assert set(ESTIMATORS.keys()) >= {"logistic_regression", "random_forest", "xgb", "lgbm"}

    def test_get_returns_correct_class(self):
        assert ESTIMATORS.get("logistic_regression") is LogisticRegressionEstimator
        assert ESTIMATORS.get("random_forest") is RandomForestEstimator
        assert ESTIMATORS.get("xgb") is XGBEstimator
        assert ESTIMATORS.get("lgbm") is LGBMEstimator


# ── Shared contract ────────────────────────────────────────────────────────────
#
# Each model must satisfy the same interface. These helpers are called from
# every model-specific test class to avoid duplication.


def _assert_predict_contract(estimator, X, y):
    estimator.fit(X, y)
    preds = estimator.predict(X)

    assert len(preds) == len(X)
    assert set(preds).issubset(set(CLASSES))


def _assert_predict_proba_contract(estimator, X, y):
    estimator.fit(X, y)
    proba = estimator.predict_proba(X)

    assert proba.shape == (len(X), 3)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def _assert_classes_contract(estimator, X, y):
    estimator.fit(X, y)
    assert hasattr(estimator, "classes_")
    assert set(estimator.classes_) == set(CLASSES)


def _assert_feature_importances_contract(estimator, X, y):
    estimator.fit(X, y)
    fi = estimator.feature_importances_
    assert fi.shape == (X.shape[1],)
    assert np.all(fi >= 0)


# ── LogisticRegressionEstimator ────────────────────────────────────────────────


class TestLogisticRegressionEstimator:
    def test_default_config(self):
        est = LogisticRegressionEstimator()
        assert est.cfg == LogisticRegressionConfig()

    def test_custom_config(self):
        cfg = LogisticRegressionConfig(C=0.1, max_iter=200)
        est = LogisticRegressionEstimator(cfg)
        assert est.cfg.C == 0.1

    def test_predict(self, toy_data):
        _assert_predict_contract(LogisticRegressionEstimator(), *toy_data)

    def test_predict_proba(self, toy_data):
        _assert_predict_proba_contract(LogisticRegressionEstimator(), *toy_data)

    def test_classes(self, toy_data):
        _assert_classes_contract(LogisticRegressionEstimator(), *toy_data)

    def test_feature_importances(self, toy_data):
        _assert_feature_importances_contract(LogisticRegressionEstimator(), *toy_data)

    def test_fit_returns_self(self, toy_data):
        X, y = toy_data
        est = LogisticRegressionEstimator()
        assert est.fit(X, y) is est


# ── RandomForestEstimator ──────────────────────────────────────────────────────


class TestRandomForestEstimator:
    def test_default_config(self):
        est = RandomForestEstimator()
        assert est.cfg == RandomForestConfig()

    def test_custom_config(self):
        cfg = RandomForestConfig(n_estimators=10, max_depth=3)
        est = RandomForestEstimator(cfg)
        assert est.cfg.n_estimators == 10

    def test_predict(self, toy_data):
        cfg = RandomForestConfig(n_estimators=10)
        _assert_predict_contract(RandomForestEstimator(cfg), *toy_data)

    def test_predict_proba(self, toy_data):
        cfg = RandomForestConfig(n_estimators=10)
        _assert_predict_proba_contract(RandomForestEstimator(cfg), *toy_data)

    def test_classes(self, toy_data):
        cfg = RandomForestConfig(n_estimators=10)
        _assert_classes_contract(RandomForestEstimator(cfg), *toy_data)

    def test_feature_importances(self, toy_data):
        cfg = RandomForestConfig(n_estimators=10)
        _assert_feature_importances_contract(RandomForestEstimator(cfg), *toy_data)

    def test_fit_returns_self(self, toy_data):
        X, y = toy_data
        cfg = RandomForestConfig(n_estimators=10)
        est = RandomForestEstimator(cfg)
        assert est.fit(X, y) is est


# ── XGBEstimator ───────────────────────────────────────────────────────────────


class TestXGBEstimator:
    def test_default_config(self):
        est = XGBEstimator()
        assert est.cfg == XGBConfig()

    def test_custom_config(self):
        cfg = XGBConfig(n_estimators=50, max_depth=3)
        est = XGBEstimator(cfg)
        assert est.cfg.n_estimators == 50

    def test_predict(self, toy_data):
        cfg = XGBConfig(n_estimators=10)
        _assert_predict_contract(XGBEstimator(cfg), *toy_data)

    def test_predict_returns_strings_not_integers(self, toy_data):
        # XGB internally uses integer labels; verify the LabelEncoder round-trips correctly.
        X, y = toy_data
        cfg = XGBConfig(n_estimators=10)
        preds = XGBEstimator(cfg).fit(X, y).predict(X)
        assert preds.dtype.kind in ("U", "O")  # unicode or object (strings)

    def test_predict_proba(self, toy_data):
        cfg = XGBConfig(n_estimators=10)
        _assert_predict_proba_contract(XGBEstimator(cfg), *toy_data)

    def test_classes(self, toy_data):
        cfg = XGBConfig(n_estimators=10)
        _assert_classes_contract(XGBEstimator(cfg), *toy_data)

    def test_feature_importances(self, toy_data):
        cfg = XGBConfig(n_estimators=10)
        _assert_feature_importances_contract(XGBEstimator(cfg), *toy_data)

    def test_fit_returns_self(self, toy_data):
        X, y = toy_data
        cfg = XGBConfig(n_estimators=10)
        est = XGBEstimator(cfg)
        assert est.fit(X, y) is est


# ── LGBMEstimator ──────────────────────────────────────────────────────────────


class TestLGBMEstimator:
    def test_default_config(self):
        est = LGBMEstimator()
        assert est.cfg == LGBMConfig()

    def test_custom_config(self):
        cfg = LGBMConfig(n_estimators=50, num_leaves=31)
        est = LGBMEstimator(cfg)
        assert est.cfg.num_leaves == 31

    def test_predict(self, toy_data):
        cfg = LGBMConfig(n_estimators=10)
        _assert_predict_contract(LGBMEstimator(cfg), *toy_data)

    def test_predict_proba(self, toy_data):
        cfg = LGBMConfig(n_estimators=10)
        _assert_predict_proba_contract(LGBMEstimator(cfg), *toy_data)

    def test_classes(self, toy_data):
        cfg = LGBMConfig(n_estimators=10)
        _assert_classes_contract(LGBMEstimator(cfg), *toy_data)

    def test_feature_importances(self, toy_data):
        cfg = LGBMConfig(n_estimators=10)
        _assert_feature_importances_contract(LGBMEstimator(cfg), *toy_data)

    def test_fit_returns_self(self, toy_data):
        X, y = toy_data
        cfg = LGBMConfig(n_estimators=10)
        est = LGBMEstimator(cfg)
        assert est.fit(X, y) is est
