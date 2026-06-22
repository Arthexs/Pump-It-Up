"""Tests for pump.tuning."""

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from pump.configs import PipelineConfig, TuningConfig
from pump.pipeline import build_pipeline
from pump.tuning import optimize

CLASSES = ["functional", "functional needs repair", "non functional"]

# 2 trials × 2 folds keeps the suite fast (< 10s total).
FAST = TuningConfig(n_trials=2, cv_folds=2)


@pytest.fixture
def toy_data() -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(0)
    n = 90
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


@pytest.fixture
def xgb_cfg() -> PipelineConfig:
    return PipelineConfig(
        name="test_xgb",
        preprocessing=[],
        features=[],
        model={"type": "xgb"},
    )


@pytest.fixture
def lgbm_cfg() -> PipelineConfig:
    return PipelineConfig(
        name="test_lgbm",
        preprocessing=[],
        features=[],
        model={"type": "lgbm"},
    )


# ── Contract ───────────────────────────────────────────────────────────────────


class TestOptimizeContract:
    def test_returns_pipeline_config(self, xgb_cfg, toy_data):
        X, y = toy_data
        assert isinstance(optimize(xgb_cfg, X, y, FAST), PipelineConfig)

    def test_name_has_tuned_suffix(self, xgb_cfg, toy_data):
        X, y = toy_data
        assert optimize(xgb_cfg, X, y, FAST).name == "test_xgb_tuned"

    def test_preprocessing_preserved(self, toy_data):
        X, y = toy_data
        cfg = PipelineConfig(
            name="test",
            preprocessing=[{"type": "numeric_imputer"}],
            features=[],
            model={"type": "xgb"},
        )
        result = optimize(cfg, X, y, FAST)
        assert result.preprocessing == [{"type": "numeric_imputer"}]

    def test_features_preserved(self, toy_data):
        X, y = toy_data
        cfg = PipelineConfig(
            name="test",
            preprocessing=[],
            features=[{"type": "variance_threshold"}],
            model={"type": "xgb"},
        )
        result = optimize(cfg, X, y, FAST)
        assert result.features == [{"type": "variance_threshold"}]

    def test_model_type_preserved(self, xgb_cfg, toy_data):
        X, y = toy_data
        assert optimize(xgb_cfg, X, y, FAST).model["type"] == "xgb"

    def test_best_cfg_is_buildable(self, xgb_cfg, toy_data):
        X, y = toy_data
        result = optimize(xgb_cfg, X, y, FAST)
        assert isinstance(build_pipeline(result), Pipeline)

    def test_best_cfg_fits_and_predicts(self, xgb_cfg, toy_data):
        X, y = toy_data
        result = optimize(xgb_cfg, X, y, FAST)
        preds = build_pipeline(result).fit(X, y).predict(X)
        assert len(preds) == len(X)
        assert set(preds).issubset(set(CLASSES))


# ── Error handling ─────────────────────────────────────────────────────────────


class TestOptimizeErrors:
    def test_unsupported_model_raises_value_error(self, toy_data):
        X, y = toy_data
        cfg = PipelineConfig(
            name="test",
            preprocessing=[],
            features=[],
            model={"type": "logistic_regression"},
        )
        with pytest.raises(ValueError, match="logistic_regression"):
            optimize(cfg, X, y, FAST)


# ── XGB search space ───────────────────────────────────────────────────────────


class TestOptimizeXGB:
    def test_best_params_within_search_space(self, xgb_cfg, toy_data):
        X, y = toy_data
        m = optimize(xgb_cfg, X, y, FAST).model
        assert 100 <= m["n_estimators"] <= 1000
        assert 3 <= m["max_depth"] <= 10
        assert 0.01 <= m["learning_rate"] <= 0.3
        assert 0.6 <= m["subsample"] <= 1.0
        assert 0.6 <= m["colsample_bytree"] <= 1.0


# ── LGBM search space ──────────────────────────────────────────────────────────


class TestOptimizeLGBM:
    def test_best_params_within_search_space(self, lgbm_cfg, toy_data):
        X, y = toy_data
        m = optimize(lgbm_cfg, X, y, FAST).model
        assert 100 <= m["n_estimators"] <= 1000
        assert 3 <= m["max_depth"] <= 12
        assert 0.01 <= m["learning_rate"] <= 0.3
        assert 20 <= m["num_leaves"] <= 300
        assert 0.6 <= m["subsample"] <= 1.0
        assert 0.6 <= m["colsample_bytree"] <= 1.0
