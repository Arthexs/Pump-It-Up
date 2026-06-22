"""Tests for pump.pipeline."""

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from pump.configs import PipelineConfig
from pump.pipeline import build_pipeline

CLASSES = ["functional", "functional needs repair", "non functional"]


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
def minimal_config() -> PipelineConfig:
    return PipelineConfig(
        name="test_rf",
        preprocessing=[],
        features=[],
        model={"type": "random_forest", "n_estimators": 10},
    )


# ── Structure ──────────────────────────────────────────────────────────────────


class TestBuildPipelineStructure:
    def test_returns_sklearn_pipeline(self, minimal_config):
        assert isinstance(build_pipeline(minimal_config), Pipeline)

    def test_model_is_last_step(self, minimal_config):
        pipe = build_pipeline(minimal_config)
        assert pipe.steps[-1][0] == "model"

    def test_no_preprocessing_or_features_gives_one_step(self, minimal_config):
        pipe = build_pipeline(minimal_config)
        assert len(pipe.steps) == 1

    def test_preprocessing_step_is_inserted(self):
        cfg = PipelineConfig(
            name="test",
            preprocessing=[{"type": "numeric_imputer"}],
            features=[],
            model={"type": "random_forest", "n_estimators": 10},
        )
        pipe = build_pipeline(cfg)
        assert len(pipe.steps) == 2
        assert pipe.steps[0][0] == "numeric_imputer"

    def test_feature_step_from_selectors_registry(self):
        cfg = PipelineConfig(
            name="test",
            preprocessing=[],
            features=[{"type": "variance_threshold"}],
            model={"type": "random_forest", "n_estimators": 10},
        )
        pipe = build_pipeline(cfg)
        assert len(pipe.steps) == 2
        assert pipe.steps[0][0] == "variance_threshold"

    def test_preprocessing_before_features_before_model(self):
        cfg = PipelineConfig(
            name="test",
            preprocessing=[{"type": "numeric_imputer"}],
            features=[{"type": "variance_threshold"}],
            model={"type": "random_forest", "n_estimators": 10},
        )
        pipe = build_pipeline(cfg)
        names = [name for name, _ in pipe.steps]
        assert names == ["numeric_imputer", "variance_threshold", "model"]

    def test_duplicate_step_types_get_unique_names(self):
        cfg = PipelineConfig(
            name="test",
            preprocessing=[{"type": "numeric_imputer"}, {"type": "numeric_imputer"}],
            features=[],
            model={"type": "random_forest", "n_estimators": 10},
        )
        pipe = build_pipeline(cfg)
        names = [name for name, _ in pipe.steps[:-1]]
        assert len(names) == len(set(names))


# ── Error handling ─────────────────────────────────────────────────────────────


class TestBuildPipelineErrors:
    def test_unknown_step_type_raises_key_error(self):
        cfg = PipelineConfig(
            name="test",
            preprocessing=[{"type": "does_not_exist"}],
            features=[],
            model={"type": "random_forest", "n_estimators": 10},
        )
        with pytest.raises(KeyError, match="does_not_exist"):
            build_pipeline(cfg)

    def test_unknown_model_type_raises_key_error(self):
        cfg = PipelineConfig(
            name="test",
            preprocessing=[],
            features=[],
            model={"type": "does_not_exist"},
        )
        with pytest.raises(KeyError, match="does_not_exist"):
            build_pipeline(cfg)

    def test_invalid_step_param_raises_validation_error(self):
        cfg = PipelineConfig(
            name="test",
            preprocessing=[{"type": "numeric_imputer", "bad_param": 99}],
            features=[],
            model={"type": "random_forest", "n_estimators": 10},
        )
        with pytest.raises(Exception):  # pydantic ValidationError
            build_pipeline(cfg)

    def test_invalid_model_param_raises_validation_error(self):
        cfg = PipelineConfig(
            name="test",
            preprocessing=[],
            features=[],
            model={"type": "random_forest", "bad_param": 99},
        )
        with pytest.raises(Exception):  # pydantic ValidationError
            build_pipeline(cfg)


# ── Config forwarding ──────────────────────────────────────────────────────────


class TestConfigForwarding:
    def test_model_hyperparams_reach_estimator_cfg(self):
        cfg = PipelineConfig(
            name="test",
            preprocessing=[],
            features=[],
            model={"type": "random_forest", "n_estimators": 42},
        )
        pipe = build_pipeline(cfg)
        assert pipe.named_steps["model"].cfg.n_estimators == 42

    def test_step_hyperparams_reach_transformer_cfg(self):
        cfg = PipelineConfig(
            name="test",
            preprocessing=[{"type": "numeric_imputer", "strategy": "mean"}],
            features=[],
            model={"type": "random_forest", "n_estimators": 10},
        )
        pipe = build_pipeline(cfg)
        assert pipe.named_steps["numeric_imputer"].cfg.strategy == "mean"


# ── Fit / predict ──────────────────────────────────────────────────────────────


class TestPipelineFitPredict:
    def test_fit_predict_minimal(self, minimal_config, toy_data):
        X, y = toy_data
        pipe = build_pipeline(minimal_config)
        preds = pipe.fit(X, y).predict(X)
        assert len(preds) == len(X)
        assert set(preds).issubset(set(CLASSES))

    def test_predict_proba_shape(self, minimal_config, toy_data):
        X, y = toy_data
        pipe = build_pipeline(minimal_config)
        pipe.fit(X, y)
        proba = pipe.predict_proba(X)
        assert proba.shape == (len(X), 3)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_preprocessing_and_selection_in_pipeline(self, toy_data):
        X, y = toy_data
        cfg = PipelineConfig(
            name="test",
            preprocessing=[{"type": "numeric_imputer"}],
            features=[{"type": "variance_threshold"}],
            model={"type": "random_forest", "n_estimators": 10},
        )
        pipe = build_pipeline(cfg)
        preds = pipe.fit(X, y).predict(X)
        assert len(preds) == len(X)

    def test_xgb_pipeline_predict_returns_strings(self, toy_data):
        X, y = toy_data
        cfg = PipelineConfig(
            name="test_xgb",
            preprocessing=[],
            features=[],
            model={"type": "xgb", "n_estimators": 10},
        )
        pipe = build_pipeline(cfg)
        preds = pipe.fit(X, y).predict(X)
        assert set(preds).issubset(set(CLASSES))

    def test_lgbm_pipeline(self, toy_data):
        X, y = toy_data
        cfg = PipelineConfig(
            name="test_lgbm",
            preprocessing=[],
            features=[],
            model={"type": "lgbm", "n_estimators": 10},
        )
        pipe = build_pipeline(cfg)
        preds = pipe.fit(X, y).predict(X)
        assert set(preds).issubset(set(CLASSES))

    def test_logistic_regression_pipeline(self, toy_data):
        X, y = toy_data
        cfg = PipelineConfig(
            name="test_lr",
            preprocessing=[],
            features=[],
            model={"type": "logistic_regression"},
        )
        pipe = build_pipeline(cfg)
        preds = pipe.fit(X, y).predict(X)
        assert set(preds).issubset(set(CLASSES))
