"""
Registered ML estimators.

Thin wrappers around sklearn/XGBoost/LightGBM. Each is constructed from
its Pydantic config and exposes a consistent interface: fit / predict /
predict_proba / feature_importances_.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin

from pump.config.models import (
    LGBMConfig,
    LogisticRegressionConfig,
    RandomForestConfig,
    XGBConfig,
)
from pump.registry import ESTIMATORS


class _BaseEstimator(BaseEstimator, ClassifierMixin):
    """Shared interface for all registered estimators."""

    @property
    def feature_importances_(self) -> np.ndarray:
        raise NotImplementedError

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "_BaseEstimator":
        raise NotImplementedError

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError


@ESTIMATORS.register("logistic_regression", config=LogisticRegressionConfig)
class LogisticRegressionEstimator(_BaseEstimator):
    """Baseline model — fast to fit, sets a performance floor."""

    def __init__(self, **kwargs: Any) -> None:
        self._cfg = LogisticRegressionConfig(**kwargs)


@ESTIMATORS.register("random_forest", config=RandomForestConfig)
class RandomForestEstimator(_BaseEstimator):
    def __init__(self, **kwargs: Any) -> None:
        self._cfg = RandomForestConfig(**kwargs)


@ESTIMATORS.register("xgb", config=XGBConfig)
class XGBEstimator(_BaseEstimator):
    """Primary model. Consistently strong on this dataset."""

    def __init__(self, **kwargs: Any) -> None:
        self._cfg = XGBConfig(**kwargs)


@ESTIMATORS.register("lgbm", config=LGBMConfig)
class LGBMEstimator(_BaseEstimator):
    def __init__(self, **kwargs: Any) -> None:
        self._cfg = LGBMConfig(**kwargs)
