"""
Registered ML estimators.

Thin wrappers around sklearn/XGBoost/LightGBM. Each is constructed from
its Pydantic config and exposes a consistent interface: fit / predict /
predict_proba / feature_importances_.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from pump.configs import LGBMConfig, LogisticRegressionConfig, RandomForestConfig, XGBConfig
from pump.registry import ESTIMATORS


@ESTIMATORS.register("logistic_regression", config=LogisticRegressionConfig)
class LogisticRegressionEstimator(BaseEstimator, ClassifierMixin):
    """Baseline linear model. Fast to fit; coefficients are interpretable."""

    def __init__(self, cfg: LogisticRegressionConfig | None = None) -> None:
        self.cfg = cfg or LogisticRegressionConfig()

    def fit(self, X: pd.DataFrame, y: pd.Series) -> LogisticRegressionEstimator:
        self._model = LogisticRegression(
            C=self.cfg.C,
            max_iter=self.cfg.max_iter,
            random_state=self.cfg.random_state,
            solver="lbfgs",
        )
        self._model.fit(X, y)
        self.classes_ = self._model.classes_
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self._model.predict_proba(X)

    @property
    def feature_importances_(self) -> np.ndarray:
        # Mean absolute coefficient across the 3 class boundaries.
        return np.abs(self._model.coef_).mean(axis=0)


@ESTIMATORS.register("random_forest", config=RandomForestConfig)
class RandomForestEstimator(BaseEstimator, ClassifierMixin):
    def __init__(self, cfg: RandomForestConfig | None = None) -> None:
        self.cfg = cfg or RandomForestConfig()

    def fit(self, X: pd.DataFrame, y: pd.Series) -> RandomForestEstimator:
        self._model = RandomForestClassifier(
            n_estimators=self.cfg.n_estimators,
            max_depth=self.cfg.max_depth,
            min_samples_leaf=self.cfg.min_samples_leaf,
            class_weight=self.cfg.class_weight,
            random_state=self.cfg.random_state,
            n_jobs=self.cfg.n_jobs,
        )
        self._model.fit(X, y)
        self.classes_ = self._model.classes_
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self._model.predict_proba(X)

    @property
    def feature_importances_(self) -> np.ndarray:
        return self._model.feature_importances_


@ESTIMATORS.register("xgb", config=XGBConfig)
class XGBEstimator(BaseEstimator, ClassifierMixin):
    """
    XGBoost requires integer labels (0, 1, 2). A LabelEncoder is fitted
    internally so that callers can pass raw string targets and receive
    string class names back from predict / classes_.
    """

    def __init__(self, cfg: XGBConfig | None = None) -> None:
        self.cfg = cfg or XGBConfig()

    def fit(self, X: pd.DataFrame, y: pd.Series) -> XGBEstimator:
        self._le = LabelEncoder()
        y_enc = self._le.fit_transform(y)
        self.classes_ = self._le.classes_

        self._model = XGBClassifier(
            n_estimators=self.cfg.n_estimators,
            max_depth=self.cfg.max_depth,
            learning_rate=self.cfg.learning_rate,
            subsample=self.cfg.subsample,
            colsample_bytree=self.cfg.colsample_bytree,
            eval_metric=self.cfg.eval_metric,
            random_state=self.cfg.random_state,
            n_jobs=self.cfg.n_jobs,
        )
        self._model.fit(X, y_enc)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._le.inverse_transform(self._model.predict(X))

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self._model.predict_proba(X)

    @property
    def feature_importances_(self) -> np.ndarray:
        return self._model.feature_importances_


@ESTIMATORS.register("lgbm", config=LGBMConfig)
class LGBMEstimator(BaseEstimator, ClassifierMixin):
    def __init__(self, cfg: LGBMConfig | None = None) -> None:
        self.cfg = cfg or LGBMConfig()

    def fit(self, X: pd.DataFrame, y: pd.Series) -> LGBMEstimator:
        self._model = LGBMClassifier(
            n_estimators=self.cfg.n_estimators,
            max_depth=self.cfg.max_depth,
            learning_rate=self.cfg.learning_rate,
            num_leaves=self.cfg.num_leaves,
            subsample=self.cfg.subsample,
            subsample_freq=1,  # required for subsample to take effect in LightGBM
            colsample_bytree=self.cfg.colsample_bytree,
            class_weight=self.cfg.class_weight,
            random_state=self.cfg.random_state,
            n_jobs=self.cfg.n_jobs,
            verbosity=self.cfg.verbosity,
        )
        self._model.fit(X, y)
        self.classes_ = self._model.classes_
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self._model.predict_proba(X)

    @property
    def feature_importances_(self) -> np.ndarray:
        return self._model.feature_importances_