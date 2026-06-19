"""
Feature engineering and feature selection.

Engineering: derive new columns (pump_age, geo_cluster, frequency encodings).
Selection: reduce to most informative subset (variance, correlation, XGB importance,
mutual information).

Both groups are registered so Pipeline.from_config() can compose them.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from pump.config.features import (
    CorrelationThresholdConfig,
    FrequencyEncoderConfig,
    GeoClusterConfig,
    MutualInfoSelectorConfig,
    PumpAgeConfig,
    RedundantColumnDropperConfig,
    VarianceThresholdConfig,
    XGBImportanceSelectorConfig,
)
from pump.registry import SELECTORS, TRANSFORMERS


# ── Engineering ──────────────────────────────────────────────────────────────

@TRANSFORMERS.register("redundant_column_dropper", config=RedundantColumnDropperConfig)
class RedundantColumnDropper(BaseEstimator, TransformerMixin):
    """Drop columns that are duplicated at a coarser granularity or carry no signal."""

    def __init__(self, **kwargs: Any) -> None:
        self._cfg = RedundantColumnDropperConfig(**kwargs)

    def fit(self, X: pd.DataFrame, y: Any = None) -> "RedundantColumnDropper":
        raise NotImplementedError

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError


@TRANSFORMERS.register("pump_age", config=PumpAgeConfig)
class PumpAgeFeature(BaseEstimator, TransformerMixin):
    """
    Derive pump_age = year(date_recorded) - construction_year.
    Requires construction_year zeros already replaced before this runs.
    """

    def __init__(self, **kwargs: Any) -> None:
        self._cfg = PumpAgeConfig(**kwargs)

    def fit(self, X: pd.DataFrame, y: Any = None) -> "PumpAgeFeature":
        raise NotImplementedError

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError


@TRANSFORMERS.register("geo_cluster", config=GeoClusterConfig)
class GeoCluster(BaseEstimator, TransformerMixin):
    """
    KMeans on lat/lon → integer geo_cluster feature.
    Captures regional failure patterns without raw coordinates leaking into splits.
    """

    def __init__(self, **kwargs: Any) -> None:
        self._cfg = GeoClusterConfig(**kwargs)

    def fit(self, X: pd.DataFrame, y: Any = None) -> "GeoCluster":
        raise NotImplementedError

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError


@TRANSFORMERS.register("frequency_encoder", config=FrequencyEncoderConfig)
class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """Replace each category with its frequency in the training set."""

    def __init__(self, **kwargs: Any) -> None:
        self._cfg = FrequencyEncoderConfig(**kwargs)

    def fit(self, X: pd.DataFrame, y: Any = None) -> "FrequencyEncoder":
        raise NotImplementedError

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError


# ── Selection ────────────────────────────────────────────────────────────────

@SELECTORS.register("variance_threshold", config=VarianceThresholdConfig)
class VarianceThresholdSelector(BaseEstimator, TransformerMixin):
    """Drop numeric features whose variance falls below the threshold."""

    def __init__(self, **kwargs: Any) -> None:
        self._cfg = VarianceThresholdConfig(**kwargs)

    def fit(self, X: pd.DataFrame, y: Any = None) -> "VarianceThresholdSelector":
        raise NotImplementedError

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError


@SELECTORS.register("correlation_threshold", config=CorrelationThresholdConfig)
class CorrelationThresholdSelector(BaseEstimator, TransformerMixin):
    """Drop one of each pair of features correlated above the threshold."""

    def __init__(self, **kwargs: Any) -> None:
        self._cfg = CorrelationThresholdConfig(**kwargs)

    def fit(self, X: pd.DataFrame, y: Any = None) -> "CorrelationThresholdSelector":
        raise NotImplementedError

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError


@SELECTORS.register("xgb_importance_selector", config=XGBImportanceSelectorConfig)
class XGBImportanceSelector(BaseEstimator, TransformerMixin):
    """
    Fit a lightweight XGBoost model, keep features above the importance percentile.
    Stores importances_ so you can inspect and plot what was kept/dropped.
    """

    def __init__(self, **kwargs: Any) -> None:
        self._cfg = XGBImportanceSelectorConfig(**kwargs)
        self.importances_: pd.Series | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "XGBImportanceSelector":
        raise NotImplementedError

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError


@SELECTORS.register("mutual_info_selector", config=MutualInfoSelectorConfig)
class MutualInfoSelector(BaseEstimator, TransformerMixin):
    """
    Keep top-k features by mutual information with the target.
    Captures non-linear and categorical relationships Pearson/Spearman miss.
    Stores scores_ for inspection.
    """

    def __init__(self, **kwargs: Any) -> None:
        self._cfg = MutualInfoSelectorConfig(**kwargs)
        self.scores_: pd.Series | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "MutualInfoSelector":
        raise NotImplementedError

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError
