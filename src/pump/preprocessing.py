"""
Registered preprocessing transformers.

Each transformer follows the sklearn protocol: fit / transform / fit_transform.
Registered via @TRANSFORMERS.register so Pipeline.from_config() can build them
from a config dict.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from pump.config.preprocessing import (
    CategoricalImputerConfig,
    ConstructionYearCleanerConfig,
    NumericImputerConfig,
    OrdinalEncoderConfig,
    TargetEncoderConfig,
)
from pump.registry import TRANSFORMERS


@TRANSFORMERS.register("construction_year_cleaner", config=ConstructionYearCleanerConfig)
class ConstructionYearCleaner(BaseEstimator, TransformerMixin):
    """
    Replace zero values in construction_year with NaN (or median).
    Must run before numeric imputation so zeros don't bias the median.
    """

    def __init__(self, **kwargs: Any) -> None:
        self._cfg = ConstructionYearCleanerConfig(**kwargs)

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "ConstructionYearCleaner":
        raise NotImplementedError

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError


@TRANSFORMERS.register("numeric_imputer", config=NumericImputerConfig)
class NumericImputer(BaseEstimator, TransformerMixin):
    """Impute numeric columns with mean, median, or a constant."""

    def __init__(self, **kwargs: Any) -> None:
        self._cfg = NumericImputerConfig(**kwargs)

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "NumericImputer":
        raise NotImplementedError

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError


@TRANSFORMERS.register("categorical_imputer", config=CategoricalImputerConfig)
class CategoricalImputer(BaseEstimator, TransformerMixin):
    """Impute categorical columns with most frequent value or a constant."""

    def __init__(self, **kwargs: Any) -> None:
        self._cfg = CategoricalImputerConfig(**kwargs)

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "CategoricalImputer":
        raise NotImplementedError

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError


@TRANSFORMERS.register("ordinal_encoder", config=OrdinalEncoderConfig)
class OrdinalEncoderTransformer(BaseEstimator, TransformerMixin):
    """
    Encode categorical columns as integers.
    Unknown categories at inference time map to -1 (not an error).
    """

    def __init__(self, **kwargs: Any) -> None:
        self._cfg = OrdinalEncoderConfig(**kwargs)

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "OrdinalEncoderTransformer":
        raise NotImplementedError

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError


@TRANSFORMERS.register("target_encoder", config=TargetEncoderConfig)
class TargetEncoder(BaseEstimator, TransformerMixin):
    """
    Replace each category with smoothed mean of target.
    Fit on training data only — avoids target leakage.
    Useful for high-cardinality columns: funder, installer.
    """

    def __init__(self, **kwargs: Any) -> None:
        self._cfg = TargetEncoderConfig(**kwargs)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "TargetEncoder":
        raise NotImplementedError

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError
