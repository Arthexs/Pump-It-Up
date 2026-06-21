"""
Registered preprocessing transformers.

Each transformer follows the sklearn protocol: fit / transform / fit_transform.
Registered via @TRANSFORMERS.register so Pipeline.from_config() can build them
from a config dict.

Typical ordering in a PipelineConfig:
    zero_to_nan  →  numeric_imputer  →  categorical_imputer
    →  target_encoder (explicit high-cardinality cols)
    →  ordinal_encoder (auto-detects remaining object cols)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder as _SklearnOrdinalEncoder

from pump.configs import (
    CategoricalImputerConfig,
    NumericImputerConfig,
    OrdinalEncoderConfig,
    TargetEncoderConfig,
    ZeroToNanCleanerConfig,
)
from pump.registry import TRANSFORMERS


@TRANSFORMERS.register("zero_to_nan", config=ZeroToNanCleanerConfig)
class ZeroToNanCleaner(BaseEstimator, TransformerMixin):
    """Replace sentinel zeros with NaN in columns where 0 means missing."""

    def __init__(self, cfg: ZeroToNanCleanerConfig | None = None) -> None:
        self.cfg = cfg or ZeroToNanCleanerConfig()

    def fit(self, X: pd.DataFrame, y=None) -> ZeroToNanCleaner:
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col in self.cfg.columns:
            if col in X.columns:
                X[col] = X[col].replace(0, np.nan)
        return X


@TRANSFORMERS.register("numeric_imputer", config=NumericImputerConfig)
class NumericImputer(BaseEstimator, TransformerMixin):
    """Impute missing values in numeric columns (median/mean/constant)."""

    def __init__(self, cfg: NumericImputerConfig | None = None) -> None:
        self.cfg = cfg or NumericImputerConfig()
        self._cols: list[str] = []
        self._imputer: SimpleImputer | None = None

    def fit(self, X: pd.DataFrame, y=None) -> NumericImputer:
        self._cols = (
            self.cfg.columns
            if self.cfg.columns
            else list(X.select_dtypes(include="number").columns)
        )
        self._imputer = SimpleImputer(
            strategy=self.cfg.strategy,
            fill_value=self.cfg.fill_value,
        )
        self._imputer.fit(X[self._cols])
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X[self._cols] = self._imputer.transform(X[self._cols])
        return X


@TRANSFORMERS.register("categorical_imputer", config=CategoricalImputerConfig)
class CategoricalImputer(BaseEstimator, TransformerMixin):
    """Impute missing values in object/category/string columns (most_frequent/constant)."""

    def __init__(self, cfg: CategoricalImputerConfig | None = None) -> None:
        self.cfg = cfg or CategoricalImputerConfig()
        self._cols: list[str] = []
        self._fills: dict[str, object] = {}

    def fit(self, X: pd.DataFrame, y=None) -> CategoricalImputer:
        self._cols = (
            self.cfg.columns
            if self.cfg.columns
            else list(X.select_dtypes(include=["object", "category", "string"]).columns)
        )
        for col in self._cols:
            if self.cfg.strategy == "most_frequent":
                modes = X[col].mode(dropna=True)
                self._fills[col] = modes.iloc[0] if len(modes) else self.cfg.fill_value
            else:
                self._fills[col] = self.cfg.fill_value
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col in self._cols:
            X[col] = X[col].fillna(self._fills[col])
        return X


@TRANSFORMERS.register("ordinal_encoder", config=OrdinalEncoderConfig)
class OrdinalEncoder(BaseEstimator, TransformerMixin):
    """
    Ordinal-encode low/medium cardinality categorical columns.

    When columns is empty, applies to all object/category columns at fit time.
    Running TargetEncoder first (on explicitly named high-cardinality columns)
    converts those columns to float, so this encoder naturally skips them.
    """

    def __init__(self, cfg: OrdinalEncoderConfig | None = None) -> None:
        self.cfg = cfg or OrdinalEncoderConfig()
        self._cols: list[str] = []
        self._encoder: _SklearnOrdinalEncoder | None = None

    def fit(self, X: pd.DataFrame, y=None) -> OrdinalEncoder:
        self._cols = (
            self.cfg.columns
            if self.cfg.columns
            else list(X.select_dtypes(include=["object", "category", "string"]).columns)
        )
        # unknown_value is only valid when handle_unknown="use_encoded_value"
        kwargs = {"handle_unknown": self.cfg.handle_unknown}
        if self.cfg.handle_unknown == "use_encoded_value":
            kwargs["unknown_value"] = self.cfg.unknown_value
        self._encoder = _SklearnOrdinalEncoder(**kwargs)
        self._encoder.fit(X[self._cols])
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X[self._cols] = self._encoder.transform(X[self._cols])
        return X


@TRANSFORMERS.register("target_encoder", config=TargetEncoderConfig)
class TargetEncoder(BaseEstimator, TransformerMixin):
    """
    Smoothed mean target encoding for high-cardinality columns (funder, installer,
    subvillage, etc.).

    Encoding: (n_cat * mean_cat + smoothing * global_mean) / (n_cat + smoothing)

    For multiclass string targets the label is mapped to its sorted ordinal index
    before computing means. Unknown categories at transform time receive the global
    mean (i.e. a neutral, non-leaking fallback).
    """

    def __init__(self, cfg: TargetEncoderConfig | None = None) -> None:
        self.cfg = cfg or TargetEncoderConfig()
        self._cols: list[str] = []
        self._encodings: dict[str, dict] = {}
        self._global_means: dict[str, float] = {}

    def fit(self, X: pd.DataFrame, y: pd.Series) -> TargetEncoder:
        self._cols = (
            self.cfg.columns
            if self.cfg.columns
            else list(X.select_dtypes(include=["object", "category", "string"]).columns)
        )

        if pd.api.types.is_numeric_dtype(y):
            y_num = y.astype(float)
        else:
            classes = sorted(y.dropna().unique())
            y_num = y.map({cls: float(i) for i, cls in enumerate(classes)})

        global_mean = float(y_num.mean())
        smoothing = self.cfg.smoothing

        for col in self._cols:
            # concat aligns on index so X and y_num don't need to be positionally matched
            combined = pd.concat([X[col].rename("cat"), y_num.rename("target")], axis=1)
            stats = combined.groupby("cat")["target"].agg(count="count", mean="mean")
            stats["encoded"] = (stats["count"] * stats["mean"] + smoothing * global_mean) / (
                stats["count"] + smoothing
            )
            self._encodings[col] = stats["encoded"].to_dict()
            self._global_means[col] = global_mean

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col in self._cols:
            # unknown categories map to NaN via .map(); fill with global mean
            X[col] = X[col].map(self._encodings[col]).fillna(self._global_means[col])
        return X
