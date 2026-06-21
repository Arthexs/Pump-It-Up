"""
Feature engineering and feature selection.

Engineering: derive new columns (pump_age, geo_cluster, frequency encodings).
Selection: reduce to most informative subset (variance, correlation, XGB importance,
mutual information).

Both groups are registered so Pipeline.from_config() can compose them.

Typical ordering:
    redundant_dropper → pump_age → geo_cluster → frequency_encoder
    → variance_threshold → correlation_threshold → mutual_info / xgb_importance
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.feature_selection import VarianceThreshold as _SklearnVT
from sklearn.feature_selection import mutual_info_classif
from xgboost import XGBClassifier

from pump.configs import (
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

# ── Feature Engineering ───────────────────────────────────────────────────────


@TRANSFORMERS.register("redundant_dropper", config=RedundantColumnDropperConfig)
class RedundantColumnDropper(BaseEstimator, TransformerMixin):
    """Drop the hardcoded redundant/low-signal columns in RedundantColumnDropperConfig."""

    def __init__(self, cfg: RedundantColumnDropperConfig | None = None) -> None:
        self.cfg = cfg or RedundantColumnDropperConfig()

    def fit(self, X: pd.DataFrame, y=None) -> RedundantColumnDropper:
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        cols = [c for c in self.cfg.columns_to_drop if c in X.columns]
        return X.drop(columns=cols)


@TRANSFORMERS.register("pump_age", config=PumpAgeConfig)
class PumpAgeFeature(BaseEstimator, TransformerMixin):
    """
    Derive pump age (years) as date_recorded.year − construction_year.
    Negative values (data errors) are coerced to NaN.
    Run after ZeroToNanCleaner so that construction_year=0 is already NaN.
    """

    def __init__(self, cfg: PumpAgeConfig | None = None) -> None:
        self.cfg = cfg or PumpAgeConfig()

    def fit(self, X: pd.DataFrame, y=None) -> PumpAgeFeature:
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        ref_year = pd.to_datetime(X[self.cfg.reference_column], errors="coerce").dt.year
        construction = pd.to_numeric(X[self.cfg.construction_column], errors="coerce")
        age = ref_year - construction
        X[self.cfg.output_column] = age.where(age >= 0, np.nan)
        return X


@TRANSFORMERS.register("geo_cluster", config=GeoClusterConfig)
class GeoClusterFeature(BaseEstimator, TransformerMixin):
    """
    KMeans cluster on (latitude, longitude). Produces a numeric cluster-ID column.
    Rows with missing coordinates receive NaN in the output column.
    Cluster IDs are unordered integers — tree models handle them correctly as-is.
    """

    def __init__(self, cfg: GeoClusterConfig | None = None) -> None:
        self.cfg = cfg or GeoClusterConfig()
        self._kmeans: KMeans | None = None

    def fit(self, X: pd.DataFrame, y=None) -> GeoClusterFeature:
        coords = X[[self.cfg.lat_column, self.cfg.lon_column]]
        valid = coords.dropna()
        self._kmeans = KMeans(n_clusters=self.cfg.n_clusters, random_state=42, n_init="auto")
        self._kmeans.fit(valid)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        coords = X[[self.cfg.lat_column, self.cfg.lon_column]]
        valid_mask = coords.notna().all(axis=1)
        clusters = np.full(len(X), np.nan)
        if valid_mask.any():
            clusters[valid_mask.to_numpy()] = self._kmeans.predict(coords[valid_mask])
        X[self.cfg.output_column] = clusters
        return X


@TRANSFORMERS.register("frequency_encoder", config=FrequencyEncoderConfig)
class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """
    Map each category to its training-set frequency (fraction of total rows).
    Writes new columns {col}{suffix} — originals are left in place.
    Unknown categories at transform time receive 0.0.
    When cfg.columns is empty, applies to all object/category/string columns.
    """

    def __init__(self, cfg: FrequencyEncoderConfig | None = None) -> None:
        self.cfg = cfg or FrequencyEncoderConfig()
        self._cols: list[str] = []
        self._freqs: dict[str, dict] = {}

    def fit(self, X: pd.DataFrame, y=None) -> FrequencyEncoder:
        self._cols = (
            self.cfg.columns
            if self.cfg.columns
            else list(X.select_dtypes(include=["object", "category", "string"]).columns)
        )
        n = len(X)
        for col in self._cols:
            self._freqs[col] = (X[col].value_counts(dropna=True) / n).to_dict()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col in self._cols:
            X[f"{col}{self.cfg.output_suffix}"] = X[col].map(self._freqs[col]).fillna(0.0)
        return X


# ── Feature Selection ─────────────────────────────────────────────────────────


@SELECTORS.register("variance_threshold", config=VarianceThresholdConfig)
class VarianceThresholdSelector(BaseEstimator, TransformerMixin):
    """
    Drop numeric features whose variance is below cfg.threshold.
    Catches near-constant columns (e.g. recorded_by) that cardinality_report surfaces.
    Assumes numeric columns are fully imputed before this runs.
    """

    def __init__(self, cfg: VarianceThresholdConfig | None = None) -> None:
        self.cfg = cfg or VarianceThresholdConfig()
        self._cols_to_drop: list[str] = []

    def fit(self, X: pd.DataFrame, y=None) -> VarianceThresholdSelector:
        numeric = X.select_dtypes(include="number")
        selector = _SklearnVT(threshold=self.cfg.threshold)
        selector.fit(numeric)
        support = selector.get_support()
        self._cols_to_drop = [
            col for col, keep in zip(numeric.columns, support, strict=True) if not keep
        ]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.drop(columns=[c for c in self._cols_to_drop if c in X.columns])


@SELECTORS.register("correlation_threshold", config=CorrelationThresholdConfig)
class CorrelationThresholdSelector(BaseEstimator, TransformerMixin):
    """
    Greedy pairwise correlation drop: for each pair of numeric columns with
    |correlation| >= cfg.threshold, the second column is marked for removal.
    This confirms or corrects the hardcoded list in RedundantColumnDropperConfig
    using the actual training-data correlations.
    """

    def __init__(self, cfg: CorrelationThresholdConfig | None = None) -> None:
        self.cfg = cfg or CorrelationThresholdConfig()
        self._cols_to_drop: list[str] = []

    def fit(self, X: pd.DataFrame, y=None) -> CorrelationThresholdSelector:
        numeric = X.select_dtypes(include="number")
        corr = numeric.corr(method=self.cfg.method).abs()
        cols = list(corr.columns)
        to_drop: set[str] = set()
        for i, col_a in enumerate(cols):
            if col_a in to_drop:
                continue
            for col_b in cols[i + 1 :]:
                if col_b in to_drop:
                    continue
                if corr.loc[col_a, col_b] >= self.cfg.threshold:
                    to_drop.add(col_b)
        self._cols_to_drop = list(to_drop)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.drop(columns=[c for c in self._cols_to_drop if c in X.columns])


@SELECTORS.register("mutual_info", config=MutualInfoSelectorConfig)
class MutualInfoSelector(BaseEstimator, TransformerMixin):
    """
    Keep the top cfg.k numeric features by mutual information with the target.
    Non-numeric columns pass through unchanged.
    Assumes numeric columns are fully imputed (no NaN).
    """

    def __init__(self, cfg: MutualInfoSelectorConfig | None = None) -> None:
        self.cfg = cfg or MutualInfoSelectorConfig()
        self._cols_to_keep: list[str] = []
        self._non_numeric: list[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> MutualInfoSelector:
        numeric = X.select_dtypes(include="number")
        self._non_numeric = list(X.select_dtypes(exclude="number").columns)
        scores = mutual_info_classif(numeric, y, random_state=self.cfg.random_state)
        k = min(self.cfg.k, len(numeric.columns))
        top_idx = np.argsort(scores)[::-1][:k]
        self._cols_to_keep = [numeric.columns[i] for i in top_idx]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        keep = [c for c in self._cols_to_keep if c in X.columns]
        non_num = [c for c in self._non_numeric if c in X.columns]
        return X[keep + non_num]


@SELECTORS.register("xgb_importance", config=XGBImportanceSelectorConfig)
class XGBImportanceSelector(BaseEstimator, TransformerMixin):
    """
    Fit a quick XGBoost model and keep numeric features above cfg.percentile_cutoff
    by cfg.importance_type (gain/weight/cover). Non-numeric columns pass through.
    XGBoost handles NaN natively, so imputation is not strictly required here.
    """

    def __init__(self, cfg: XGBImportanceSelectorConfig | None = None) -> None:
        self.cfg = cfg or XGBImportanceSelectorConfig()
        self._cols_to_keep: list[str] = []
        self._non_numeric: list[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> XGBImportanceSelector:
        numeric = X.select_dtypes(include="number")
        self._non_numeric = list(X.select_dtypes(exclude="number").columns)

        if not pd.api.types.is_numeric_dtype(y):
            classes = sorted(y.dropna().unique())
            y_enc = y.map({cls: i for i, cls in enumerate(classes)})
        else:
            y_enc = y

        model = XGBClassifier(
            n_estimators=self.cfg.n_estimators,
            max_depth=self.cfg.max_depth,
            eval_metric="mlogloss",
            random_state=42,
            n_jobs=-1,
        )
        model.fit(numeric, y_enc)

        scores = pd.Series(
            model.feature_importances_,
            index=numeric.columns,
        )
        cutoff = np.percentile(scores.to_numpy(), self.cfg.percentile_cutoff)
        self._cols_to_keep = list(scores[scores >= cutoff].index)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        keep = [c for c in self._cols_to_keep if c in X.columns]
        non_num = [c for c in self._non_numeric if c in X.columns]
        return X[keep + non_num]
