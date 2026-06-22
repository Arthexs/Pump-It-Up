"""Pydantic config models for all pipeline components."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class BaseConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ── Preprocessing ─────────────────────────────────────────────────────────────


class NumericImputerConfig(BaseConfig):
    strategy: Literal["mean", "median", "constant"] = Field(
        default="median",
        description="Imputation strategy for numeric columns",
    )
    fill_value: float = Field(
        default=0.0,
        description="Value to use when strategy='constant'",
    )
    columns: list[str] = Field(
        default_factory=list,
        description="Columns to impute; empty = all numeric",
    )


class CategoricalImputerConfig(BaseConfig):
    strategy: Literal["most_frequent", "constant"] = Field(
        default="most_frequent",
        description="Imputation strategy for categorical columns",
    )
    fill_value: str = Field(
        default="unknown",
        description="Value to use when strategy='constant'",
    )
    columns: list[str] = Field(
        default_factory=list,
        description="Columns to impute; empty = all object/category columns",
    )


class OrdinalEncoderConfig(BaseConfig):
    columns: list[str] = Field(
        default_factory=list,
        description="Columns to ordinal-encode; empty = all object columns",
    )
    handle_unknown: Literal["use_encoded_value", "error"] = Field(
        default="use_encoded_value",
    )
    unknown_value: int = Field(default=-1)


class TargetEncoderConfig(BaseConfig):
    """
    Encodes high-cardinality columns (funder, installer, subvillage)
    by replacing each category with the mean target value for that category.
    Only fitted on training data to avoid leakage.
    """

    columns: list[str] = Field(
        default_factory=list,
        description="High-cardinality columns to target-encode",
    )
    smoothing: float = Field(
        default=1.0,
        gt=0,
        description="Smoothing factor to shrink toward global mean",
    )


class ZeroToNanCleanerConfig(BaseConfig):
    """
    Several columns use 0 as a sentinel for missing data.
    Replace those zeros with NaN before numeric imputation runs.
    The default list matches the columns flagged by missing_summary's pct_zero column.
    """

    columns: list[str] = Field(
        default_factory=lambda: ["construction_year", "gps_height", "population"],
        description="Columns where 0 means missing — replaced with NaN",
    )


# ── Feature Engineering ───────────────────────────────────────────────────────


class PumpAgeConfig(BaseConfig):
    """Derive pump age from construction_year and date_recorded."""

    reference_column: str = Field(default="date_recorded")
    construction_column: str = Field(default="construction_year")
    output_column: str = Field(default="pump_age")


class GeoClusterConfig(BaseConfig):
    """
    Cluster pumps by lat/lon using KMeans — produces a categorical
    'geo_cluster' feature that captures regional failure patterns
    without leaking the exact coordinates directly.
    """

    n_clusters: int = Field(default=20, gt=1)
    output_column: str = Field(default="geo_cluster")
    lat_column: str = Field(default="latitude")
    lon_column: str = Field(default="longitude")


class FrequencyEncoderConfig(BaseConfig):
    """
    Replace high-cardinality categories with their frequency in training data.
    Lighter alternative to target encoding for columns like subvillage.
    """

    columns: list[str] = Field(default_factory=list)
    output_suffix: str = Field(default="_freq")


class RedundantColumnDropperConfig(BaseConfig):
    """
    The dataset has column groups at different granularities:
    extraction_type / extraction_type_group / extraction_type_class
    water_quality / quality_group
    etc.
    Keep the most granular; drop the redundant aggregations.
    """

    columns_to_drop: list[str] = Field(
        default_factory=lambda: [
            "extraction_type_group",
            "extraction_type_class",
            "payment_type",  # duplicate of payment
            "quantity_group",  # duplicate of quantity
            "quality_group",  # duplicate of water_quality
            "source_type",  # duplicate of source
            "source_class",
            "waterpoint_type_group",
            "recorded_by",  # only one value in training data
            "wpt_name",  # too many unique values, low signal
            "num_private",  # undocumented column
            "region_code",  # redundant with region
            "district_code",  # redundant with lga
            "subvillage",  # extreme cardinality, handled separately
            "scheme_name",  # extreme cardinality
        ],
        description="Columns to drop before selection",
    )


# ── Feature Selection ─────────────────────────────────────────────────────────


class VarianceThresholdConfig(BaseConfig):
    threshold: float = Field(
        default=0.01,
        ge=0,
        description="Drop features whose variance is below this value",
    )


class CorrelationThresholdConfig(BaseConfig):
    threshold: float = Field(
        default=0.95,
        ge=0,
        le=1,
        description="Drop one of each pair of features correlated above this value",
    )
    method: Literal["pearson", "spearman"] = Field(default="spearman")


class XGBImportanceSelectorConfig(BaseConfig):
    """
    Fit a quick XGBoost model and keep only features above the importance
    percentile cutoff. Used as a wrapper method for feature selection.
    """

    importance_type: Literal["gain", "weight", "cover"] = Field(default="gain")
    percentile_cutoff: float = Field(
        default=10.0,
        ge=0,
        le=100,
        description="Drop features below this importance percentile",
    )
    n_estimators: int = Field(default=100, gt=0)
    max_depth: int = Field(default=4, gt=0)


class MutualInfoSelectorConfig(BaseConfig):
    """
    Mutual information between each feature and the target.
    Captures non-linear and categorical relationships that correlation misses.
    """

    k: int = Field(
        default=20,
        gt=0,
        description="Number of top features to keep",
    )
    random_state: int = Field(default=42)


# ── Models ────────────────────────────────────────────────────────────────────


class LogisticRegressionConfig(BaseConfig):
    """Baseline model — fast to fit, interpretable coefficients."""

    C: float = Field(default=1.0, gt=0, description="Inverse regularization strength")
    max_iter: int = Field(default=1000, gt=0)
    random_state: int = Field(default=42)


class RandomForestConfig(BaseConfig):
    n_estimators: int = Field(default=300, gt=0)
    max_depth: int | None = Field(default=None, description="None = grow fully")
    min_samples_leaf: int = Field(default=2, gt=0)
    class_weight: str | None = Field(
        default="balanced",
        description="'balanced' to compensate for the minority 'needs repair' class",
    )
    random_state: int = Field(default=42)
    n_jobs: int = Field(default=-1)


class XGBConfig(BaseConfig):
    """
    Primary model. XGBoost handles mixed feature types well and typically
    tops the leaderboard on this dataset.
    """

    n_estimators: int = Field(default=500, gt=0)
    max_depth: int = Field(default=6, gt=0)
    learning_rate: float = Field(default=0.05, gt=0)
    subsample: float = Field(default=0.8, gt=0, le=1)
    colsample_bytree: float = Field(default=0.8, gt=0, le=1)
    eval_metric: str = Field(default="mlogloss")
    random_state: int = Field(default=42)
    n_jobs: int = Field(default=-1)

    @field_validator("learning_rate")
    @classmethod
    def lr_sensible(cls, v: float) -> float:
        if v > 0.5:
            raise ValueError("Learning rate > 0.5 is almost always a mistake for XGB")
        return v


class LGBMConfig(BaseConfig):
    n_estimators: int = Field(default=500, gt=0)
    max_depth: int = Field(default=-1, description="-1 = no limit")
    learning_rate: float = Field(default=0.05, gt=0)
    num_leaves: int = Field(default=63, gt=1)
    subsample: float = Field(default=0.8, gt=0, le=1)
    colsample_bytree: float = Field(default=0.8, gt=0, le=1)
    class_weight: str | None = Field(default="balanced")
    random_state: int = Field(default=42)
    n_jobs: int = Field(default=-1)
    verbosity: int = Field(default=-1)


# ── Evaluation ────────────────────────────────────────────────────────────────


class EvaluationConfig(BaseConfig):
    """Top-level evaluation settings."""

    labels: list[str] = Field(
        default=["functional", "functional needs repair", "non functional"],
        description="Class labels in the order the model outputs them",
    )
    positive_class: str = Field(
        default="non functional",
        description="Class treated as 'positive' for binary metrics",
    )


class CostMatrixConfig(BaseConfig):
    """
    Asymmetric misclassification costs for the business case.

    Rows = true label, Cols = predicted label.
    Order: functional, needs_repair, non_functional.

    Rationale:
    - Predicting 'functional' when it's actually 'non functional' means
      a community has no water — highest cost.
    - Predicting 'non functional' when it's 'functional' means a wasted
      inspection trip — moderate cost.
    - 'needs repair' misclassifications sit in between.
    """

    matrix: list[list[float]] = Field(
        default=[
            # pred: func  needs_repair  non_func
            [0.0, 1.0, 3.0],  # true: functional
            [2.0, 0.0, 2.0],  # true: needs_repair
            [5.0, 3.0, 0.0],  # true: non_functional
        ],
        description="Cost[true][predicted] — tune these for the presentation",
    )


class SHAPConfig(BaseConfig):
    n_background_samples: int = Field(
        default=100,
        gt=0,
        description="Number of background samples for TreeExplainer",
    )
    max_display: int = Field(
        default=20,
        gt=0,
        description="Number of features to show in summary plot",
    )


# ── Pipeline ──────────────────────────────────────────────────────────────────


class PipelineConfig(BaseConfig):
    """
    Declarative description of a full training pipeline.
    Passed to Pipeline.from_config() to build the end-to-end object.

    Example:
        cfg = PipelineConfig(
            name="xgb_baseline",
            preprocessing=[
                {"type": "construction_year_cleaner"},
                {"type": "numeric_imputer", "strategy": "median"},
                {"type": "categorical_imputer"},
                {"type": "ordinal_encoder"},
            ],
            features=[
                {"type": "redundant_column_dropper"},
                {"type": "pump_age"},
                {"type": "geo_cluster", "n_clusters": 20},
                {"type": "xgb_importance_selector", "percentile_cutoff": 10},
            ],
            model={"type": "xgb", "n_estimators": 500, "learning_rate": 0.05},
        )
    """

    name: str = Field(..., description="Human-readable name; used as filename stem in store")
    preprocessing: list[dict] = Field(default_factory=list)
    features: list[dict] = Field(default_factory=list)
    model: dict = Field(..., description="Must contain 'type' key matching a registered estimator")
