"""Pydantic configs for feature engineering and selection."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class BaseConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ── Engineering ──────────────────────────────────────────────────────────────


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


# ── Selection ────────────────────────────────────────────────────────────────


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
