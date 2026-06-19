"""Pydantic config for the end-to-end Pipeline."""

from pydantic import BaseModel, ConfigDict, Field


class BaseConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")


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
