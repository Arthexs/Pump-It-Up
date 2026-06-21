"""Pydantic configs for registered ML estimators."""

from pydantic import BaseModel, ConfigDict, Field, field_validator


class BaseConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")


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
    use_label_encoder: bool = Field(default=False)
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
