"""Pydantic configs for evaluation settings."""

from pydantic import BaseModel, ConfigDict, Field


class BaseConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")


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
            [0.0,  1.0,         3.0],   # true: functional
            [2.0,  0.0,         2.0],   # true: needs_repair
            [5.0,  3.0,         0.0],   # true: non_functional
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
