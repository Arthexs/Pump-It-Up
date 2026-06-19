"""Pydantic configs for preprocessing transformers."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class BaseConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")


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


class ConstructionYearCleanerConfig(BaseConfig):
    """
    Zero values in construction_year represent missing data.
    Replace them before any numeric imputation runs.
    """

    column: str = Field(default="construction_year")
    zero_replacement: Literal["nan", "median"] = Field(
        default="nan",
        description="Replace zeros with NaN (then imputed) or column median",
    )
