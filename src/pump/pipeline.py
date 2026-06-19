"""
End-to-end Pipeline: preprocessing → feature engineering/selection → model.

Built from a PipelineConfig via Pipeline.from_config() — mirrors the
Network.from_config() pattern from the neural-net project.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from pump.config.pipeline import PipelineConfig
from pump.registry import ESTIMATORS, SELECTORS, TRANSFORMERS


class Pipeline:
    """
    Chains preprocessors (list of transformers), feature_steps (engineering +
    selection), and a final estimator.

    All steps follow the sklearn Transformer/Estimator protocol so each can
    also be used standalone for diagnosis and experimentation.
    """

    def __init__(
        self,
        name: str,
        preprocessors: list[Any],
        feature_steps: list[Any],
        estimator: Any,
    ) -> None:
        self.name = name
        self.preprocessors = preprocessors
        self.feature_steps = feature_steps
        self.estimator = estimator
        self._feature_names_out: list[str] = []

    @classmethod
    def from_config(cls, config: PipelineConfig) -> "Pipeline":
        """
        Build a Pipeline from a PipelineConfig.
        Looks up each step type in TRANSFORMERS / SELECTORS / ESTIMATORS.
        """
        raise NotImplementedError

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "Pipeline":
        """Fit all steps in sequence on training data."""
        raise NotImplementedError

    def predict(self, X: pd.DataFrame) -> Any:
        raise NotImplementedError

    def predict_proba(self, X: pd.DataFrame) -> Any:
        raise NotImplementedError

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return transformed features after preprocessing + feature steps.
        Useful for SHAP and post-processing diagnosis."""
        raise NotImplementedError

    @property
    def feature_names_out(self) -> list[str]:
        """Feature names after the full transform chain — needed for SHAP plots."""
        return self._feature_names_out

    def save(self, model_dir: Path | None = None) -> Path:
        """Persist this pipeline via store.save()."""
        from pump import store
        return store.save(self, name=self.name, model_dir=model_dir)

    def summary(self) -> None:
        """Print a human-readable summary of all steps."""
        raise NotImplementedError
