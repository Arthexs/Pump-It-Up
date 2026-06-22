"""
End-to-end Pipeline: preprocessing → feature engineering/selection → model.

Built from a PipelineConfig via build_pipeline() — each dict in preprocessing/features
is resolved against the TRANSFORMERS + SELECTORS registries; the model dict is resolved
against ESTIMATORS.
"""

from __future__ import annotations

import pump.features  # noqa: F401 — registers TRANSFORMERS/SELECTORS entries
import pump.models  # noqa: F401 — registers ESTIMATORS entries
import pump.preprocessing  # noqa: F401 — registers TRANSFORMERS entries

from sklearn.pipeline import Pipeline

from pump.configs import PipelineConfig
from pump.registry import ESTIMATORS, SELECTORS, TRANSFORMERS


def _resolve_transformer(spec: dict) -> tuple[str, object]:
    spec = dict(spec)
    step_type = spec.pop("type")
    for registry in (TRANSFORMERS, SELECTORS):
        if step_type in registry.keys():
            cls = registry.get(step_type)
            cfg = registry._configs[step_type](**spec)
            return step_type, cls(cfg)
    raise KeyError(
        f"Unknown step type: '{step_type}'. "
        f"Available: transformers={TRANSFORMERS.keys()}, selectors={SELECTORS.keys()}"
    )


def _resolve_estimator(spec: dict) -> object:
    spec = dict(spec)
    model_type = spec.pop("type")
    cls = ESTIMATORS.get(model_type)
    cfg = ESTIMATORS._configs[model_type](**spec)
    return cls(cfg)


def build_pipeline(config: PipelineConfig) -> Pipeline:
    """Construct an sklearn Pipeline from a PipelineConfig."""
    steps: list[tuple[str, object]] = []
    name_counts: dict[str, int] = {}

    def unique_name(base: str) -> str:
        n = name_counts.get(base, 0)
        name_counts[base] = n + 1
        return base if n == 0 else f"{base}_{n}"

    for spec in config.preprocessing:
        name, transformer = _resolve_transformer(spec)
        steps.append((unique_name(name), transformer))

    for spec in config.features:
        name, transformer = _resolve_transformer(spec)
        steps.append((unique_name(name), transformer))

    steps.append(("model", _resolve_estimator(config.model)))

    return Pipeline(steps)
