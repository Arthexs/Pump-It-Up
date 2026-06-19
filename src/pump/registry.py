"""
Registry for pipeline components: transformers, selectors, estimators.

Identical pattern to the neural-net project — Registry for classes with
Pydantic configs, FnRegistry for plain callables (e.g. metric functions).
"""

from typing import Any, Callable, Type

from pydantic import BaseModel


class Registry:
    """Registry that pairs a component class with its Pydantic config schema."""

    def __init__(self, label: str) -> None:
        self._registry: dict[str, Type] = {}
        self._configs: dict[str, type[BaseModel]] = {}
        self.label = label

    def register(self, name: str, config: type[BaseModel]) -> Callable[[Type], Type]:
        """Decorator: @TRANSFORMERS.register("imputer", config=ImputerConfig)"""

        def decorator(cls: Type) -> Type:
            if name in self._registry:
                raise ValueError(f"{self.label} '{name}' already registered.")
            self._registry[name] = cls
            self._configs[name] = config
            return cls

        return decorator

    def get(self, name: str) -> Type:
        if name not in self._registry:
            raise KeyError(
                f"Unknown {self.label}: '{name}'. "
                f"Available: {list(self._registry.keys())}"
            )
        return self._registry[name]

    def schemas(self) -> dict[str, dict[str, Any]]:
        """JSON schemas for all registered configs — useful for introspection."""
        return {
            name: cfg.model_json_schema()
            for name, cfg in self._configs.items()
        }

    def keys(self) -> list[str]:
        return list(self._registry.keys())


class FnRegistry:
    """Registry for named callables (metric functions, scoring fns, etc.)."""

    def __init__(self, label: str) -> None:
        self._fns: dict[str, Callable[..., Any]] = {}
        self.label = label

    def register(self, name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
            if name in self._fns:
                raise ValueError(f"{self.label} '{name}' already registered.")
            self._fns[name] = fn
            return fn
        return decorator

    def get(self, name: str) -> Callable[..., Any]:
        if name not in self._fns:
            raise KeyError(
                f"Unknown {self.label}: '{name}'. "
                f"Available: {list(self._fns.keys())}"
            )
        return self._fns[name]

    def keys(self) -> list[str]:
        return list(self._fns.keys())


# ── Module-level singletons ──────────────────────────────────────────────────
TRANSFORMERS = Registry("Transformer")   # preprocessing steps
SELECTORS    = Registry("Selector")      # feature selection methods
ESTIMATORS   = Registry("Estimator")     # ML models
METRICS      = FnRegistry("Metric")      # evaluation functions
