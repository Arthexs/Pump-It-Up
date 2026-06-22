"""
Optuna hyperparameter optimisation for pump classifiers.

Usage:
    from pump.tuning import optimize
    best_cfg = optimize(base_cfg, X_train, y_train)
    # → PipelineConfig with best hyperparams; name is base_cfg.name + "_tuned"

Supported model types: xgb, lgbm, random_forest.
"""

from __future__ import annotations

import logging

import optuna
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score

from pump.configs import PipelineConfig, TuningConfig
from pump.pipeline import build_pipeline

logger = logging.getLogger(__name__)


def _suggest_xgb(trial: optuna.Trial) -> dict[str, object]:
    return {
        "type": "xgb",
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
    }


def _suggest_lgbm(trial: optuna.Trial) -> dict[str, object]:
    return {
        "type": "lgbm",
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 20, 300, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
    }


def _suggest_random_forest(trial: optuna.Trial) -> dict[str, object]:
    return {
        "type": "random_forest",
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
    }


_SUGGEST = {
    "xgb": _suggest_xgb,
    "lgbm": _suggest_lgbm,
    "random_forest": _suggest_random_forest,
}


def optimize(
    base_cfg: PipelineConfig,
    X: pd.DataFrame,
    y: pd.Series,
    tuning_cfg: TuningConfig | None = None,
) -> PipelineConfig:
    """Run Optuna HPO on the model type in base_cfg; return the best PipelineConfig."""
    cfg = tuning_cfg if tuning_cfg is not None else TuningConfig()
    model_type = base_cfg.model["type"]

    if model_type not in _SUGGEST:
        raise ValueError(
            f"No search space defined for model type '{model_type}'. Supported: {sorted(_SUGGEST)}"
        )

    suggest_fn = _SUGGEST[model_type]
    cv = StratifiedKFold(n_splits=cfg.cv_folds, shuffle=True, random_state=cfg.random_state)

    def objective(trial: optuna.Trial) -> float:
        trial_cfg = PipelineConfig(
            name=base_cfg.name,
            preprocessing=base_cfg.preprocessing,
            features=base_cfg.features,
            model=suggest_fn(trial),
        )
        pipe = build_pipeline(trial_cfg)
        scores = cross_val_score(pipe, X, y, cv=cv, scoring="f1_macro", n_jobs=1)
        return float(scores.mean())

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=cfg.random_state),
    )
    study.optimize(objective, n_trials=cfg.n_trials, timeout=cfg.timeout)

    best_cfg = PipelineConfig(
        name=f"{base_cfg.name}_tuned",
        preprocessing=base_cfg.preprocessing,
        features=base_cfg.features,
        model={"type": model_type, **study.best_params},
    )

    logger.info(
        "HPO done — best F1-macro %.4f | %s",
        study.best_value,
        study.best_params,
    )

    return best_cfg
