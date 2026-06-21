"""
Registered ML estimators.

Thin wrappers around sklearn/XGBoost/LightGBM. Each is constructed from
its Pydantic config and exposes a consistent interface: fit / predict /
predict_proba / feature_importances_.
"""
