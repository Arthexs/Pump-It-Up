"""
Presentation-ready visualizations.

Each function returns a matplotlib Figure (or folium Map for the Tanzania map).
Never saves directly — the caller decides where to put it:

    fig = confusion_matrix_heatmap(cm_df)
    fig.savefig("artifacts/figures/confusion_matrix.png", dpi=150, bbox_inches="tight")
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure

from pump.configs import CostMatrixConfig, EvaluationConfig

if TYPE_CHECKING:
    import folium

# Switch to non-interactive backend after pyplot import so figures are never shown on screen.
plt.switch_backend("Agg")

_CLASS_ORDER = ["functional", "functional needs repair", "non functional"]
_CLASS_COLORS = ["#2196F3", "#FF9800", "#F44336"]  # blue, orange, red


def class_distribution(y: pd.Series) -> Figure:
    """Bar chart of class counts and percentages."""
    counts = y.value_counts().reindex(_CLASS_ORDER).dropna()
    total = len(y)

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(counts.index, counts.values, color=_CLASS_COLORS[: len(counts)])

    for bar, val in zip(bars, counts.values, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + total * 0.005,
            f"{int(val):,}\n({val / total:.1%})",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_title("Label Distribution")
    ax.set_ylabel("Count")
    ax.set_ylim(0, counts.max() * 1.22)
    ax.tick_params(axis="x", labelrotation=10)
    fig.tight_layout()
    return fig


def model_comparison(results: dict[str, dict[str, float]]) -> Figure:
    """Grouped bar chart of accuracy and F1-macro across models."""
    models = list(results.keys())
    metrics = [("accuracy", "Accuracy"), ("f1_macro", "F1-macro")]

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(6, len(models) * 2), 5))
    for i, (key, label) in enumerate(metrics):
        vals = [results[m][key] for m in models]
        offset = (i - len(metrics) / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=label)
        for bar, val in zip(bars, vals, strict=True):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.004,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_title("Model Comparison")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.1)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def per_class_metrics(
    per_class: dict[str, dict[str, float]],
    *,
    model_name: str = "",
) -> Figure:
    """Grouped bar chart of precision/recall/F1 per class for one model."""
    classes = [c for c in _CLASS_ORDER if c in per_class]
    metric_keys = ["precision", "recall", "f1"]
    metric_labels = ["Precision", "Recall", "F1"]

    x = np.arange(len(classes))
    width = 0.25

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, (key, label) in enumerate(zip(metric_keys, metric_labels, strict=True)):
        vals = [per_class[c][key] for c in classes]
        offset = (i - 1) * width
        ax.bar(x + offset, vals, width, label=label)

    title = f"Per-class Metrics -- {model_name}" if model_name else "Per-class Metrics"
    ax.set_title(title)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.1)
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=10, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def confusion_matrix_heatmap(cm_df: pd.DataFrame, *, model_name: str = "") -> Figure:
    """Normalised confusion matrix heatmap (rows=true, cols=predicted)."""
    cm_norm = cm_df.div(cm_df.sum(axis=1), axis=0)

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".1%",
        cmap="Blues",
        vmin=0,
        vmax=1,
        ax=ax,
        linewidths=0.5,
    )
    title = f"Confusion Matrix -- {model_name}" if model_name else "Confusion Matrix"
    ax.set_title(title)
    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    fig.tight_layout()
    return fig


def cost_matrix_heatmap(
    cost_cfg: CostMatrixConfig | None = None,
    eval_cfg: EvaluationConfig | None = None,
) -> Figure:
    """Heatmap of the asymmetric misclassification cost matrix."""
    ccfg = cost_cfg if cost_cfg is not None else CostMatrixConfig()
    ecfg = eval_cfg if eval_cfg is not None else EvaluationConfig()

    df = pd.DataFrame(ccfg.matrix, index=ecfg.labels, columns=ecfg.labels)

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(
        df,
        annot=True,
        fmt=".0f",
        cmap="Reds",
        vmin=0,
        ax=ax,
        linewidths=0.5,
    )
    ax.set_title("Business Cost Matrix\n(row = true class, col = predicted class)")
    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    fig.tight_layout()
    return fig


def feature_importance(
    pipeline,
    *,
    top_n: int = 20,
    model_name: str = "",
) -> Figure:
    """Horizontal bar chart of top-N feature importances from the pipeline's model step."""
    model_step = pipeline.named_steps.get("model")
    if model_step is None:
        raise ValueError("Pipeline has no 'model' step")

    importances: np.ndarray = model_step.feature_importances_

    # Resolve feature names from the inner sklearn / XGBoost / LightGBM model.
    # All transformers return DataFrames, so feature_names_in_ / feature_name_ are set.
    inner = getattr(model_step, "_model", model_step)
    names: list[str] | None = None
    for attr in ("feature_name_", "feature_names_in_"):
        candidate = getattr(inner, attr, None)
        if candidate is not None:
            names = list(candidate)
            break
    if names is None:
        names = [f"feature_{i}" for i in range(len(importances))]

    df = (
        pd.DataFrame({"feature": names, "importance": importances})
        .sort_values("importance", ascending=False)
        .head(top_n)
    )

    fig, ax = plt.subplots(figsize=(8, max(4, top_n * 0.35)))
    ax.barh(df["feature"][::-1], df["importance"][::-1], color="#2196F3")
    title = (
        f"Feature Importance (top {top_n}) -- {model_name}"
        if model_name
        else f"Feature Importance (top {top_n})"
    )
    ax.set_title(title)
    ax.set_xlabel("Importance")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    return fig


def pump_map(
    X: pd.DataFrame,
    y_pred: np.ndarray,
    *,
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    max_points: int = 5_000,
) -> folium.Map:
    """Folium map of Tanzania pump locations coloured by predicted status."""
    import folium as _folium

    color_map = {
        "functional": "#2196F3",
        "functional needs repair": "#FF9800",
        "non functional": "#F44336",
    }

    df = X[[lat_col, lon_col]].copy()
    df["pred"] = y_pred
    # Drop zero-coordinate rows (sentinel for missing in this dataset)
    df = df[(df[lat_col] != 0) & (df[lon_col] != 0)].dropna(subset=[lat_col, lon_col])

    if len(df) > max_points:
        df = df.sample(max_points, random_state=42)

    df = df.rename(columns={lat_col: "lat", lon_col: "lon"})

    m = _folium.Map(location=[-6.0, 35.0], zoom_start=6, tiles="CartoDB positron")

    for row in df.itertuples():
        _folium.CircleMarker(
            location=[row.lat, row.lon],
            radius=3,
            color=color_map.get(str(row.pred), "#999999"),
            fill=True,
            fill_opacity=0.6,
            weight=0,
            popup=str(row.pred),
        ).add_to(m)

    legend_html = """
    <div style="position:fixed;bottom:30px;left:30px;z-index:1000;
                background:white;padding:10px;border-radius:5px;font-size:13px;">
        <b>Pump status</b><br>
        <span style="color:#2196F3;">&#9679;</span> Functional<br>
        <span style="color:#FF9800;">&#9679;</span> Needs repair<br>
        <span style="color:#F44336;">&#9679;</span> Non-functional
    </div>
    """
    m.get_root().html.add_child(_folium.Element(legend_html))
    return m
