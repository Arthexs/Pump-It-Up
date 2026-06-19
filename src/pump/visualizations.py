"""
Presentation-ready visualizations.

Each function returns a matplotlib Figure (or folium Map for the Tanzania map).
Never saves directly — the caller decides where to put it:

    fig = visualizations.confusion_matrix(cm_df)
    fig.savefig("artifacts/figures/confusion_matrix.png", dpi=150, bbox_inches="tight")
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# ── Diagnosis ─────────────────────────────────────────────────────────────────

def missing_heatmap(
    missing_df: pd.DataFrame,
    top_n: int = 30,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Bar chart of missing value % per column (top_n most missing)."""
    raise NotImplementedError


def label_distribution(
    label_df: pd.DataFrame,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Horizontal bar chart of class distribution, colored by status."""
    raise NotImplementedError


# ── Evaluation ────────────────────────────────────────────────────────────────

def confusion_matrix(
    cm_df: pd.DataFrame,
    normalize: bool = False,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Annotated confusion matrix heatmap."""
    raise NotImplementedError


def shap_importance(
    importance_df: pd.DataFrame,
    top_n: int = 20,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Horizontal bar chart of mean |SHAP| per feature."""
    raise NotImplementedError


# ── Feature insights ──────────────────────────────────────────────────────────

def pump_age_vs_failure(
    X: pd.DataFrame,
    y_raw: pd.Series,
    age_col: str = "pump_age",
    bins: int = 10,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """
    Stacked bar: failure rate by pump age bucket.
    Key business case slide — shows older pumps fail more.
    """
    raise NotImplementedError


def payment_vs_failure(
    X: pd.DataFrame,
    y_raw: pd.Series,
    payment_col: str = "payment",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """
    Grouped bar: failure rate by payment type.
    Key insight: 'never pay' pumps fail at higher rates (governance signal).
    """
    raise NotImplementedError


# ── Tanzania map ──────────────────────────────────────────────────────────────

def tanzania_map(
    X: pd.DataFrame,
    y_raw: pd.Series,
    sample: int = 5000,
    save_path: str | Path | None = None,
) -> "folium.Map":
    """
    Interactive Folium map of pump locations colored by status.
    Call .save("artifacts/figures/map.html") to export for the presentation.
    sample: subsample to keep the HTML file manageable.
    """
    raise NotImplementedError
