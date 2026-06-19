"""
Presentation-ready visualizations.

Each function returns a matplotlib Figure (or folium Map for the Tanzania map).
Never saves directly — the caller decides where to put it:

    fig = visualizations.confusion_matrix(cm_df)
    fig.savefig("artifacts/figures/confusion_matrix.png", dpi=150, bbox_inches="tight")
"""
