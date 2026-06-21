"""
CLI entrypoint — thin wrapper around the domain modules.

Commands:
    python main.py diagnose  --features data/raw/training_set_values.csv --labels data/raw/training_set_labels.csv
    python main.py train     --features ... --labels ...
    python main.py evaluate  --model-stem xgb_baseline_20250619_143022 --features ... --labels ...
    python main.py predict   --model-stem xgb_baseline_20250619_143022 --features data/raw/test_set_values.csv
    python main.py models
"""

import typer

from pump.data import load_dataset
from pump.diagnosis import diagnose as run_diagnosis

app = typer.Typer()

_DEFAULT_FEATURES = "data/raw/training_set_values.csv"
_DEFAULT_LABELS = "data/raw/training_set_labels.csv"


@app.command()
def diagnose(
    features: str = typer.Option(_DEFAULT_FEATURES, help="Path to features CSV"),
    labels: str = typer.Option(_DEFAULT_LABELS, help="Path to labels CSV"),
) -> None:
    """Load data and print a full diagnosis report."""
    typer.echo(f"Loading data from:\n  {features}\n  {labels}\n")
    X, y = load_dataset(features, labels)
    typer.echo(f"Loaded: {X.shape[0]:,} rows, {X.shape[1]} features\n")

    report = run_diagnosis(X, y)
    typer.echo(report)

    missing = report.missing[report.missing["pct_missing"] > 0]
    if not missing.empty:
        typer.echo(f"\n--- Missing values ({len(missing)} columns) ---")
        typer.echo(missing.to_string())

    flagged = report.dtype_audit[report.dtype_audit["flagged"]]
    if not flagged.empty:
        typer.echo(f"\n--- Dtype issues ({len(flagged)} columns) ---")
        typer.echo(flagged.to_string())

    typer.echo("\n--- Label distribution ---")
    typer.echo(report.label_dist.to_string())


@app.command()
def train(
    features: str = typer.Option(_DEFAULT_FEATURES, help="Path to features CSV"),
    labels: str = typer.Option(_DEFAULT_LABELS, help="Path to labels CSV"),
) -> None:
    """Train the full pipeline."""
    raise NotImplementedError


if __name__ == "__main__":
    import sys

    if len(sys.argv) == 1:
        sys.argv.append("diagnose")
    app()
