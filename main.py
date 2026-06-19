"""
CLI entrypoint — thin wrapper around the domain modules.

Commands:
    python main.py diagnose  --features data/raw/train_values.csv --labels data/raw/train_labels.csv
    python main.py train     --features ... --labels ...
    python main.py evaluate  --model-stem xgb_baseline_20250619_143022 --features ... --labels ...
    python main.py predict   --model-stem xgb_baseline_20250619_143022 --features data/raw/test_values.csv
    python main.py models
"""

from __future__ import annotations

from pathlib import Path

import typer
from dotenv import load_dotenv

load_dotenv()

app = typer.Typer(help="Tanzania water pump failure prediction")


@app.command()
def diagnose(
    features: Path = typer.Option(...),
    labels: Path = typer.Option(...),
    output_dir: Path = typer.Option(Path("artifacts/reports")),
) -> None:
    """Run data diagnosis and save reports to artifacts/reports/."""
    raise NotImplementedError


@app.command()
def train(
    features: Path = typer.Option(...),
    labels: Path = typer.Option(...),
    pipeline_name: str = typer.Option("xgb_baseline"),
    val_size: float = typer.Option(0.2),
) -> None:
    """Train a pipeline and save the fitted model."""
    raise NotImplementedError


@app.command()
def evaluate(
    model_stem: str = typer.Option(...),
    features: Path = typer.Option(...),
    labels: Path = typer.Option(...),
    output_dir: Path = typer.Option(Path("artifacts/reports")),
) -> None:
    """Load a saved model and run full evaluation."""
    raise NotImplementedError


@app.command()
def predict(
    model_stem: str = typer.Option(...),
    features: Path = typer.Option(...),
    output: Path = typer.Option(Path("artifacts/submission.csv")),
) -> None:
    """Generate a competition submission CSV."""
    raise NotImplementedError


@app.command()
def models() -> None:
    """List all saved models."""
    raise NotImplementedError


if __name__ == "__main__":
    app()
