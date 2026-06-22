"""
CLI entry point — four commands:

    pump diagnose [OPTIONS]
    pump train    [OPTIONS]
    pump evaluate MODEL [OPTIONS]
    pump predict  MODEL [OPTIONS]
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from pump import store
from pump.configs import PipelineConfig, TuningConfig
from pump.data import load_dataset, load_features, split_dataset
from pump.evaluation import cross_val_eval, holdout_eval
from pump.pipeline import build_pipeline

app = typer.Typer(add_completion=False, pretty_exceptions_short=True)

_DEFAULT_FEATURES = "data/raw/training_set_values.csv"
_DEFAULT_LABELS = "data/raw/training_set_labels.csv"
_DEFAULT_TEST = "data/raw/test_set_values.csv"
_DEFAULT_MODELS_DIR = "artifacts/models"

# Sensible baseline used when --config is not supplied.
_BASELINE_CFG = PipelineConfig(
    name="xgb_baseline",
    preprocessing=[
        {"type": "zero_to_nan"},
        {"type": "numeric_imputer"},
        {"type": "categorical_imputer"},
        {"type": "ordinal_encoder"},
    ],
    features=[
        {"type": "redundant_dropper"},
        {"type": "pump_age"},
    ],
    model={"type": "xgb"},
)


@app.command()
def diagnose(
    features: Annotated[
        Path,
        typer.Option("--features", help="Features CSV to diagnose"),
    ] = Path(_DEFAULT_FEATURES),
    labels: Annotated[
        Path | None,
        typer.Option("--labels", help="Labels CSV; include for label distribution"),
    ] = Path(_DEFAULT_LABELS),
    config: Annotated[
        Path | None,
        typer.Option(
            "--config",
            help="JSON PipelineConfig; if supplied, also diagnoses data after preprocessing+features and saves a diff",
        ),
    ] = None,
    output_dir: Annotated[
        Path,
        typer.Option("--output-dir", help="Directory to save CSV reports"),
    ] = Path("artifacts/reports"),
    redundancy_threshold: Annotated[
        float,
        typer.Option("--redundancy-threshold", help="Correlation threshold for redundancy report"),
    ] = 0.95,
) -> None:
    """Diagnose raw data; with --config also diagnoses after preprocessing+features and diffs."""
    from sklearn.pipeline import Pipeline as SkPipeline

    from pump.data import load_features, load_labels
    from pump.diagnosis import diagnose as run_diagnose
    from pump.diagnosis import diff_reports

    typer.echo(f"Loading features: {features}")
    X = load_features(str(features))

    y = None
    if labels is not None and labels.exists():
        y = load_labels(str(labels))
        typer.echo(f"Loading labels:   {labels}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Before report ──────────────────────────────────────────────────────────
    typer.echo(f"\nDiagnosing raw data ({len(X):,} rows, {X.shape[1]} columns) ...")
    before = run_diagnose(X, y, redundancy_threshold=redundancy_threshold)
    _save_report(before, output_dir, prefix="before")
    _print_summary(before)

    # ── After report (only when --config is given) ─────────────────────────────
    if config is not None:
        cfg = PipelineConfig.model_validate_json(config.read_text())

        # Build preprocessing+feature steps only (drop the final estimator).
        full_pipe = build_pipeline(cfg)
        transformer_pipe = SkPipeline(
            [(name, step) for name, step in full_pipe.steps if name != "model"]
        )

        typer.echo(f"\nApplying preprocessing+features from '{cfg.name}' ...")
        X_transformed = transformer_pipe.fit_transform(X, y)
        typer.echo(f"  Shape after transform: {X_transformed.shape}")

        typer.echo("Diagnosing transformed data ...")
        after = run_diagnose(X_transformed, redundancy_threshold=redundancy_threshold)
        _save_report(after, output_dir, prefix="after")

        typer.echo("\nGenerating diff ...")
        diff = diff_reports(before, after)
        for name, df in diff.items():
            path = output_dir / f"diff_{name}.csv"
            df.to_csv(path)
            typer.echo(f"  Saved diff_{name:<18} -> {path}  ({len(df)} rows)")

        # Highlight what changed most.
        typer.echo("\n-- Columns with most missing change after transform --")
        miss_diff = diff["missing"].dropna(subset=["pct_change"])
        miss_diff = (
            miss_diff[miss_diff["pct_change"].abs() > 0]
            .sort_values("pct_change", key=abs, ascending=False)
            .head(10)
        )
        if len(miss_diff):
            for col, row in miss_diff.iterrows():
                typer.echo(
                    f"  {col:<35} {row['pct_missing_before']:.1%} -> {row['pct_missing_after']:.1%}"
                )
        else:
            typer.echo("  (no change)")

        typer.echo("\n-- Encoding suggestions that changed --")
        enc_diff = diff["cardinality"]
        changed = enc_diff[enc_diff.get("encoding_changed", False)]
        if len(changed):
            for col, row in changed.iterrows():
                typer.echo(
                    f"  {col:<35} {row['suggested_encoding_before']} -> {row['suggested_encoding_after']}"
                )
        else:
            typer.echo("  (none)")


def _save_report(report, output_dir: Path, prefix: str) -> None:
    sub_reports = {
        "missing": report.missing,
        "cardinality": report.cardinality,
        "distribution": report.distribution,
        "dtype_audit": report.dtype_audit,
        "redundancy": report.redundancy,
    }
    if report.label_dist is not None:
        sub_reports["label_distribution"] = report.label_dist

    for name, df in sub_reports.items():
        path = output_dir / f"{prefix}_{name}.csv"
        df.to_csv(path)
        typer.echo(f"  Saved {prefix}_{name:<22} -> {path}  ({len(df)} rows)")


def _print_summary(report) -> None:
    typer.echo("\n-- Missing / zero sentinels (top 10) --")
    top_missing = report.missing[report.missing["pct_missing"] > 0].head(10)
    if len(top_missing):
        for col, row in top_missing.iterrows():
            line = f"  {col:<35} missing={row['pct_missing']:.1%}"
            if row["pct_zero"] > 0:
                line += f"  zero={row['pct_zero']:.1%}"
            typer.echo(line)
    else:
        typer.echo("  (none)")

    typer.echo("\n-- High-cardinality columns (suggested: target_encode) --")
    high_card = report.cardinality[report.cardinality["suggested_encoding"] == "target_encode"]
    if len(high_card):
        for col, row in high_card.iterrows():
            typer.echo(f"  {col:<35} n_unique={row['n_unique']}")
    else:
        typer.echo("  (none)")

    typer.echo("\n-- Redundant column pairs --")
    if len(report.redundancy):
        for _, row in report.redundancy.iterrows():
            typer.echo(f"  {row['col_a']} <-> {row['col_b']}  r={row['correlation']:.3f}")
    else:
        typer.echo("  (none above threshold)")

    if report.label_dist is not None:
        typer.echo("\n-- Label distribution --")
        for label, row in report.label_dist.iterrows():
            typer.echo(f"  {label:<30} {row['count']:>6,}  ({row['pct']:.1%})")


@app.command()
def split(
    features: Annotated[
        Path,
        typer.Option("--features", help="Raw features CSV to split"),
    ] = Path(_DEFAULT_FEATURES),
    labels: Annotated[
        Path,
        typer.Option("--labels", help="Raw labels CSV to split"),
    ] = Path(_DEFAULT_LABELS),
    test_size: Annotated[
        float,
        typer.Option("--test-size", help="Fraction of data held out as the test set"),
    ] = 0.2,
    output_dir: Annotated[
        Path,
        typer.Option("--output-dir", help="Directory to write the four split CSVs"),
    ] = Path("data/processed"),
    random_state: Annotated[
        int,
        typer.Option("--random-state"),
    ] = 42,
) -> None:
    """Stratified train/test split of the labeled dataset; saves four CSVs to output_dir."""
    typer.echo(f"Loading: {features} / {labels}")
    X_train, X_test, y_train, y_test = split_dataset(
        str(features),
        str(labels),
        str(output_dir),
        test_size=test_size,
        random_state=random_state,
    )
    typer.echo(f"  Train: {len(X_train):,} rows")
    typer.echo(f"  Test:  {len(X_test):,} rows  ({test_size:.0%} holdout)")
    typer.echo("\n-- Class distribution (train / test) --")
    for cls in sorted(y_train.unique()):
        n_tr = (y_train == cls).sum()
        n_te = (y_test == cls).sum()
        typer.echo(
            f"  {cls:<30} {n_tr:>6,} ({n_tr / len(y_train):.1%}) / {n_te:>5,} ({n_te / len(y_test):.1%})"
        )
    typer.echo(f"\nSaved to {output_dir}/")
    typer.echo("  train_values.csv, train_labels.csv")
    typer.echo("  test_values.csv,  test_labels.csv")
    typer.echo("\nNext steps:")
    typer.echo(
        f"  pump train    --features {output_dir}/train_values.csv --labels {output_dir}/train_labels.csv --config configs/xgb.json"
    )
    typer.echo(
        f"  pump evaluate MODEL --features {output_dir}/train_values.csv --labels {output_dir}/train_labels.csv"
    )
    typer.echo(
        f"  pump evaluate MODEL --features {output_dir}/test_values.csv  --labels {output_dir}/test_labels.csv --holdout"
    )


def _resolve_pipeline(model: str, *, latest: bool, models_dir: Path):
    """Load a pipeline from store, or exit with a clean error message."""
    try:
        if latest:
            return store.load_latest(model, models_dir=models_dir)
        return store.load(model, models_dir=models_dir)
    except FileNotFoundError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc


@app.command()
def train(
    features: Annotated[
        Path,
        typer.Option("--features", help="Training features CSV"),
    ] = Path(_DEFAULT_FEATURES),
    labels: Annotated[
        Path,
        typer.Option("--labels", help="Training labels CSV"),
    ] = Path(_DEFAULT_LABELS),
    config: Annotated[
        Path | None,
        typer.Option("--config", help="JSON file with PipelineConfig; omit to use XGB baseline"),
    ] = None,
    name: Annotated[
        str | None,
        typer.Option("--name", help="Override the pipeline name used as the saved-model stem"),
    ] = None,
    optimize: Annotated[
        bool,
        typer.Option("--optimize/--no-optimize", help="Run Optuna HPO before fitting"),
    ] = False,
    n_trials: Annotated[
        int,
        typer.Option("--n-trials", help="Optuna trials (only with --optimize)"),
    ] = 40,
    cv_folds: Annotated[
        int,
        typer.Option("--cv-folds", help="CV folds for HPO (only with --optimize)"),
    ] = 3,
    models_dir: Annotated[
        Path,
        typer.Option("--models-dir", help="Directory in which to save the trained model"),
    ] = Path(_DEFAULT_MODELS_DIR),
) -> None:
    """Load data, optionally tune hyperparameters, fit a pipeline, and save it."""
    if config is not None:
        cfg = PipelineConfig.model_validate_json(config.read_text())
    else:
        cfg = _BASELINE_CFG

    if name is not None:
        cfg = cfg.model_copy(update={"name": name})

    if cfg.model is None:
        typer.echo("Error: config has no model — use a full pipeline config for train.", err=True)
        raise typer.Exit(1)

    typer.echo(f"Loading data: {features} / {labels}")
    X, y = load_dataset(str(features), str(labels))
    typer.echo(f"  {len(X):,} rows, {X.shape[1]} features, {y.nunique()} classes")

    if optimize:
        from pump.tuning import optimize as run_hpo

        typer.echo(f"Tuning: {n_trials} trials · {cv_folds}-fold CV")
        cfg = run_hpo(cfg, X, y, TuningConfig(n_trials=n_trials, cv_folds=cv_folds))
        typer.echo(f"  Best config: {cfg.name}")

    typer.echo(f"Fitting '{cfg.name}' ...")
    pipe = build_pipeline(cfg)
    pipe.fit(X, y)

    path = store.save(pipe, cfg.name, models_dir=models_dir)
    typer.echo(f"Saved → {path}")


@app.command()
def evaluate(
    model: Annotated[
        str,
        typer.Argument(help="Full model stem, or name when --latest is set"),
    ],
    latest: Annotated[
        bool,
        typer.Option(
            "--latest/--no-latest", help="Load the most recent model whose name matches MODEL"
        ),
    ] = False,
    features: Annotated[
        Path,
        typer.Option("--features", help="Features CSV"),
    ] = Path(_DEFAULT_FEATURES),
    labels: Annotated[
        Path,
        typer.Option("--labels", help="Labels CSV"),
    ] = Path(_DEFAULT_LABELS),
    n_splits: Annotated[
        int,
        typer.Option("--n-splits", help="Stratified-KFold splits (ignored with --holdout)"),
    ] = 5,
    holdout: Annotated[
        bool,
        typer.Option(
            "--holdout/--no-holdout",
            help="Score the already-fitted pipeline directly on the provided data (no CV refitting)",
        ),
    ] = False,
    models_dir: Annotated[
        Path,
        typer.Option("--models-dir"),
    ] = Path(_DEFAULT_MODELS_DIR),
) -> None:
    """Cross-validate a saved pipeline, or score it directly on a held-out set with --holdout."""
    pipe = _resolve_pipeline(model, latest=latest, models_dir=models_dir)

    typer.echo(f"Loading data: {features} / {labels}")
    X, y = load_dataset(str(features), str(labels))
    typer.echo(f"  {len(X):,} rows")

    if holdout:
        typer.echo("Scoring on held-out set ...")
        results = holdout_eval(pipe, X, y)
        typer.echo("\n-- Holdout results --")
        for key in sorted(results):
            typer.echo(f"  {key:<24} {results[key]:.4f}")
    else:
        typer.echo(f"Running {n_splits}-fold cross-validation ...")
        results = cross_val_eval(pipe, X, y, n_splits=n_splits)
        typer.echo("\n-- Cross-validation results --")
        for key in sorted(results):
            typer.echo(f"  {key:<24} {results[key]:.4f}")


@app.command()
def predict(
    model: Annotated[
        str,
        typer.Argument(help="Full model stem, or name when --latest is set"),
    ],
    latest: Annotated[
        bool,
        typer.Option(
            "--latest/--no-latest", help="Load the most recent model whose name matches MODEL"
        ),
    ] = False,
    features: Annotated[
        Path,
        typer.Option("--features", help="Test features CSV"),
    ] = Path(_DEFAULT_TEST),
    output: Annotated[
        Path | None,
        typer.Option(
            "--output", help="Output CSV (default: artifacts/submissions/submission_{model}.csv)"
        ),
    ] = None,
    models_dir: Annotated[
        Path,
        typer.Option("--models-dir"),
    ] = Path(_DEFAULT_MODELS_DIR),
) -> None:
    """Generate predictions on the test set and write a competition-ready submission CSV."""
    import pandas as pd

    pipe = _resolve_pipeline(model, latest=latest, models_dir=models_dir)

    typer.echo(f"Loading test features: {features}")
    X_test = load_features(str(features))
    typer.echo(f"  {len(X_test):,} rows, {X_test.shape[1]} features")

    typer.echo("Generating predictions ...")
    preds = pipe.predict(X_test)

    if output is None:
        out_dir = Path("artifacts/submissions")
        out_dir.mkdir(parents=True, exist_ok=True)
        output = out_dir / f"submission_{model}.csv"
    else:
        output.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame({"id": X_test.index, "status_group": preds}).to_csv(output, index=False)
    typer.echo(f"Submission saved → {output}  ({len(X_test):,} rows)")
