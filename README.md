# Pump It Up — Tanzania Water Pump Failure Prediction

SIA consultancy case: predict which water pumps are functional, need repair, or non-functional using the DrivenData competition dataset.

## Setup

```bash
# Create and activate the conda environment
conda env create -f environment.yml
conda activate pump

# Download data from DrivenData and place in:
#   data/raw/training_set_values.csv
#   data/raw/training_set_labels.csv
#   data/raw/test_set_values.csv
```

The package installs in editable mode automatically via `environment.yml` (`pip install -e .`), so the `pump` CLI is available as soon as the environment is active.

## Workflow

```bash
# 1. Split labeled data into train/test once (stratified 80/20)
pump split

# 2. Inspect raw data quality before doing anything
pump diagnose

# 3. Optionally inspect the effect of your preprocessing pipeline
pump diagnose --config configs/preprocessing.json

# 4. Train a model on the train split
pump train --features data/processed/train_values.csv \
           --labels data/processed/train_labels.csv \
           --config configs/xgb.json

# 5. CV evaluate on the train split (model selection / tuning)
pump evaluate xgb --latest \
  --features data/processed/train_values.csv \
  --labels data/processed/train_labels.csv

# 6. Final unbiased score on the held-out test split (once, for the report)
pump evaluate xgb --latest \
  --features data/processed/test_values.csv \
  --labels data/processed/test_labels.csv \
  --holdout

# 7. Generate competition submission from the unlabeled test set
pump predict xgb --latest
```

## CLI

All commands run under `pump`:

```bash
pump --help
```

### Split

Stratified train/test split of the labeled dataset. Run this once before training.

```bash
pump split                        # 80/20 split, saves to data/processed/
pump split --test-size 0.15       # custom holdout fraction
```

Saves four CSVs to `data/processed/`: `train_values.csv`, `train_labels.csv`, `test_values.csv`, `test_labels.csv`. Prints class distribution to confirm stratification.

### Diagnose

Inspect raw data quality — missing values, cardinality, redundant columns, and label distribution.

```bash
pump diagnose
```

Compare raw data against what it looks like after a preprocessing pipeline runs:

```bash
pump diagnose --config configs/preprocessing.json
```

Outputs CSV reports to `artifacts/reports/`:

- `before_missing.csv`, `before_cardinality.csv`, `before_distribution.csv`, `before_dtype_audit.csv`, `before_redundancy.csv`, `before_label_distribution.csv`
- `after_*` and `diff_*` variants when `--config` is supplied

### Train

Fit a pipeline on the train split and save the model:

```bash
# XGB baseline (no config needed)
pump train --features data/processed/train_values.csv --labels data/processed/train_labels.csv

# Use a specific config
pump train --features data/processed/train_values.csv --labels data/processed/train_labels.csv \
           --config configs/xgb.json

# With Optuna hyperparameter optimisation (40 trials, 3-fold CV)
pump train --config configs/xgb.json --optimize

# Custom trial budget
pump train --config configs/lgbm.json --optimize --n-trials 60 --cv-folds 5

# Override the saved model name
pump train --config configs/xgb.json --name my_xgb_v2
```

Models are saved to `artifacts/models/<name>_<timestamp>.joblib`.

### Evaluate

Cross-validate on the train split, or score directly on the held-out test split:

```bash
# CV on train split (model selection)
pump evaluate xgb --latest \
  --features data/processed/train_values.csv --labels data/processed/train_labels.csv

# Final holdout score on test split — uses the already-fitted pipeline, no refitting
pump evaluate xgb --latest \
  --features data/processed/test_values.csv --labels data/processed/test_labels.csv \
  --holdout

# By full stem instead of --latest
pump evaluate xgb_20250622_143022 --holdout \
  --features data/processed/test_values.csv --labels data/processed/test_labels.csv

# Change number of CV folds
pump evaluate xgb --latest --n-splits 10 \
  --features data/processed/train_values.csv --labels data/processed/train_labels.csv
```

### Visualize

Generate all charts and the interactive pump map for one or more trained models:

```bash
# All four models at once (pass multiple names as positional args)
pump visualize xgb lgbm random_forest logistic_regression --latest

# Single model, custom figure directory
pump visualize xgb --latest --output-dir artifacts/figures/run1

# Change the number of features shown in the importance chart
pump visualize xgb --latest --top-n 30
```

Reads test data from `data/processed/test_values.csv` / `test_labels.csv` by default.

Saves to `artifacts/figures/`:

| File | Description |
| --- | --- |
| `class_distribution.png` | Bar chart of training-set label counts and percentages |
| `cost_matrix.png` | Asymmetric business cost matrix heatmap |
| `model_comparison.png` | Grouped bar chart of accuracy and F1-macro across all models |
| `confusion_matrix_<name>.png` | Normalised confusion matrix per model |
| `per_class_<name>.png` | Precision / recall / F1 per class per model |
| `feature_importance_<name>.png` | Top-N feature importances per model |
| `pump_map_<name>.html` | Interactive Folium map of Tanzania pump locations coloured by predicted status |

### Predict

Generate a competition-ready submission CSV from the unlabeled test set:

```bash
pump predict xgb_20250622_143022

# Or by name
pump predict xgb --latest

# Custom output path
pump predict xgb --latest --output submissions/my_submission.csv
```

Submissions are saved to `artifacts/submissions/submission_<model>.csv` by default.

## Configs

`configs/` contains ready-made `PipelineConfig` JSON files:

| File | Model | HPO support |
| --- | --- | --- |
| `xgb.json` | XGBoost | yes |
| `lgbm.json` | LightGBM | yes |
| `random_forest.json` | Random Forest | yes |
| `logistic_regression.json` | Logistic Regression | no |
| `preprocessing.json` | no model (diagnose only) | — |

Edit these freely to change preprocessing steps, feature engineering, or hyperparameters. The `preprocessing.json` config has no `model` key and is intended for use with `pump diagnose --config` to inspect the effect of your pipeline on the data before training.

## Docker

```bash
docker build -t pump-it-up .

# Mount your local data and artifacts directories at runtime
docker run --rm \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/artifacts:/app/artifacts" \
  -v "$(pwd)/configs:/app/configs" \
  pump-it-up train --config configs/xgb.json \
    --features data/processed/train_values.csv \
    --labels data/processed/train_labels.csv
```

## Project Structure

```text
pump-it-up/
├── configs/                  # PipelineConfig JSON files
│   ├── xgb.json
│   ├── lgbm.json
│   ├── random_forest.json
│   ├── logistic_regression.json
│   └── preprocessing.json    # preprocessing-only (for diagnose --config)
├── src/pump/
│   ├── main.py               # CLI (pump diagnose / train / evaluate / predict)
│   ├── configs.py            # Pydantic config models
│   ├── pipeline.py           # build_pipeline() from PipelineConfig
│   ├── registry.py           # TRANSFORMERS / SELECTORS / ESTIMATORS registries
│   ├── data.py               # typed CSV loading
│   ├── diagnosis.py          # DiagnosisReport, diagnose(), diff_reports()
│   ├── preprocessing.py      # registered transformers
│   ├── features.py           # registered feature engineering + selection steps
│   ├── models.py             # registered estimators
│   ├── store.py              # save / load / load_latest
│   ├── evaluation.py         # cross_val_eval, cost matrix
│   ├── tuning.py             # Optuna HPO
│   └── visualizations.py     # matplotlib / folium figures
├── tests/
├── data/
│   ├── raw/                  # original CSVs (gitignored)
│   └── processed/            # cached splits (gitignored)
├── artifacts/
│   ├── models/               # versioned .joblib files (gitignored)
│   ├── reports/              # diagnosis CSVs
│   ├── figures/              # charts and maps from pump visualize
│   └── submissions/          # prediction CSVs
├── environment.yml
└── pyproject.toml
```

## Registered Pipeline Steps

### Preprocessing

| Type | Description |
| --- | --- |
| `zero_to_nan` | Replace zero sentinels with NaN before imputation |
| `numeric_imputer` | Impute numeric columns (default: median) |
| `categorical_imputer` | Impute categorical columns (default: most_frequent) |
| `ordinal_encoder` | Encode all object columns as integers |
| `target_encoder` | Mean-target encode high-cardinality columns |

### Feature engineering / selection

| Type | Description |
| --- | --- |
| `redundant_dropper` | Drop known redundant/duplicate columns |
| `pump_age` | Derive pump age from `construction_year` and `date_recorded` |
| `geo_cluster` | KMeans cluster on lat/lon (default: 20 clusters) |
| `frequency_encoder` | Replace categories with training-set frequency |
| `variance_threshold` | Drop low-variance features |
| `correlation_threshold` | Drop highly correlated feature pairs |
| `mutual_info` | Keep top-k features by mutual information with target |
| `xgb_importance` | Keep features above an XGBoost importance percentile |

### Models

| Type | Notes |
| --- | --- |
| `xgb` | XGBoost — primary model |
| `lgbm` | LightGBM |
| `random_forest` | sklearn RandomForest |
| `logistic_regression` | Baseline; does not support `--optimize` |
