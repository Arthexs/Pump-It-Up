# Pump It Up — Tanzania Water Pump Failure Prediction

SIA consultancy case: predict which water pumps are functional, need repair, or are non-functional.

## Setup

```bash
# Clone and install (editable mode — imports work from anywhere)
pip install -e ".[dev]"

# Copy env template
cp .env.example .env
# Edit .env with your local data paths

# Download data from DrivenData and place in:
#   data/raw/train_values.csv
#   data/raw/train_labels.csv
#   data/raw/test_values.csv
```

## Usage

```bash
# Diagnose raw data
python main.py diagnose --features data/raw/train_values.csv --labels data/raw/train_labels.csv

# Train a pipeline
python main.py train --features data/raw/train_values.csv --labels data/raw/train_labels.csv

# List saved models
python main.py models

# Evaluate a saved model
python main.py evaluate --model-stem xgb_baseline_20250619_143022 \
  --features data/raw/train_values.csv --labels data/raw/train_labels.csv

# Generate submission
python main.py predict --model-stem xgb_baseline_20250619_143022 \
  --features data/raw/test_values.csv
```

## Structure

```
pump-it-up/
├── main.py                   # CLI (typer)
├── pyproject.toml            # deps, mypy, ruff
├── Dockerfile
├── src/pump/
│   ├── registry.py           # Registry + FnRegistry
│   ├── data.py               # typed loading, splits
│   ├── diagnosis.py          # missing, cardinality, distribution reports
│   ├── preprocessing.py      # registered transformers
│   ├── features.py           # registered engineering + selection steps
│   ├── models.py             # registered estimators
│   ├── store.py              # save / load / list models
│   ├── pipeline.py           # Pipeline.from_config()
│   ├── evaluation.py         # metrics, SHAP, cost matrix
│   ├── visualizations.py     # matplotlib / folium figures
│   └── config/
│       ├── preprocessing.py
│       ├── features.py
│       ├── models.py
│       ├── evaluation.py
│       └── pipeline.py
├── data/
│   ├── raw/                  # original CSVs (gitignored)
│   └── processed/            # cached splits (gitignored)
└── artifacts/
    ├── models/               # versioned .joblib files (gitignored)
    ├── figures/              # exported plots
    └── reports/              # diagnosis + evaluation CSVs
```
