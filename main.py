"""
CLI entrypoint — thin wrapper around the domain modules.

Commands:
    python main.py diagnose  --features data/raw/train_values.csv --labels data/raw/train_labels.csv
    python main.py train     --features ... --labels ...
    python main.py evaluate  --model-stem xgb_baseline_20250619_143022 --features ... --labels ...
    python main.py predict   --model-stem xgb_baseline_20250619_143022 --features data/raw/test_values.csv
    python main.py models
"""
