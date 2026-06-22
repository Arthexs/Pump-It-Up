"""
Typed data loading, schema validation, and train/test splitting.

Single entry point for raw data — nothing else reads CSVs directly.
"""

import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


def load_features(path: str) -> pd.DataFrame:
    return pd.read_csv(path, index_col="id")


def load_labels(path: str) -> pd.Series:
    return pd.read_csv(path, index_col="id").squeeze(axis="columns")


def validate_alignment(features: pd.DataFrame, labels: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    for name, idx in (("features", features.index), ("labels", labels.index)):
        dupes = idx[idx.duplicated()].unique()
        if len(dupes):
            raise ValueError(f"Duplicate IDs in {name}: {dupes.tolist()}")

    features_only = features.index.difference(labels.index)
    labels_only = labels.index.difference(features.index)
    if len(features_only):
        logger.warning("%d IDs in features but not labels — dropping", len(features_only))
    if len(labels_only):
        logger.warning("%d IDs in labels but not features — dropping", len(labels_only))

    common = features.index.intersection(labels.index)
    return features.loc[common], labels.loc[common]


def load_dataset(features_path: str, labels_path: str) -> tuple[pd.DataFrame, pd.Series]:
    features = load_features(features_path)
    labels = load_labels(labels_path)
    return validate_alignment(features, labels)


def split_dataset(
    features_path: str,
    labels_path: str,
    output_dir: str,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified train/test split; saves train_values, train_labels, test_values, test_labels CSVs."""
    X, y = load_dataset(features_path, labels_path)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    X_train.to_csv(out / "train_values.csv")
    y_train.to_frame().to_csv(out / "train_labels.csv")
    X_test.to_csv(out / "test_values.csv")
    y_test.to_frame().to_csv(out / "test_labels.csv")
    return X_train, X_test, y_train, y_test
