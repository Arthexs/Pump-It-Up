"""
Typed data loading, schema validation, and train/test splitting.

Single entry point for raw data — nothing else reads CSVs directly.
"""

import logging

import pandas as pd

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
