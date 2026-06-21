"""Tests for pump.data — data loading, validation, and alignment."""

import logging

import pandas as pd
import pytest

from pump.data import load_dataset, load_features, load_labels, validate_alignment


def _make_features(ids: list, cols: list[str] | None = None) -> pd.DataFrame:
    cols = cols or ["a", "b"]
    return pd.DataFrame(
        {c: range(len(ids)) for c in cols},
        index=pd.Index(ids, name="id"),
    )


def _make_labels(ids: list) -> pd.Series:
    return pd.Series(
        range(len(ids)),
        index=pd.Index(ids, name="id"),
        name="label",
    )


# ── load_features ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "csv_content, expected_cols",
    [
        ("id,a,b\n1,10,20\n2,30,40", ["a", "b"]),
        ("id,x\n5,99", ["x"]),
    ],
)
def test_load_features(tmp_path, csv_content, expected_cols):
    p = tmp_path / "features.csv"
    p.write_text(csv_content)
    df = load_features(str(p))
    assert isinstance(df, pd.DataFrame)
    assert df.index.name == "id"
    assert list(df.columns) == expected_cols


# ── load_labels ───────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "csv_content, expected_name, expected_values",
    [
        ("id,status\n1,functional\n2,non functional", "status", ["functional", "non functional"]),
        ("id,label\n10,a", "label", ["a"]),
    ],
)
def test_load_labels(tmp_path, csv_content, expected_name, expected_values):
    p = tmp_path / "labels.csv"
    p.write_text(csv_content)
    s = load_labels(str(p))
    assert isinstance(s, pd.Series)
    assert s.index.name == "id"
    assert s.name == expected_name
    assert list(s) == expected_values


# ── validate_alignment ────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "feature_ids, label_ids, expected_common, expect_warnings",
    [
        # happy path — all IDs match
        ([1, 2, 3], [1, 2, 3], [1, 2, 3], 0),
        # extra IDs in features only
        ([1, 2, 3], [1, 2], [1, 2], 1),
        # extra IDs in labels only
        ([1, 2], [1, 2, 3], [1, 2], 1),
        # both sides have extra IDs
        ([1, 2, 3], [2, 3, 4], [2, 3], 2),
    ],
)
def test_validate_alignment_common_ids(
    feature_ids, label_ids, expected_common, expect_warnings, caplog
):
    features = _make_features(feature_ids)
    labels = _make_labels(label_ids)
    with caplog.at_level(logging.WARNING, logger="pump.data"):
        X, y = validate_alignment(features, labels)
    assert list(X.index) == expected_common
    assert list(y.index) == expected_common
    assert len(caplog.records) == expect_warnings


@pytest.mark.parametrize(
    "duplicate_in, feature_ids, label_ids",
    [
        ("features", [1, 1, 2], [1, 2]),
        ("labels", [1, 2], [2, 2, 1]),
    ],
)
def test_validate_alignment_duplicate_ids_raises(duplicate_in, feature_ids, label_ids):
    features = _make_features(feature_ids)
    labels = _make_labels(label_ids)
    with pytest.raises(ValueError, match=f"Duplicate IDs in {duplicate_in}"):
        validate_alignment(features, labels)


# ── load_dataset ──────────────────────────────────────────────────────────────


def test_load_dataset_happy_path(tmp_path):
    (tmp_path / "features.csv").write_text("id,a,b\n1,10,20\n2,30,40\n3,50,60")
    (tmp_path / "labels.csv").write_text("id,status\n1,functional\n2,broken\n3,functional")
    X, y = load_dataset(str(tmp_path / "features.csv"), str(tmp_path / "labels.csv"))
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert list(X.index) == [1, 2, 3]
    assert list(y) == ["functional", "broken", "functional"]
    assert "status" not in X.columns


def test_load_dataset_drops_unmatched(tmp_path, caplog):
    (tmp_path / "features.csv").write_text("id,a\n1,10\n2,20\n3,30")
    (tmp_path / "labels.csv").write_text("id,status\n1,functional\n2,broken")
    with caplog.at_level(logging.WARNING, logger="pump.data"):
        X, y = load_dataset(str(tmp_path / "features.csv"), str(tmp_path / "labels.csv"))
    assert list(X.index) == [1, 2]
    assert len(caplog.records) == 1
